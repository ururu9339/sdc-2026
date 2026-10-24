# Deploying the API on Replit

The backend runs on Replit with the committed **serving bundle** (`serving/`). The Replit copy has no
`data/` folder (it is gitignored), so the API reads the bundle. The bundle is a snapshot written from your local
data by the refresh command. It is not the data, and it is not recomputed on Replit.

## 1. Run command

The repository's `.replit` sets everything up: Python 3.12 and Node 24, Tesseract and Poppler for CV OCR, a
deployment build that installs `requirements.txt` and builds `frontend/` with pnpm 10, and this run command:

```bash
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

FastAPI serves the built website from `frontend/dist` and the API under `/api/v1` on the same origin, so the site
needs no `CORS_ORIGINS`. Port 8000 is mapped to the public port 80. The **Run** button builds the frontend on first
use; a deployment builds it during its build step. Install with `pip install -r requirements.txt`. The health check is `GET /api/v1/health` (always HTTP 200; the body
says `ready`, the `feed` being served, and `data.data_layout`). On Replit it should read `data_layout: "bundle"`.
If it reads `repo` or `env`, the deployment is not using the bundle; see section 5.

## 2. Environment variables

| Variable | Needed? | Meaning |
|---|---|---|
| `CORS_ORIGINS` | recommended | Comma-separated frontend origins, for example `https://my-frontend.replit.app,https://localhost:5173`. Default `*`, which allows any origin but never credentials. |
| `DATABASE_URL` | optional | Enables the Qwen/pgvector matcher for `POST /report` (`engine: auto`). Without it the offline matcher is used and `warnings` says so. Also reported by `/health`. |
| `SDC_DATA_DIR` | leave unset | Forces another data directory. Unset means "use `serving/` on Replit". |
| `SDC_MODELS_DIR` | leave unset | Forces another models directory. |
| `PREVIEW_FORECASTS` | optional | `false` hides the preview feed (then forecasts are `feed: none`). Default `true`. |
| `GATE_PROFILE` | optional | `demo` (default) or `production`. Affects only the gate of a refresh run, not what is served. |

The candidate database is a local SQLite file (`sdc2026.db`) created at server start. It is not restored after a
redeploy, so candidate records do not survive a redeploy. Do not rely on them there.

## 3. Refreshing the bundle (on your own machine)

Refresh needs the local `data/` folder, so it runs on your machine, not on Replit. It refuses to run from the bundle.

```bash
python3 -m src.pipeline.refresh --dry-run        # check inputs and print the plan; writes nothing
python3 -m src.pipeline.refresh                   # one agent-loop pass, then export, then write serving/
python3 -m src.pipeline.refresh --as-of 2023-09-11   # optional: a specific origin date
```

A refresh does, in order: take `data/refresh.lock` (a stale lock from a dead process is taken over); run one pass of
the agent loop (perceive, decide, retrain if needed, forecast, reconcile, log); write
`data/handoff/entity_forecast_<feed>.csv`; write `serving/` atomically (the previous bundle stays until the new one
is complete and under the size limit); write `data/last_refresh.json` with what ran, the gate verdict, record counts
before and after, timings and the manifest hash.

Exit codes: `0` success (including "no model yet", which is a valid answer), `1` failure (no data, feature mismatch
or an exception; nothing is published), `2` refused (another refresh holds the lock, or the run is in the bundle).

Running it twice for the same `--as-of` adds no forecast or outcome records. Check the `records added` line.

Then publish:

```bash
git add serving/
git commit -m "Refresh serving bundle"
git push
```

Replit pulls the repository; restart or redeploy the app so it reads the new bundle. Check
`/api/v1/health` (the `bundle.generated_at` and `bundle.as_of` should be the new values).

Optional, after a refresh, to rebuild only the bundle from the current local state:

```bash
python3 -m src.pipeline.export_bundle            # writes <repo>/serving
```

## 4. What the bundle is, and is not

The bundle is a **snapshot** of what the API reads, produced by the refresh:

- `serving/manifest.json`: generated time, origin date (`as_of`), feed, model version, `validated`, row counts,
  outcome summary, and every file with its size and sha256.
- `serving/preview_log.jsonl` (or `forecast_log.jsonl` when a live model exists): only the newest origin's records,
  re-chained. The hash chain verifies, and the original `logged_at` timestamps are kept.
- `serving/processed/job_skill_graph.csv` and `observations.csv` (the target source, plus the enabled leading
  sources that have data).
- `serving/raw/<source>.csv`: the best-tier leading source's raw file, for `/snapshot`.
- `serving/model_report.json` and `serving/models/version_log.json` (the last five attempts, plus the live and
  preview entries).
- `serving/models/forecaster_<v>/meta.json`, only when a live model exists.

It is **not**: the full data, the forecast history, the models' weights, the candidate database, CVs, or any
secret. It is never computed on Replit, and it is never more recent than your last refresh.

**Size limit.** Everything in `serving/`, manifest included, must be under 5 MiB (5,242,880 bytes). A refresh that
exceeds it fails loudly, and the previous bundle stays in place.

## 5. How the API chooses its data

`src/training/config.py` (`resolve_layout`) picks the data directory:

1. `SDC_DATA_DIR` is set: that directory (mode `env`).
2. `SDC_DATA_DIR` is unset, the repo has no forecast or preview log in `data/`, and `serving/manifest.json` exists:
   `serving/`, with `serving/models` for models (mode `bundle`). This is the Replit case.
3. Otherwise: the repo's `data/` and `src/models/` (mode `repo`), your local working state.

`/api/v1/health` reports the mode as `data.data_layout`. Mode `repo` on Replit would mean a `data/` folder was
present in the deployment, which is not expected: the folder is gitignored.

## 6. Checks before you publish a bundle

- `python3 -m pytest -q tests` passes.
- `/api/v1/health` shows `ready: true` and the expected `feed`.
- `/api/v1/forecast` on a preview feed returns `validated: false` and the banner. Never publish a bundle that shows
  a number without the banner.
- The bundle size is printed by refresh (`total_bytes`), and is under 5 MiB.

## 7. Known limits

- A refresh is a manual step on your machine. The bundle is only as recent as your last refresh.
- The Qwen/pgvector path needs `DATABASE_URL`, Postgres with pgvector, psycopg and Ollama. It is not verified here; the
  offline matcher is what the bundle deployment runs.
- The OCR path (`POST /api/v1/candidates/pdf`) needs Tesseract on the server; without it the route returns 503.

## 8. Hand-off to the embeddings matcher

`python3 -m src.pipeline.export_forecasts --feed preview --csv rows.csv` (or `refresh`, which writes
`data/handoff/entity_forecast_<feed>.csv`) produces rows for the `entity_forecast` table. Copies for the repo are in
`handoff/`.

- `entity_id` is a taxonomy id (`skill:python`, `job:accountant`); ids outside the taxonomy are skipped.
- `horizon_months` is 3 (our 12-week horizon). Call the matcher with `--horizon 3`.
- `growth` is `exp(p50) - 1`, the forecast relative change in the smoothed share of postings.
- `model_version` is `vN` for a validated model and `preview:vN` for an unvalidated preview.
- The matcher takes the newest row per entity regardless of version, so preview and live rows must not share a
  database. The exporter refuses `--feed preview` while a live model exists. Suggested schema fix: a
  `validated BOOLEAN` column on `entity_forecast`, filtered in `forecasts()`.
- Load the CSV with `\copy entity_forecast(entity_id,horizon_months,growth,model_version,created_at) FROM 'file.csv' CSV HEADER`.

