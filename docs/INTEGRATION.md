# Integration notes for the embeddings team

For: whoever owns `embeddings/` (match.py, graph.py, schema.sql). This describes what the forecasting side
writes into `entity_forecast`, and what it does not guarantee. Nothing in `embeddings/` was changed.

## 1. Entity ids

- Our vocabulary **is** the taxonomy in `embeddings/taxonomy/{skills,jobs}.csv`. Ids are used verbatim:
  `skill:python`, `job:accountant`. The prefix gives the kind.
- We only forecast the entities that our data has enough postings for (the rest are not exported). Current
  counts are in `docs/RESULTS.md`, section "Taxonomy integration and preview feed".
- An id the taxonomy does not have is never exported. The exporter counts and skips it.

## 2. What a row means

| column | meaning |
|---|---|
| `entity_id` | taxonomy id |
| `horizon_months` | 3, from our 12-week horizon (`round(12 * 7 / 30.44)`) |
| `growth` | `exp(p50) - 1`: the median forecast relative change in the **smoothed share of postings** that mention the entity, over the horizon. 0.12 means the share is forecast to be 12% higher. It is not a change in absolute posting counts. |
| `model_version` | `v<N>` for a validated model. `preview:v<N>` for an unvalidated preview (see 4). |
| `created_at` | export time, UTC |

The share is postings mentioning the entity divided by all postings in the same week. Growth is measured on
the log of a 4-week smoothed share, so the forecast is a relative move, not a level.

## 3. Horizon: call match.py with `--horizon 3`

`match.py` reads `entity_forecast` with `horizon_months = 12` by default. Our rows have `horizon_months = 3`,
so match them with:

    python3 -m embeddings.match "python" "SQL" --horizon 3

Until a live model is validated (see 4), no live rows exist, so the matcher will report `no_forecast`.

## 4. Preview rows: unvalidated, and not mixed with live rows

- `model_version` starting with `preview:` marks a candidate model that **did not pass our validation gate**.
  Its numbers are for demonstrating the pipeline. Do not use them for decisions.
- The exporter refuses `--feed preview` while a validated live model exists.
- `match.py` picks the newest row per entity by `created_at`, **regardless of `model_version`**. So a preview row
  written after a live row would silently replace the live number. Keep preview rows out of a database that
  also holds live rows, or make the matcher skip them.
- Proposal (not implemented here, your schema): add `validated BOOLEAN NOT NULL DEFAULT false` to
  `entity_forecast`, set it `true` only for live rows, and filter `AND validated` in `forecasts()`.

## 5. Commands (our side)

    python3 -m src.sources.build_all                       # rebuild the three sources from the raw files
    python3 -m src.pipeline.agent_loop replay --start 2022-06-13 --end 2023-09-11 --step 4
    python3 -m src.pipeline.export_forecasts --feed live --dry-run      # print rows, write nothing
    python3 -m src.pipeline.export_forecasts --feed live --csv rows.csv # write a CSV instead of the DB
    DATABASE_URL=postgresql://... python3 -m src.pipeline.export_forecasts --feed live   # upsert

Writing needs `psycopg` and `DATABASE_URL`. The upsert is
`INSERT ... ON CONFLICT (entity_id, horizon_months, model_version) DO UPDATE SET growth, created_at`.
Re-running the export updates rows in place.

Taxonomy location: `SDC_TAXONOMY_DIR` overrides `embeddings/taxonomy/`.

## 6. Verified and not verified

Verified (automated tests, `tests/test_export.py`, `tests/test_api_preview.py`):
- row conversion: horizon months, the exp(p50)-1 transform, model_version prefixes, newest-record choice,
  skipping of ids outside the taxonomy;
- the upsert SQL text and parameter order, against a **fake** connection;
- the preview refusal when a live model exists;
- the dry-run and CSV output.

Not verified:
- writing to a real Postgres (no Postgres, pgvector or psycopg connection was available in this environment);
- `match.py` run against our rows, and `graph.py` against our data;
- alias quality: short names such as `Go` and `R` match ordinary words in job text. See `docs/RESULTS.md`.

## 7. Report endpoint (CV to report)

`POST /report` takes skills (taxonomy ids or free text) and optional CV text, and returns best-fit jobs, skill gaps,
a learning order, a descriptive market snapshot and labelled outlook figures. `GET /report/candidate/{id}` builds the
same report from a stored candidate. `GET /taxonomy/skills` lists the skill vocabulary.

**Engine selection** (`engine` in the request body, default `auto`):
- `auto`: uses the embeddings team's matcher (`embeddings/match.py`, reported as `qwen_pgvector`) when the
  `DATABASE_URL` environment variable is set and that matcher imports and runs. Otherwise it uses the offline matcher.
- `qwen_pgvector`: the same, but it still falls back to offline when unavailable.
- `offline_alias`: always the offline matcher.
If the real engine raises, the response uses the offline matcher and `warnings` says so. The `engine` field always
states which one produced the result.

**Offline vs Qwen.** The offline matcher resolves skills with the taxonomy alias rules (`src/sources/taxonomy.py`) and
scores with the job-skill graph in `data/processed/job_skill_graph.csv`, using the same formulas as
`embeddings/match.py` (constants are checked by a test). It has no language model, and free-text matches can be
noisy: two-character aliases are ignored in free text. The Qwen path calls `match_user`, `resolve_skills`,
`job_skill_profile`, and reads growth from `entity_forecast`.

**Outlook labels.** `feed` is `live` (validated model), `preview` (unvalidated; `validated: false` and the store's
PREVIEW banner) or `none` (no forecast; no growth shown). Growth is `exp(p50) - 1` over 3 months, and is shown only
for a 3-month horizon.

**Verified:** all report tests (`tests/test_report.py`, `tests/test_api_report.py`) run offline on hand-written
fixtures, including the hand-computed scores; the constants match `embeddings/match.py` and `graph.py`; a live
uvicorn smoke of `POST /report` and `GET /taxonomy/skills` on the real taxonomy and graph.

**Not verified:** the Qwen/pgvector path. No Postgres, pgvector, psycopg or Ollama was available, so that path only
has its fallback tested (a failing `DATABASE_URL` falls back offline). `GET /report/candidate/{id}` was tested with a
temporary SQLite database; the PDF OCR path depends on Tesseract, which was not installed here.
