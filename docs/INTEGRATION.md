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
