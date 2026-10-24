# Data sources: how to swap, add or upgrade

The model only ever sees the **long format** below, so a dataset is replaceable without touching
any modelling code. Three sources are planned: `job_postings` (target), one government source,
one other signal. All are open/student-accessible proxies for training; an insider or licensed
dataset can replace any of them by following the same steps.

    entity_id, kind (skill|job), period (month start), source_id, value, support

`value` is the measure (share, count, MYR...). `support` is the raw sample size behind it (optional).
Each source is normalised within its own history, so units do not need to match across sources.

## Swap a dataset
1. Convert the new data to the long format -> `data/raw/<source_id>.csv`.
2. `python -m src.sources.validate data/raw/<source_id>.csv --source-id <source_id>`
3. Edit the entry in `sources_registry.json` (url, license, attribution, granularity, lag).
4. `python -m src.pipeline.agent_loop replay ...` retrains and re-gates automatically. Features adapt to whichever sources have data; the gate decides whether the new combination is good enough to go live.

## Add a dataset
Same steps with a new `source_id` and `"role": "leading"`. No code changes needed.

## Registry fields that matter
| field | meaning |
|---|---|
| `role` | `target` (the series we forecast, exactly one) or `leading` |
| `enabled` | disabled sources are ignored |
| `scope` | `entity` rows are per skill/job; `mapped` rows are per sector/occupation key and `mappings/<source_id>.csv` (`entity_id,kind,key`) fans them out; `global` rows are economy-wide and broadcast to every entity |
| `granularity`, `ffill_limit` | quarterly/annual series are forward-filled up to `ffill_limit` months |
| `release_lag_months` | a figure for period P only becomes usable at P+lag (stops annual releases leaking into backtests) |
| `max_staleness_months` | older than this, the agent loop drops the source until it refreshes |
| `reliability`, `tier` | prior weight for the composite signal; the model also reports the learned contribution per source |
| `license`, `attribution`, `source_url` | shown in the demo; fill these for every source |

## Current adapters
- `jobstreet.py`: JobStreet listings (CC BY 4.0) -> `job_postings.csv`. `skills_lexicon.json` is the keyword skill extractor.
