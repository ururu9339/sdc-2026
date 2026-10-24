# Data sources: how to swap, add or upgrade

The model only ever sees the **long format** below, so a dataset is replaceable without touching
any modelling code. Sources are open or student-accessible proxies for training; an insider or
licensed dataset can replace any of them by following the same steps.

    entity_id, kind (skill|job), period (start of the source's own period), source_id, value, support

`value` is the measure (share, count, MYR...). `support` is the raw sample size behind it (optional).
Each source is normalised within its own history, so units do not need to match across sources.

## Time grid
- The model works on a **weekly** grid (weeks start on Monday, `config.PERIOD_FREQ = "W"`). The horizon
  (12 weeks), growth windows (4, 13, 26 weeks), minimum history (13 weeks), gate folds (13 weeks) and
  retraining cadence (every 4 weeks) are all counted in weeks.
- A period is identified by its start date and labelled `YYYY-MM-DD`. A source keeps its own
  `granularity` (weekly, monthly, quarterly, annual). Its figures are moved onto the weekly grid by
  `features.first_visible_period`: a figure for period P is known at the end of P plus
  `release_lag_days`, and becomes usable at the first origin week that ends after that time.
  Values are then carried forward for `ffill_limit_days`.
- A monthly figure is therefore never visible before its month is complete. A test in
  `tests/test_sources.py` checks that removing a not-yet-visible figure changes no feature.

## Swap a dataset
1. Convert the new data to the long format -> `data/raw/<source_id>.csv`, with periods at the start of their granularity period.
2. `python3 -m src.sources.validate data/raw/<source_id>.csv --source-id <source_id>`
3. Edit the entry in `sources_registry.json` (url, license, attribution, granularity, lag, thresholds).
4. `python3 -m src.pipeline.agent_loop replay ...` retrains and re-gates automatically. Features adapt to whichever sources have data; the gate decides whether the new combination is good enough to go live.

## Add a dataset
Same steps with a new `source_id` and `"role": "leading"`. No code changes needed.

## Registry fields that matter
| field | meaning |
|---|---|
| `role` | `target` (the series we forecast, exactly one) or `leading` |
| `enabled` | disabled sources are ignored |
| `scope` | `entity` rows are per skill/job; `mapped` rows are per sector/occupation key and `mappings/<source_id>.csv` (`entity_id,kind,key`) fans them out; `global` rows are economy-wide and broadcast to every entity |
| `granularity` | `weekly`, `monthly`, `quarterly` or `annual`: the native period of the source |
| `release_lag_days` | default 0. A figure for period P is only known `release_lag_days` after P ends (stops late releases leaking into backtests) |
| `ffill_limit_days` | optional. How long a published figure is carried forward on the weekly grid. Default: the granularity span (7 / 31 / 92 / 366 days) |
| `max_staleness_days` | leading sources. Older than this, measured from the period end plus release lag, the agent loop drops the source until it refreshes |
| `min_period_postings` | adapters drop periods with fewer total postings (reported in the meta sidecar as `dropped_periods`) |
| `min_entity_total` | adapters keep an entity only if it has at least this many matches over the kept periods |
| `dense` | `true`: every period between the first and last kept period must exist and is zero-filled. `false`: gaps stay gaps |
| `reliability`, `tier` | prior weight for the composite signal; the model also reports the learned contribution per source |
| `license`, `attribution`, `source_url` | shown in the demo; fill these for every source |

## Gate profiles
The gate rule is fixed: the candidate must beat the best naive baseline on time-ordered held-out folds.
Only the numeric constraints depend on the profile (`config.GATE_PROFILES`). The active profile is
`demo` unless the environment variable `GATE_PROFILE=production` is set. The profile name is written
to every version-log entry, every forecast record (`gate_profile`), each model's `meta.json` and the report.

| constraint | production | demo |
|---|---|---|
| folds / step (periods) | 6 / 13 | 6 / 13 |
| min train periods | 52 | 26 |
| min folds | 3 | 3 |
| MAE margin over best baseline | 3% | 0% |
| min fold win rate | 60% | 50% |
| 10-90% interval coverage | 65-95% | 55-98% |

The demo profile is relaxed because these sources are weak proxies. It is not a stricter test.

## Current adapters
Each adapter reads its own parameters from its registry entry, writes `data/raw/<source_id>.csv` and a `.meta.json` sidecar.
`python3 -m src.sources.build_all` runs all of them and validates the outputs.
- `job_descriptions.py` -> `job_postings` (target, weekly, dense). Also writes `data/processed/job_skill_graph.csv` (explanation only).
- `jobstreet.py` -> `jobstreet_my` (weekly, sparse, CC BY 4.0).
- `job_market.py` -> `market_signal` (monthly, occupation families only).
- `taxonomy.py` holds the shared role families (`role_families.json`) and the skill extractor (`skills_lexicon.json`). Family ids and skill ids must not collide; `build_all` fails if they do.
