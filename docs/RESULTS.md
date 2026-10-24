# Results

Prediction module of Currvix: data ingestion, a LightGBM demand forecaster, a validation gate, an agent loop and an API. All figures come from `python3 -m src.training.report`, the agent loop logs and `python3 -m src.validation.simulate`, and can be reproduced with the commands at the end.

## 1. Summary

- The pipeline runs end to end: three CSV sources, causal features, one global LightGBM model with 10, 50 and 90 percent outputs, a fixed walk forward gate, abstention rules, tamper evident logs and a versioned API. 277 automated tests pass.
- No model is validated. The gate rejects the forecaster on every dataset we have, so forecasts are served as an unvalidated preview with a banner, the as of date and the reason.
- On the Malaysian data the cause is the data: the series vary no more than random sampling would, and the one clearly real source (JobStreet) does not overlap the target period.
- On a real public benchmark with genuine trends (Job-SDF) the model is close to the baselines but still 3.9 percent worse than a damped trend under the strict gate.
- Nothing was tuned to pass. The weak points are listed in section 8.

## 2. What the system does

| Part | Function | Location |
|---|---|---|
| Sources | Three CSV sources described in a registry (tier, reliability, license, granularity, publication lag). New datasets are added by converting to the long format, running the validator and adding a registry entry. | `src/sources/` |
| Features | Growth, z scores and lags per skill or job. Leading sources are aligned so a value is only used after it would have been published. | `src/training/features.py` |
| Model | One global LightGBM point model plus 10 and 90 percent quantile models. Horizon: 12 weeks (about one quarter). | `src/training/train.py` |
| Gate | Walk forward folds. A candidate must beat the best of five baselines (persistence, trend, damped trend, mean reversion, half reversion). Two profiles: `production` (strict) and `demo` (relaxed). The profile is recorded in every result. | `src/training/backtest.py` |
| Audit | Flags series that vary no more than sampling noise, and groups of entities with identical series. | `src/training/audit.py` |
| Agent loop | Load data, decide whether to retrain, refit on the full table, gate, forecast, score matured forecasts, log. | `src/pipeline/agent_loop.py` |
| API | Forecasts, explanations, history, snapshot, CV report, taxonomy, health. | `src/backend/` |
| Serving bundle | Small committed snapshot the API reads on Replit. | `serving/` |

## 3. Data and licenses

All sources are open or student accessible proxies. They are used to train and demonstrate the method and do not describe the true Malaysian labour market.

| Source | File | Role | Period | License |
|---|---|---|---|---|
| `job_postings` | job_descriptions.csv (1.6M rows) | target | 2021 09 to 2023 09, weekly, 105 periods | CC BY 4.0 (confirmed by the team) |
| `jobstreet_my` | jobstreet_all_job_dataset.csv | leading | 2024 01 to 2025 05, 30 usable weeks, with a gap | CC BY 4.0 (azrai99/job-dataset, Hugging Face) |
| `market_signal` | job_market_dataset.csv (500k rows) | leading | 2022 01 to 2025 12, monthly | CC BY 4.0 (confirmed by the team) |
| Job-SDF | benchmark repository | method check only | 2021 01 to 2023 12, monthly | CC BY NC SA 4.0 per the paper (non commercial) |

Findings about the data:
- `job_postings` and `market_signal` vary no more than random sampling would. The ratio of observed to expected sampling variance is about 1.0 (median 1.02 and 0.64), and 100 percent and 89 percent of their series are noise like.
- `jobstreet_my` is the only source with clear structure, but it starts after the target ends, so it cannot be a time valid input.
- Exact duplicate postings are rare: 0.41 percent repeat on title, role, skills, company and date.
- 16 entities in 6 groups have identical weekly series (for example Kotlin, Swift and Mobile Developer). Each group is treated as one signal. Members still receive a forecast copied from the group, flagged `shares_signal_with`.
- Under the shared skill and job taxonomy (108 skills, 56 jobs), 82.8 percent of job description postings match a skill and 55.3 percent match a job. For JobStreet the figures are 95.6 and 43.8 percent.

## 4. Result A: Malaysian data (weekly, relaxed demo gate)

113 modelled series, 3 walk forward folds, error on the 12 week change in log share:

| Fold cutoff | Model | Mean reversion (best baseline) | Persistence |
|---|---|---|---|
| 2023 03 20 | 0.0343 | 0.0329 | 0.0438 |
| 2022 12 19 | 0.0304 | 0.0294 | 0.0447 |
| 2022 09 19 | 0.0370 | 0.0359 | 0.0474 |

- Overall the model is 3.7 percent worse than reverting to the series average and wins 0 of 3 folds. It beats persistence in every fold, but the gate uses the best baseline.
- Direction is right 73 percent of the time. The 10 to 90 percent interval covers about 76 percent (nominal 80).
- Removing the duplicate series did not change the verdict (before: 3.8 percent worse).
- The agent loop replayed the history in 17 steps. Every candidate was rejected, so no model was promoted.

## 5. Result B: Job-SDF proof run (monthly, strict production gate)

Isolated run on real data: 314 series, 4 folds.

| Fold cutoff | Model | Persistence | Trend | Damped trend (best) |
|---|---|---|---|---|
| 2023 06 | 0.1142 | 0.1356 | 0.0997 | 0.1090 |
| 2023 03 | 0.1193 | 0.1189 | 0.1034 | 0.1083 |
| 2022 12 | 0.0868 | 0.0924 | 0.1043 | 0.0934 |
| 2022 09 | 0.0890 | 0.0786 | 0.1033 | 0.0830 |
| Mean | 0.1023 | 0.106 | 0.103 | 0.0984 |

- The model is 3.9 percent worse than the best baseline (the gate needs 3 percent better), wins 1 of 4 folds, and its interval covers 62 percent (needs 65 to 95). Verdict: rejected.
- None of the 314 series is noise like, so this data does carry signal.
- Likely causes, not tested: only 36 months, only the series' own history as input, a shift in 2023, and narrow intervals.

## 6. Preview feed

While the gate rejects the model, the candidate's forecasts are served separately and labelled `validated: false`.
- 123 forecasts (113 modelled plus 10 copies): 114 stable, 6 declining, 3 emerging. All are low confidence and noise like.
- As of 2023 09 11, so every forecast is more than three years old. The API reports the age and a stale flag.
- Refits are unstable: 29 of 113 modelled series changed direction between the last two versions.
- Retrospective preview error on matured forecasts is 0.0349, against 0.0450 for persistence and 0.0569 for trend, with 72 percent interval coverage. These do not compare against the mean reversion baseline the gate uses, so they are not evidence of skill.

## 7. Simulation check (synthetic, not Malaysian data)

Generated series with known structure, run through the same pipeline (`docs/SIMULATION.md`):
- With no signal the gate rejected 5 of 5 runs under both profiles. Five runs cannot show the false positive rate is below 5 percent.
- With planted trends or a planted leading source the model beat the best baseline by about 3 to 11 percent from signal strength 0.5 upward. Under the strict profile one setting passed 0 of 5 runs.
- The model relied on the planted leading source and ignored a pure noise source. Dropping the planted source cut the gain (for example 10.7 to 3.3 percent), dropping the noise source cost nothing.
- Planted duplicate groups were recovered exactly.
- The model closes only about a third to two fifths of the reducible error.

This shows the gate and pipeline respond to real signal. It says nothing about Malaysia and was never used to train or tune the deployed model.

## 8. What is and is not shown

Shown:
- The gate rejects models that do not beat naive baselines, and the API labels such forecasts.
- Noise like data is separated from data with signal.
- Backtests are time ordered and future values are not used before publication.
- Data sources can be swapped without code changes.

Not shown:
- That the forecaster beats baselines on any real dataset.
- Combining several informative real sources, because only one source has real structure and it does not overlap the target.
- The live model path on real data, the Qwen and Postgres matcher, PDF OCR and a real Replit deployment. These are covered by test fixtures or untested.

Known weaknesses:
- Forecasts are stale (as of 2023 09).
- 41 of 164 taxonomy entities have no forecast, including newer AI skills.
- Job matching is limited (about half of postings), and two letter skill names such as Go and R match ordinary words.

## 9. Reproduce

```bash
python3 -m pytest -q tests
python3 -m src.sources.build_all
python3 -m src.training.report
python3 -m src.pipeline.agent_loop replay --start 2022-06-13 --end 2023-09-11 --step 4
python3 -m src.pipeline.refresh
python3 -m src.validation.simulate

export SDC_REGISTRY=$PWD/src/sources/registries/job_sdf_proof.json SDC_DATA_DIR=$PWD/data/job_sdf \
       SDC_MODELS_DIR=$PWD/src/models/job_sdf PERIOD_FREQ=M GATE_PROFILE=production
python3 -m src.sources.job_sdf build && python3 -m src.training.report
```

## 10. Attribution

- JobStreet listings: azrai99/job-dataset (Hugging Face), CC BY 4.0.
- Job descriptions and job market datasets: CC BY 4.0. Original titles and authors to be added.
- Job-SDF: *Job-SDF: A Multi-Granularity Dataset for Job Skill Demand Forecasting and Benchmarking*, NeurIPS 2024 Datasets and Benchmarks Track (non commercial use).
