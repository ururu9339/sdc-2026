# Prediction engine: results and honest limitations

Malaysia Skills & Jobs Forecasting Platform, prediction module (LightGBM forecaster + agent loop).
All numbers below come from `python3 -m src.training.report` and the agent-loop logs, and can be reproduced with the commands in section 7.

## 1. Summary

- We built a complete, tested forecasting pipeline: ingestion of three CSV sources, causal feature engineering, one global LightGBM model with 10/50/90% intervals, a fixed walk-forward gate, an abstention layer, append-only logs and a read-only API. 80 automated tests pass.
- **No model is currently live.** The gate compares every candidate with simple baselines on time-ordered held-out data and **rejected the model on all our datasets**. This is the system working as designed: it refuses to publish forecasts it cannot show are better than naive ones.
- On the Malaysian-demo data the cause is the data: two of the three files are statistically indistinguishable from random sampling noise, and the real one (JobStreet) has no time overlap with the target.
- On a real public dataset with genuine trends (Job-SDF) the model was **competitive but not good enough**: 3.9% worse than a damped-trend baseline under the strict gate. So better data alone is not enough to promote this model; we report that as is.
- We did not tune anything to pass. Weakness of the sources is disclosed in every forecast record and API response.

## 2. What the system does

| Stage | What it does | Where |
|---|---|---|
| Sources | Three CSV sources described in a registry (tier, reliability, license, granularity, publication lag). Swappable: a new or insider dataset needs a long-format CSV, `python3 -m src.sources.validate`, and a registry entry; no model code changes. | `src/sources/` |
| Features | Per skill/job and period: growth, z-scores within each series' own history, lags of leading sources, reliability-weighted composite. Coarse sources become visible only after they would have been published (no future leakage). | `src/training/features.py` |
| Model | One global LightGBM point model plus 10% and 90% quantile models. Horizon = 12 weeks (weekly mode) or 3 months (monthly mode), i.e. about one quarter ahead, not next week. | `src/training/train.py` |
| Gate | Walk-forward folds (train on the past, test on later periods). A candidate must beat the best of five baselines (persistence, trend, damped trend, mean reversion, half reversion) and have calibrated intervals. Two named profiles, `production` (strict) and `demo` (relaxed because our sources are weak proxies); the profile is recorded in every result. | `src/training/backtest.py` |
| Signal audit | Flags series whose variation is no larger than random sampling noise. | `src/training/audit.py` |
| Abstention | Thin history, thin counts, or too-wide intervals produce "insufficient data" instead of a number. | `src/training/forecast.py` |
| Agent loop | Perceive new data, decide whether to retrain, refit on the full accumulated table, gate, promote or reject, log. Hash-chained append-only logs; matured forecasts are scored later. | `src/pipeline/agent_loop.py` |
| API | `/forecast`, `/forecast/{id}`, `/history/{id}`, `/explain/{id}`, `/model`, `/sources`. With no live model they return the gate's reason, never invented numbers. Explanations are templates over structured fields (no language model sets or invents numbers). | `src/backend/` |

## 3. Data used and its status

All sources are open or student-accessible proxies used to train and demonstrate the method. They do **not** describe the true state of the Malaysian labour market, which would need far higher-quality (licensed) data.

| Source | File | Role | Period | License / provenance | Finding |
|---|---|---|---|---|---|
| `job_postings` | job_descriptions.csv (1.6M rows) | target (medium credibility) | 2021-09 to 2023-09, weekly (105 periods) | Provided by the team; license unverified | Looks synthetic. Median dispersion ratio 1.02: all 109 series are noise-like. |
| `jobstreet_my` | jobstreet_all_job_dataset.csv | leading (high credibility) | 2024-01 to 2025-05, 30 usable weeks with a gap | CC BY 4.0 per the Hugging Face card for azrai99/job-dataset (this file seems an updated copy; to be confirmed) | Real Malaysian listings, but **zero time overlap with the target**, so it cannot be a time-valid input. |
| `market_signal` | job_market_dataset.csv (500k rows) | leading (low credibility) | 2022-01 to 2025-12, monthly | Provided by the team; license unverified | Looks synthetic. Median dispersion ratio 0.81: 92% of series noise-like. |
| `job_sdf` (method check only) | Job-SDF benchmark | target in an isolated proof run | 2021-01 to 2023-12, monthly (36 periods) | Paper states CC BY-NC-SA 4.0 (non-commercial); the GitHub page states no license | Real Chinese job-ad data with anonymised ids. 0 of 314 series noise-like. Not Malaysian data. |

The "dispersion ratio" is the observed variance of a series' monthly share divided by the variance expected from pure sampling (binomial) noise. A value near 1 means the series moves no more than chance.

## 4. Result A: Malaysian-demo data (weekly, relaxed `demo` gate)

3 usable walk-forward folds; mean absolute error on 12-week-ahead change in log share:

| Fold cutoff | Model | Best baseline (mean reversion) |
|---|---|---|
| 2023-03-20 | 0.0251 | 0.0243 |
| 2022-12-19 | 0.0238 | 0.0227 |
| 2022-09-19 | 0.0275 | 0.0255 |

- Overall the model is 5.6% **worse** than reverting to the series' own average and wins 0 of 3 folds. Interval coverage is about 0.75 (nominal 0.80).
- This is what we expect on noise: no forecaster can beat reversion to the mean on a random series, and a flexible model slightly overfits it.
- Leave-one-source-out: all sources 0.02549, without `market_signal` 0.02527, target only 0.02527. The noise-like `market_signal` adds nothing. `jobstreet_my` could not be tested because it has no overlap.
- The agent loop replayed 66 weekly steps. Every candidate was rejected, so 0 models were promoted and 0 forecasts issued.

## 5. Result B: Job-SDF proof run (monthly, strict `production` gate)

Run in an isolated folder with real data that does carry signal. 4 usable folds:

| Fold cutoff | Model | Persistence | Trend | Damped trend (best) |
|---|---|---|---|---|
| 2023-06 | 0.1142 | 0.1356 | 0.0997 | 0.1090 |
| 2023-03 | 0.1193 | 0.1189 | 0.1034 | 0.1083 |
| 2022-12 | 0.0868 | 0.0924 | 0.1043 | 0.0934 |
| 2022-09 | 0.0890 | 0.0786 | 0.1033 | 0.0830 |
| **Mean** | **0.1023** | 0.106 | 0.103 | **0.0984** |

- The model is 3.9% worse than the best baseline (the gate needs 3% better), wins 1 of 4 folds (needs 60%), and its 10-90% interval covers only 62% (needs 65-95%). Verdict: **rejected**.
- It does get the direction right 53-78% of the time, so it carries some information, but a simple damped trend line is at least as accurate.
- Plausible reasons, not tested: only 36 months and 4 folds, the model uses only the series' own history (no leading sources in this dataset), later folds look like a distribution shift, and the interval models are too narrow.

## 6. What this does and does not show

Shown (verified by tests and runs):
- The gate genuinely rejects models that are not better than naive baselines, instead of publishing them.
- Noise detection correctly separates noise-like data (Malaysian-demo files) from data with real signal (Job-SDF).
- Future leakage is prevented by publication-lag-aware alignment; backtests are time-ordered.
- The pipeline is swappable: three different sources and a different dataset in a different period length ran through the same code.
- The agent loop, logging (tamper-evident hash chain), abstention and API handle the "no live model" state honestly.

Not shown:
- That the forecaster beats baselines on any dataset we have. We have no evidence it adds value yet.
- The live-model path (forecast records, outcome scoring, API serving forecasts) on real data: it is covered only by hand-written test fixtures.
- Combining several sources effectively: the only real leading source has no overlap, and the other is noise.
- Anything about the true Malaysian labour market.

## 7. Reproduce

```bash
python3 -m pytest -q tests                      # 80 tests
python3 -m src.sources.build_all                # rebuild the three Malaysian-demo sources (~35 s)
python3 -m src.training.report                  # audit, folds, gate verdict, ablation
python3 -m src.pipeline.agent_loop replay --start 2022-06-13 --end 2023-09-11 --step 1

# Job-SDF proof run (isolated)
export SDC_REGISTRY=$PWD/src/sources/registries/job_sdf_proof.json SDC_DATA_DIR=$PWD/data/job_sdf \
       SDC_MODELS_DIR=$PWD/src/models/job_sdf PERIOD_FREQ=M GATE_PROFILE=production
python3 -m src.sources.job_sdf build && python3 -m src.training.report
```

## 8. Next steps

1. Accumulate Malaysian listings weekly from a source whose terms allow it, so the strict gate can eventually apply to real local data.
2. Add real leading sources with long histories (for example Google Trends exports or Wikipedia pageviews per skill) once a target with signal exists.
3. If we try model improvements (predicting the correction on top of the damped trend, pooling more series, calibrated intervals), decide the experiments first and keep one fold frozen as a confirmation set; do not tune on the gate folds.
4. Skill-to-user matchmaking with embeddings is handled by teammates after our forecasts are produced.

## 9. Attribution

- JobStreet listings: azrai99/job-dataset (Hugging Face), CC BY 4.0 per its card.
- Job-SDF: *Job-SDF: A Multi-Granularity Dataset for Job Skill Demand Forecasting and Benchmarking*, NeurIPS 2024 Datasets and Benchmarks Track (non-commercial use).

## 10. Taxonomy integration and preview feed

Added after sections 1 to 9, which are unchanged. Where a rebuilt number differs from an earlier one, both are given.

**Vocabulary.** The teammate taxonomy (`embeddings/taxonomy`: 108 skills, 56 jobs) is now the only vocabulary, and entity ids are its ids (`skill:python`, `job:accountant`). The legacy `skills_lexicon.json` (78 skills) and `role_families.json` (37 families) are removed. 28 legacy skills and 16 legacy families had a taxonomy counterpart (same id, or same normalised name) and were carried over as regex extras in `src/sources/vocab_extra.json`. The other 50 legacy skills and 21 legacy families have no counterpart and were dropped; they are in git history (`git show HEAD:src/sources/skills_lexicon.json`, `git show HEAD:src/sources/role_families.json`).

Rebuilt with `python3 -m src.sources.build_all`:

| Source | Measure | Legacy vocabulary | Taxonomy |
|---|---|---|---|
| job_postings (target) | skills / jobs kept | 72 / 37 | 82 / 41 |
| job_postings | postings with at least one skill match | 84.1% | 82.8% |
| job_postings | postings with at least one job match | 97.6% | **55.3%** |
| jobstreet_my | skills / jobs kept | 77 / 37 | 104 / 49 |
| jobstreet_my | listings with at least one skill match | 92.0% | 95.6% |
| jobstreet_my | listings with at least one job match | 75.5% | **43.8%** |
| market_signal | occupations kept (of 12 distinct) | 12 | 9 |
| market_signal | rows whose occupation matches a job | 100% | 75.1% |

- Entity counts did not collapse (none fell below 40% of before). Job matching per posting **did** fall sharply: role strings often match no taxonomy name or alt label, and the legacy families they used to match were dropped. The market occupations HR Analyst, Marketing Specialist and Operations Manager have no taxonomy counterpart and are dropped; the sidecar lists them.
- A flat taxonomy has no priority order, so a job takes its most specific matching alias (longest), with ties to the earlier row. Legacy families were first-match in priority order. This is a deliberate change.
- Short names match ordinary words. `skill:go` matches 2.7% of JobStreet listings and `skill:r` 2.9%, mostly through everyday use of the word. This is not fixed. Case-sensitive matching for names of two characters or fewer would be the obvious next step.

**Model results after the rebuild.** The target now has 123 series instead of 109, so the error figures are not comparable across the two vocabularies.
- Legacy vocabulary (section 4): model 5.6% worse than the best baseline, 0 of 3 folds won.
- Taxonomy: model MAE 0.0362 against the best baseline (mean reversion) 0.0349, 3.8% worse, 0 of 3 folds won. The gate verdict is unchanged: **rejected**.
- `market_signal` audit: median dispersion ratio 0.64 with 89% of series noise-like (9 series). Before: 0.81 and 92% (12 series).

**Preview feed.** While no model is live, the gate-rejected candidate is still fitted on the full usable table, and its forecasts go to `data/preview_log.jsonl` with `validated: false` and `feed: "preview"`. They are scored into `data/preview_outcomes.jsonl`, separate from the live logs. Candidate models are in `src/models/preview_v<N>/` (git-ignored). Replay `2022-06-13` to `2023-09-11`, step 4:
- 17 runs, all on the preview feed. The live forecast log was not written (it does not exist).
- 2,091 preview forecasts (123 series x 17 runs). None abstained, so the abstention rules did not trigger on this dense data.
- Direction (all 2,091): stable 1,992, declining 51, emerging 48. Signal quality: noise_like 2,072, trend_detected 19. Confidence: low 2,072, moderate 3, high 16.
- 1,722 preview forecasts have matured, covering 14 target weeks that overlap. Their mean absolute error is 0.0375, against 0.0483 for persistence and 0.0605 for trend. The 10-90% interval covers 72% (nominal 80%). These numbers are retrospective, from overlapping targets, and unvalidated. They compare only against persistence and trend; the gate uses the stricter mean-reversion baseline, which the model does not beat (section 10, model results above), so this is not evidence of skill. They do not change the gate verdict. Nothing was tuned on them.

**Tests.** 116 pass (`python3 -m pytest tests`). A bare `python3 -m pytest` also collects a vendored benchmark under `src/data/job_sdf/benchmark` and errors on collection. That predates this work.

**Not verified.** Writing to a real Postgres (see `docs/INTEGRATION.md`); `embeddings/match.py` run against our rows.
