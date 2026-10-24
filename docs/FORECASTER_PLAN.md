# Forecaster improvement plan

Why the LightGBM forecaster underperforms, and what to change, in order. Numbers are from
`serving/model_report.json`, `serving/raw/jobstreet_my.csv` and `src/sources/sources_registry.json` as committed
on 2026-10-25.

## 1. Diagnosis: the data is the problem, not the model

| Finding | Evidence |
|---|---|
| The model loses to a naive baseline | MAE 0.0362 vs mean reversion 0.0349 (3.8% worse); fold win rate 0 of 3 |
| The target series has no learnable signal | `job_postings` audit: 100% of 123 series are `noise_like`; every forecast is `confidence: low` |
| The target is very likely synthetic | Registry: `"license": "UNVERIFIED - appears synthetic"`. Many job–skill edges have weight 1.0 (every DevOps posting lists AWS, Azure *and* GCP), and Kotlin, Swift and Mobile Developer forecast exactly −16.8% with identical intervals |
| The forecasts are stale | Target runs 2021-09-13 to 2023-09-11, so the "next 3 months" are late 2023 |
| The only real Malaysian data contributes nothing | `jobstreet_my` (2024-01-29 to 2025-05-05) overlaps the target by 0 periods |
| Key entities have no forecast | 41 of 164 taxonomy entities, including every AI skill and job (LLM, prompt engineering, RAG, AI agents, MLOps) and the most automation-exposed jobs (bookkeeper, admin clerk, production operator, delivery driver) |

The intervals are calibrated (76% coverage), and direction accuracy is 70–76%. The pipeline itself is
sound: audit, walk-forward folds, baselines and gate all work. A gradient-boosted model cannot beat mean
reversion on series that are noise. Tuning hyperparameters will not change that.

## 2. Should we just retrain and push again?

**No, not on the same data.** Retraining on `job_postings` will reproduce the same gate failure. Retrain after
the target data changes (section 3). The workflow then stays as it is:

```bash
python3 -m src.pipeline.agent_loop replay ...   # retrain + gate on the new data
python3 -m src.training.report                  # check the verdict and folds against every baseline
python3 -m src.pipeline.refresh                 # writes serving/
git add serving/ && git commit && git push      # the API serves the bundle, not a model binary
```

Push the refreshed `serving/` bundle whatever the verdict: a failed gate is still served, labelled PREVIEW,
which is the honest outcome. Never loosen the gate to make a model pass.

## 3. Plan

### Phase 1: real target data (highest impact)

1. **Disable `job_postings`** as the target (`"enabled": false`), or at most keep it as a leading source with low
   `reliability`. Synthetic data must not be what we validate against.
2. **Make `jobstreet_my` the target** and move to the monthly grid (`PERIOD_PROFILES["M"]`), since JobStreet is
   sparse weekly (30 of 67 weeks have data).
3. **Get more real history.** JobStreet alone has 9 months with data and an 8-month gap (2024-07 to 2025-02).
   The monthly demo gate needs about 18 contiguous months (9 training + 3 folds × 3). Candidates, in order of
   effort (all to verify for licence and coverage first):
   - The original source of the repo's 2025 JobStreet rows (the Hugging Face copy stops at 2024-09).
   - OpenDOSM / data.gov.my labour-market series by occupation or sector (official, monthly or quarterly). Add as
     a `mapped` source via `mappings/<source_id>.csv` (MASCO group → taxonomy jobs).
   - Google Trends interest for Malaysia per skill keyword (weekly, years of history, covers AI terms). A
     leading indicator, not postings. Label it so in the registry.
   - Fresh JobStreet / MYFutureJobs listings, only where the terms of service allow collection.
4. **Fix the JobStreet licence entry**: the dataset card says Apache-2.0, the registry says CC BY 4.0.

### Phase 2: coverage

5. Lower `min_entity_total` for the real target where support allows, so that AI skills present in JobStreet
   (`skill:llm`, `skill:llm_finetuning`, `skill:machine_learning`, `skill:deep_learning`, `skill:ai_ethics`) get
   forecasts. Entities with too little support should return `insufficient_data`, as the API already supports.
6. Report coverage per kind in `model_report.json`: forecasts issued vs taxonomy size, and how many entities
   abstained and why.

### Phase 3: evaluation that answers "is it better than doing nothing?"

7. Add per-kind gate metrics (skills vs jobs) and direction accuracy *of the best baseline*, so the model's
   70–76% can be compared with something.
8. Add a check that flags identical forecasts across entities (the Kotlin/Swift lockstep). It is a cheap
   detector of templated or duplicated source data.
9. If LightGBM still cannot beat the best baseline on real data, say so in the demo and serve that result.
   Showing a rigorous gate that rejects a weak model is a stronger story than an unvalidated number.

## 4. Done when

- The target is real Malaysian postings data, and `as_of` is within the last 6 months.
- `python3 -m src.training.report` shows the model against every baseline on the real target, with the
  verdict printed (pass or fail).
- Forecasts exist for the AI skills present in the data, and no two unrelated entities have identical forecasts.
- `serving/` is refreshed and pushed. `/api/v1/health` shows the new `as_of` and model version.
