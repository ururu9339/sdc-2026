# Synthetic simulation: positive and negative controls for the forecasting pipeline

> **All data on this page is SYNTHETIC. It was generated from a known latent process and says NOTHING about the Malaysian labour market. It was never used to train, tune or validate the deployed model.**

Generated 2026-10-24T19:47:27.024372+00:00 by `python3 -m src.validation.simulate` (code: `src/validation/simulate.py`). Machine-readable results: `docs/simulation_results.json`. Runtime of this run: 686 s.

## 1. Purpose and why this is synthetic

The Malaysian demo data is mostly noise, so the gate rejects the model, and that rejection cannot tell us whether the gate or the model is at fault. This simulation has known ground truth, so each question has an answer we can check:

1. **False positives.** With no signal at all (S0), does the gate reject?
2. **Power.** With planted trends (S1) and a planted leading source (S2), does the model beat the baselines and pass the gate, and how much signal is needed?
3. **Use of the leading source.** Does the model learn the planted leading source (learned importance and the leave-one-source-out ablation)?

The gate, the features, the baselines and the audit are the pipeline's own code, called unchanged. Only the data is generated. Nothing was tuned to make a result look better. If the model fails to detect a planted signal, the tables say so.

## 2. Generator specification

- Entities: 120 (80 skills, 40 jobs). Periods: 130 weekly periods from 2021-01-04. Model grid: `PERIOD_FREQ=W`, horizon 12 periods, gate profiles `production` and `demo` (both from `src/training/config.py`).
- Seeds: 5 (0 to 4). Strengths: 0, 0.5, 1, 2. Scenarios: S0 (no strength; k = 0), S1 and S2 on the strength grid, S3 generation only.
- Latent log-share of entity `e`, period `t`: `x[e,t] = a[e] + k * D[e,t]`. `a[e]` is a constant base share drawn uniformly from 1% to 4% of postings. `k` is the strength (0 in S0).
- `D` (S1, S2) is a damped trend with random regime changes. Velocity relaxes with persistence PHI=0.9 toward a regime slope; the regime is redrawn with chance 0.0385 per period (about every 26 periods) and the slope is drawn from N(0, 0.01). The level decays with RHO=0.995.
- Observed share: counts are binomial draws, `count[e,t] ~ Binomial(N[t], exp(x[e,t]))`, with posting totals `N[t]` = 4000 (target) or 3000 (leading sources) times a log-normal jitter of sd 0.05. Then `value = count / N[t]` and `support = count`. This keeps the sampling noise that the signal audit measures.
- Each entity's count is drawn independently. A posting can mention several skills, so shares need not sum to 1 (an approximation, stated for honesty).
- Leading source A (`sim_lead_a`, labelled tier 1, reliability 0.8): in S2 it is a noisy copy that runs 8 periods ahead: `a[e] + k * D[e,t+8] + eps`, `eps ~ N(0, 0.30^2)` (STD_UNIT=0.3 divided by snr=1.0). In S0 and S1 it is pure sampling noise around `a[e]`.
- Leading source B (`sim_lead_b`, tier 3, reliability 0.25): pure sampling noise around `a[e]` in every scenario.
- Target (`sim_postings`, tier 2, reliability 0.55): the counts described above.
- Registry: in memory, in the real schema (role, granularity, reliability, release lag, ffill, enabled). The labels are fixed across scenarios, so the simulation does not reveal which source carries the signal.
- Grid note: S1 at k=0 is not re-run; its records are the S0 records (identical by construction). Strength 0.25 was dropped to keep the run within budget.
- S2 at k=0 is not a pure sampling-noise control for sim_lead_a: its eps noise is added on the log scale, so the audit flags it as non-noise although it carries no information about the target.
- S3 (duplicates): 10 groups of 3 entities whose target counts are identical copies. Generation and the duplicate check only.
- Determinism: each component draws from its own stream keyed by (seed, component). The same seed gives the same draws in every scenario and strength (common random numbers), so strength changes only the trend and the planted leading signal.

Gate profiles used (from the pipeline, unchanged):

| Profile | min_train_periods | mae_margin | min_fold_win_rate | coverage range |
|---|---|---|---|---|
| production | 52 | 0.03 | 0.6 | (0.65, 0.95) |
| demo | 26 | 0.0 | 0.5 | (0.55, 0.98) |

## 3. Results

Mean +- sd over seeds unless stated. Pass rates are counts with a 95% Wilson interval. Improvement is `100 * (1 - model MAE / best baseline MAE)` on the walk-forward folds: positive means the model is better.

### 3.1 Run inventory

| Scenario | Strength | Runs | Folds (production / demo) | Target noise-like share | Target median dispersion |
|---|---|---|---|---|---|
| S0 noise_only | 0 | 5 | 3.0 +- 0.0 / 5.0 +- 0.0 | 1.00 +- 0.00 | 1.00 +- 0.02 |
| S1 own_trend | 0 | 5 | 3.0 +- 0.0 / 5.0 +- 0.0 | 1.00 +- 0.00 | 1.00 +- 0.02 |
| S1 own_trend | 0.5 | 5 | 3.0 +- 0.0 / 5.0 +- 0.0 | 0.47 +- 0.05 | 1.57 +- 0.10 |
| S1 own_trend | 1 | 5 | 3.0 +- 0.0 / 5.0 +- 0.0 | 0.17 +- 0.06 | 3.23 +- 0.26 |
| S1 own_trend | 2 | 5 | 3.0 +- 0.0 / 5.0 +- 0.0 | 0.05 +- 0.01 | 9.76 +- 0.98 |
| S2 leading_signal | 0 | 5 | 3.0 +- 0.0 / 5.0 +- 0.0 | 1.00 +- 0.00 | 1.00 +- 0.02 |
| S2 leading_signal | 0.5 | 5 | 3.0 +- 0.0 / 5.0 +- 0.0 | 0.47 +- 0.05 | 1.57 +- 0.10 |
| S2 leading_signal | 1 | 5 | 3.0 +- 0.0 / 5.0 +- 0.0 | 0.17 +- 0.06 | 3.23 +- 0.26 |
| S2 leading_signal | 2 | 5 | 3.0 +- 0.0 / 5.0 +- 0.0 | 0.05 +- 0.01 | 9.76 +- 0.98 |

### 3.2 False-positive rate (S0, no signal)

| Gate profile | Passed (false positives) | Best baseline (mode) | Improvement over best baseline | Folds won |
|---|---|---|---|---|
| production | 0/5 (0%; 95% CI 0.00-0.43) | mean_reversion | -1.7% +- 1.0 | 0.13 +- 0.18 |
| demo | 0/5 (0%; 95% CI 0.00-0.43) | mean_reversion | -2.7% +- 0.7 | 0.08 +- 0.11 |

### 3.3 Pass rate and improvement against strength

| Scenario | Strength | Pass production | Pass demo | Improvement, demo folds | Model MAE (demo) | Best baseline MAE (demo) | Folds won | Coverage (demo) | Direction accuracy |
|---|---|---|---|---|---|---|---|---|---|
| S1 own_trend | 0 | 0/5 (0%; 95% CI 0.00-0.43) | 0/5 (0%; 95% CI 0.00-0.43) | -2.7% +- 0.7 | 0.0430 +- 0.0010 | mean_reversion 0.0419 +- 0.0012 | 0.08 +- 0.11 | 0.74 +- 0.00 | 0.73 +- 0.00 |
| S1 own_trend | 0.5 | 5/5 (100%; 95% CI 0.57-1.00) | 5/5 (100%; 95% CI 0.57-1.00) | +4.9% +- 0.6 | 0.0652 +- 0.0012 | persistence 0.0685 +- 0.0016 | 1.00 +- 0.00 | 0.74 +- 0.01 | 0.61 +- 0.00 |
| S1 own_trend | 1 | 0/5 (0%; 95% CI 0.00-0.43) | 4/5 (80%; 95% CI 0.38-0.96) | +0.7% +- 0.7 | 0.0865 +- 0.0017 | damped_trend 0.0871 +- 0.0015 | 0.80 +- 0.14 | 0.72 +- 0.01 | 0.63 +- 0.01 |
| S1 own_trend | 2 | 4/5 (80%; 95% CI 0.38-0.96) | 5/5 (100%; 95% CI 0.57-1.00) | +3.1% +- 1.7 | 0.1265 +- 0.0033 | damped_trend 0.1306 +- 0.0045 | 0.80 +- 0.00 | 0.71 +- 0.00 | 0.69 +- 0.02 |
| S2 leading_signal | 0 | 0/5 (0%; 95% CI 0.00-0.43) | 0/5 (0%; 95% CI 0.00-0.43) | -2.7% +- 0.8 | 0.0430 +- 0.0009 | mean_reversion 0.0419 +- 0.0012 | 0.00 +- 0.00 | 0.74 +- 0.01 | 0.74 +- 0.01 |
| S2 leading_signal | 0.5 | 5/5 (100%; 95% CI 0.57-1.00) | 5/5 (100%; 95% CI 0.57-1.00) | +7.1% +- 0.8 | 0.0637 +- 0.0014 | persistence 0.0685 +- 0.0016 | 1.00 +- 0.00 | 0.73 +- 0.01 | 0.63 +- 0.00 |
| S2 leading_signal | 1 | 5/5 (100%; 95% CI 0.57-1.00) | 5/5 (100%; 95% CI 0.57-1.00) | +5.5% +- 0.9 | 0.0824 +- 0.0017 | damped_trend 0.0871 +- 0.0015 | 0.96 +- 0.09 | 0.72 +- 0.01 | 0.66 +- 0.01 |
| S2 leading_signal | 2 | 5/5 (100%; 95% CI 0.57-1.00) | 5/5 (100%; 95% CI 0.57-1.00) | +10.7% +- 1.6 | 0.1166 +- 0.0037 | damped_trend 0.1306 +- 0.0045 | 1.00 +- 0.00 | 0.70 +- 0.01 | 0.73 +- 0.01 |

### 3.3b Error floor (reference, not a forecaster)

`Floor` is the MAE of predicting each observed 12-period target change with the TRUE latent change over the same horizon. That predictor uses the future, so it is unattainable. The floor is what sampling noise and smoothing leave even for a perfect forecaster of the latent. `Zero change` is the error of predicting no change.

| Scenario | Strength | Floor (true latent change) | Zero change | Model MAE (demo) | Best baseline MAE (demo) | Gap closed by model |
|---|---|---|---|---|---|---|
| S0 noise_only | 0 | 0.0580 +- 0.0014 | 0.0580 +- 0.0014 | 0.0430 +- 0.0010 | mean_reversion 0.0419 +- 0.0012 | n/a |
| S1 own_trend | 0 | 0.0580 +- 0.0014 | 0.0580 +- 0.0014 | 0.0430 +- 0.0010 | mean_reversion 0.0419 +- 0.0012 | n/a |
| S1 own_trend | 0.5 | 0.0587 +- 0.0010 | 0.0686 +- 0.0014 | 0.0652 +- 0.0012 | persistence 0.0685 +- 0.0016 | 34% |
| S1 own_trend | 1 | 0.0602 +- 0.0017 | 0.0925 +- 0.0038 | 0.0865 +- 0.0017 | damped_trend 0.0871 +- 0.0015 | 18% |
| S1 own_trend | 2 | 0.0651 +- 0.0026 | 0.1538 +- 0.0087 | 0.1265 +- 0.0033 | damped_trend 0.1306 +- 0.0045 | 31% |
| S2 leading_signal | 0 | 0.0580 +- 0.0014 | 0.0580 +- 0.0014 | 0.0430 +- 0.0009 | mean_reversion 0.0419 +- 0.0012 | n/a |
| S2 leading_signal | 0.5 | 0.0587 +- 0.0010 | 0.0686 +- 0.0014 | 0.0637 +- 0.0014 | persistence 0.0685 +- 0.0016 | 50% |
| S2 leading_signal | 1 | 0.0602 +- 0.0017 | 0.0925 +- 0.0038 | 0.0824 +- 0.0017 | damped_trend 0.0871 +- 0.0015 | 31% |
| S2 leading_signal | 2 | 0.0651 +- 0.0026 | 0.1538 +- 0.0087 | 0.1166 +- 0.0037 | damped_trend 0.1306 +- 0.0045 | 42% |

`Gap closed` = (zero change - model) / (zero change - floor): the share of the reducible error that the demo model removes. 100% would match the floor.

### 3.4 Power curve (demo gate, pass rate)

Bars show the pass rate under the demo profile. The production rate is in table 3.3.

```
S1 k=0     |..............................| 0%
S1 k=0.5   |##############################| 100%
S1 k=1     |########################......| 80%
S1 k=2     |##############################| 100%
S2 k=0     |..............................| 0%
S2 k=0.5   |##############################| 100%
S2 k=1     |##############################| 100%
S2 k=2     |##############################| 100%
```

### 3.5 Learned source importance (mean share of |SHAP|, in-sample on the full usable table)

Shares sum to one across groups: target own history, the two leading sources, the composite of leading sources, and calendar and metadata columns. The composite is reported as the pipeline reports it.

| Scenario | Strength | Target | sim_lead_a (planted in S2) | sim_lead_b (noise) | composite | calendar | sim_lead_a > sim_lead_b |
|---|---|---|---|---|---|---|---|
| S0 noise_only | 0 | 0.890 +- 0.009 | 0.039 +- 0.005 | 0.046 +- 0.005 | 0.006 +- 0.002 | 0.019 +- 0.005 | 1/5 |
| S1 own_trend | 0 | 0.890 +- 0.009 | 0.039 +- 0.005 | 0.046 +- 0.005 | 0.006 +- 0.002 | 0.019 +- 0.005 | 1/5 |
| S1 own_trend | 0.5 | 0.784 +- 0.023 | 0.074 +- 0.012 | 0.071 +- 0.009 | 0.023 +- 0.010 | 0.048 +- 0.020 | 2/5 |
| S1 own_trend | 1 | 0.830 +- 0.018 | 0.055 +- 0.013 | 0.057 +- 0.004 | 0.013 +- 0.010 | 0.044 +- 0.014 | 3/5 |
| S1 own_trend | 2 | 0.891 +- 0.016 | 0.037 +- 0.011 | 0.036 +- 0.006 | 0.010 +- 0.004 | 0.026 +- 0.005 | 3/5 |
| S2 leading_signal | 0 | 0.885 +- 0.010 | 0.042 +- 0.003 | 0.045 +- 0.005 | 0.008 +- 0.003 | 0.019 +- 0.006 | 2/5 |
| S2 leading_signal | 0.5 | 0.574 +- 0.031 | 0.315 +- 0.026 | 0.049 +- 0.007 | 0.032 +- 0.010 | 0.031 +- 0.015 | 5/5 |
| S2 leading_signal | 1 | 0.387 +- 0.025 | 0.497 +- 0.021 | 0.031 +- 0.005 | 0.060 +- 0.024 | 0.026 +- 0.012 | 5/5 |
| S2 leading_signal | 2 | 0.342 +- 0.022 | 0.539 +- 0.032 | 0.020 +- 0.005 | 0.085 +- 0.018 | 0.014 +- 0.004 | 5/5 |

### 3.6 Leave-one-source-out ablation (demo gate folds)

Improvement over the best baseline of each configuration. `without_sim_lead_a` removes the planted source. Positive means that configuration beats the baselines on these folds.

| Scenario | Strength | all sources | without lead A (planted) | without lead B (noise) | target only |
|---|---|---|---|---|---|
| S0 noise_only | 0 | -2.7% +- 0.7 | -2.8% +- 0.7 | -2.8% +- 0.8 | -2.8% +- 0.8 |
| S1 own_trend | 0 | -2.7% +- 0.7 | -2.8% +- 0.7 | -2.8% +- 0.8 | -2.8% +- 0.8 |
| S1 own_trend | 0.5 | +4.9% +- 0.6 | +4.8% +- 0.3 | +5.0% +- 0.6 | +4.8% +- 0.5 |
| S1 own_trend | 1 | +0.7% +- 0.7 | +0.6% +- 0.8 | +0.7% +- 0.8 | +0.7% +- 0.8 |
| S1 own_trend | 2 | +3.1% +- 1.7 | +3.3% +- 1.9 | +3.0% +- 1.6 | +3.3% +- 1.4 |
| S2 leading_signal | 0 | -2.7% +- 0.8 | -2.8% +- 0.7 | -2.6% +- 0.9 | -2.8% +- 0.8 |
| S2 leading_signal | 0.5 | +7.1% +- 0.8 | +4.8% +- 0.3 | +7.2% +- 0.8 | +4.8% +- 0.5 |
| S2 leading_signal | 1 | +5.5% +- 0.9 | +0.6% +- 0.8 | +5.6% +- 1.1 | +0.7% +- 0.8 |
| S2 leading_signal | 2 | +10.7% +- 1.6 | +3.3% +- 1.9 | +10.9% +- 1.4 | +3.3% +- 1.4 |

### 3.7 Signal audit

Dispersion ratio = observed variance of the share over the variance expected from sampling alone. About 1 means indistinguishable from noise. The noise-like flag uses the pipeline's threshold.

| Scenario | Strength | Target median dispersion | Target noise-like share | Lead A dispersion | Lead B dispersion |
|---|---|---|---|---|---|
| S0 noise_only | 0 | 1.00 +- 0.02 | 1.00 +- 0.00 | 1.01 +- 0.02 | 1.00 +- 0.01 |
| S1 own_trend | 0 | 1.00 +- 0.02 | 1.00 +- 0.00 | 1.01 +- 0.02 | 1.00 +- 0.01 |
| S1 own_trend | 0.5 | 1.57 +- 0.10 | 0.47 +- 0.05 | 1.01 +- 0.02 | 1.00 +- 0.01 |
| S1 own_trend | 1 | 3.23 +- 0.26 | 0.17 +- 0.06 | 1.01 +- 0.02 | 1.00 +- 0.01 |
| S1 own_trend | 2 | 9.76 +- 0.98 | 0.05 +- 0.01 | 1.01 +- 0.02 | 1.00 +- 0.01 |
| S2 leading_signal | 0 | 1.00 +- 0.02 | 1.00 +- 0.00 | 8.60 +- 0.34 | 1.00 +- 0.01 |
| S2 leading_signal | 0.5 | 1.57 +- 0.10 | 0.47 +- 0.05 | 9.19 +- 0.31 | 1.00 +- 0.01 |
| S2 leading_signal | 1 | 3.23 +- 0.26 | 0.17 +- 0.06 | 10.78 +- 0.53 | 1.00 +- 0.01 |
| S2 leading_signal | 2 | 9.76 +- 0.98 | 0.05 +- 0.01 | 15.72 +- 0.78 | 1.00 +- 0.01 |

**S3 duplicates.** Ground truth: 10 groups, 20 duplicate entities per seed. Duplicate detection: {"detector": "duplicate_groups", "truth_groups": 10, "detected_groups": 10, "groups_exact_match": true, "truth_members_detected": 30, "false_members": 0}.

## 4. Honest reading

- Noise only (S0), production gate: passed 0/5 (0%; 95% CI 0.00-0.43). A pass here is a false positive.
- Noise only (S0), production gate: with 5 seeds the 95% upper bound on the false-positive rate is 43%, so this run cannot show the rate is below 5%.
- Noise only (S0), demo gate: passed 0/5 (0%; 95% CI 0.00-0.43). A pass here is a false positive.
- Noise only (S0), demo gate: with 5 seeds the 95% upper bound on the false-positive rate is 43%, so this run cannot show the rate is below 5%.
- S1 (own_trend), production gate: the pass rate first reached 80% at strength 0.5.
- S1 (own_trend), demo gate: the pass rate first reached 80% at strength 0.5.
- S1 at the largest strength (2): the demo-gate model closes about 31% of the gap between a no-change forecast and the error floor (the best achievable).
- S1 at the largest strength (2): demo-gate improvement over the best baseline +3.1% +- 1.7; model MAE 0.1265 +- 0.0033 against best baseline damped_trend 0.1306 +- 0.0045; error floor with perfect latent knowledge 0.0651 +- 0.0026.
- S2 (leading_signal), production gate: the pass rate first reached 80% at strength 0.5.
- S2 (leading_signal), demo gate: the pass rate first reached 80% at strength 0.5.
- S2 at the largest strength (2): the demo-gate model closes about 42% of the gap between a no-change forecast and the error floor (the best achievable).
- S2 at the largest strength (2): demo-gate improvement over the best baseline +10.7% +- 1.6; model MAE 0.1166 +- 0.0037 against best baseline damped_trend 0.1306 +- 0.0045; error floor with perfect latent knowledge 0.0651 +- 0.0026.
- S2 at strength 2: mean share of |SHAP| for the planted leading source 0.539 +- 0.032 against the noise source 0.020 +- 0.005; the planted source has the larger share in 5/5 runs.
- S2 at strength 2, leave-one-out: model MAE with all sources 0.1166 against 0.1263 without the planted source. Improvement over the best baseline with all sources +10.7% +- 1.6, without the planted source +3.3% +- 1.9, target only +3.3% +- 1.4.

What these controls can and cannot show:

- **The generator is ours.** Every planted pattern matches the model's assumptions (damped trends, a leading source that is a lagged copy). Passing here shows the machinery works when its assumptions hold. It is not evidence that the forecast works on labour-market data.
- **The leading source's quality label is correlated with its truth** (lead A is both labelled higher quality and carries the signal). This flatters the composite. A real test would hide the labels.
- **Few seeds.** With the default 5 seeds, a false-positive rate has a resolution of 20 points and a 95% interval about +-40 points wide. Use the Wilson intervals, and raise `--seeds` for a finer estimate.
- **Importance is in-sample** on the full usable table, as the pipeline reports it. It shows what the fitted model relies on, not out-of-sample value. The ablation is the out-of-sample check.
- **Sample size.** 130 periods with 12-period horizon and 13-period folds leaves few production folds: the production profile needs 52 training periods, so only the earliest folds qualify. Table 3.1 shows the fold counts.
- **No tuning.** The model, features, baselines and gate are unchanged. The generator parameters were fixed before the run and are listed in section 2. Changing them changes the power curve, so the thresholds shown are for these settings only.
- **Reading the tables.** Read the pass rates together with the error floor. A model can fail the gate while still being far from the floor, and it can pass only by beating a baseline that itself is only a rough guess.
- **The planted lag is not identifiable from observed correlations.** The latent moves slowly, so the correlation of an observed leading source with the latent is nearly flat across lags 0 to 16. The lag is checked exactly on the noise-free planted lead (tests/test_simulation.py), not on the noisy observations.
- **Error floor.** The floor in table 3.3b is unattainable for any forecaster: it uses the future latent. The gap between the model and the best baseline is measured against it only to show how much error sampling noise leaves, not as a target.
- **Single entity count, single period grid, single leading lag** (lag 8 weeks). Power at other lags or with more entities is not tested.

## 5. How to reproduce

```bash
python3 -m src.validation.simulate                     # full run, writes docs/SIMULATION.md and docs/simulation_results.json
python3 -m src.validation.simulate --seeds 10 --out /path/to/elsewhere   # more seeds, other output folder
python3 -m src.validation.simulate --quick --out /path/to/elsewhere     # tiny grid, 2 seeds
python3 -m pytest -q tests/test_simulation.py
```

The run is deterministic for fixed arguments. The gate and the features use the pipeline's default weekly grid unless `PERIOD_FREQ=M` is set in the environment.
