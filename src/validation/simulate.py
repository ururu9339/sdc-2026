from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.sources.loaders import COLUMNS
from src.training import audit as audit_module
from src.training import config as C
from src.training.ablation import run_ablation
from src.training.audit import signal_audit, summarise_audit
from src.training.backtest import evaluate_gate, walk_forward
from src.training.features import build_feature_table, column_source_map, feature_columns, usable
from src.training.train import fit_models, source_contributions

ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_DISCLAIMER = ("SYNTHETIC SIMULATION. Every series here is generated from a known latent process. It says "
                        "nothing about the Malaysian labour market and was not used to train, tune or validate "
                        "the deployed model.")
SCENARIOS = {"S0": "noise_only", "S1": "own_trend", "S2": "leading_signal", "S3": "duplicates"}
TARGET_ID, LEAD_A, LEAD_B = "sim_postings", "sim_lead_a", "sim_lead_b"
LEADS = (LEAD_A, LEAD_B)
CONFIGS = ("all_sources", "without_sim_lead_a", "without_sim_lead_b", "target_only")
PROFILES = ("production", "demo")
DEFAULT_STRENGTHS = (0.0, 0.5, 1.0, 2.0)
QUICK_STRENGTHS = (0.0, 1.0)
DEFAULT_SEEDS, QUICK_SEEDS = 5, 2
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_ENTITIES, DEFAULT_PERIODS = 120, 130
START = {"W": "2021-01-04", "M": "2021-01-01"}
GRANULARITY = {"W": "weekly", "M": "monthly"}


LEAD_LAG = 8
SNR = 1.0
STD_UNIT = 0.3
PHI = 0.9
RHO = 0.995
REGIME_RATE = 1 / 26
SIGMA_MU = 0.01
BASE_SHARE = (0.01, 0.04)
POSTINGS = 4000
LEAD_POSTINGS = 3000
TOTAL_JITTER = 0.05
SKILL_FRACTION = 2 / 3
DUP_GROUPS, DUP_GROUP_SIZE = 10, 3


@dataclass
class Dataset:
    obs: pd.DataFrame
    registry: dict
    truth: dict


def _check_freq(freq: str | None) -> str:
    f = C.PERIOD_FREQ if freq is None else freq
    if f != C.PERIOD_FREQ:
        raise ValueError(f"the pipeline runs on PERIOD_FREQ={C.PERIOD_FREQ!r}; set PERIOD_FREQ={f} in the "
                         "environment to simulate on that grid")
    return f


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def _damped_trends(rng: np.random.Generator, n_ent: int, n_t: int) -> np.ndarray:
    redraw = rng.random((n_ent, n_t)) < REGIME_RATE
    redraw[:, 0] = True
    slopes = rng.normal(0.0, SIGMA_MU, (n_ent, n_t))
    out = np.empty((n_ent, n_t))
    regime, vel, level = np.zeros(n_ent), np.zeros(n_ent), np.zeros(n_ent)
    for t in range(n_t):
        regime = np.where(redraw[:, t], slopes[:, t], regime)
        vel = PHI * vel + (1 - PHI) * regime
        level = RHO * level + vel
        out[:, t] = level
    return out


def _totals(rng: np.random.Generator, n_t: int, base: int) -> np.ndarray:
    return np.maximum(1, np.round(base * np.exp(TOTAL_JITTER * rng.standard_normal(n_t)))).astype(np.int64)


def _long(ids, kinds, periods: np.ndarray, source_id: str, counts: np.ndarray, totals: np.ndarray) -> pd.DataFrame:
    n, t = counts.shape
    return pd.DataFrame({
        "entity_id": np.repeat(ids, t), "kind": np.repeat(kinds, t), "period": np.tile(periods, n),
        "source_id": source_id, "value": (counts / totals[None, :]).ravel(),
        "support": counts.ravel().astype(float),
    })[COLUMNS]


def make_registry(freq: str | None = None) -> dict:
    f = _check_freq(freq)
    span = C.GRANULARITY_SPAN_DAYS[GRANULARITY[f]]
    common = {"granularity": GRANULARITY[f], "release_lag_days": 0, "ffill_limit_days": span, "dense": True,
              "enabled": True, "license": "SYNTHETIC (generated in process)", "attribution": "none: simulation",
              "source_url": "none", "adapter": "src.validation.simulate", "loader": {"type": "in_memory"},
              "scope": "entity"}
    return {"version": 1, "synthetic": True, "disclaimer": SYNTHETIC_DISCLAIMER, "tiers": {}, "sources": [
        {**common, "source_id": TARGET_ID, "name": "SYNTHETIC postings share (target)", "role": "target",
         "tier": 2, "reliability": 0.55},
        {**common, "source_id": LEAD_A, "name": "SYNTHETIC leading source A (higher labelled quality)",
         "role": "leading", "tier": 1, "reliability": 0.8, "max_staleness_days": 730},
        {**common, "source_id": LEAD_B, "name": "SYNTHETIC leading source B (lower labelled quality)",
         "role": "leading", "tier": 3, "reliability": 0.25, "max_staleness_days": 730},
    ]}


def generate(scenario: str, strength: float, seed: int, entities: int = DEFAULT_ENTITIES,
             periods: int = DEFAULT_PERIODS, freq: str | None = None, lead_lag: int = LEAD_LAG,
             snr: float = SNR) -> Dataset:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    f = _check_freq(freq)
    n, t_n, lag = int(entities), int(periods), int(lead_lag)
    k = 0.0 if scenario == "S0" else float(strength)
    n_skill = round(n * SKILL_FRACTION)
    ids = [f"skill_sim_{i:03d}" for i in range(n_skill)] + [f"job_sim_{j:03d}" for j in range(n - n_skill)]
    kinds = ["skill"] * n_skill + ["job"] * (n - n_skill)
    periods = pd.period_range(START[f], periods=t_n, freq=f).start_time.to_numpy()

    base = np.log(_rng(seed, 0).uniform(*BASE_SHARE, n))
    trends = _damped_trends(_rng(seed, 1), n, t_n + lag)
    x_t = base[:, None] + k * trends[:, :t_n]
    tot_t = _totals(_rng(seed, 2), t_n, POSTINGS)
    counts_t = _rng(seed, 3).binomial(tot_t[None, :], np.clip(np.exp(x_t), 0, 1))

    groups: list[list[str]] = []
    if scenario == "S3":
        for g in range(min(DUP_GROUPS, n // DUP_GROUP_SIZE)):
            members = list(range(g * DUP_GROUP_SIZE, (g + 1) * DUP_GROUP_SIZE))
            counts_t[members[1:]] = counts_t[members[0]]
            groups.append([ids[m] for m in members])

    lead_signal = None
    eps = _rng(seed, 4).standard_normal((n, t_n))
    if scenario == "S2":
        lead_signal = base[:, None] + k * trends[:, lag:lag + t_n]
        x_a = lead_signal + (STD_UNIT / snr) * eps
    else:
        x_a = np.repeat(base[:, None], t_n, axis=1)
    tot_a = _totals(_rng(seed, 5), t_n, LEAD_POSTINGS)
    counts_a = _rng(seed, 6).binomial(tot_a[None, :], np.clip(np.exp(x_a), 0, 1))
    tot_b = _totals(_rng(seed, 7), t_n, LEAD_POSTINGS)
    x_b = np.repeat(base[:, None], t_n, axis=1)
    counts_b = _rng(seed, 8).binomial(tot_b[None, :], np.clip(np.exp(x_b), 0, 1))

    obs = pd.concat([_long(ids, kinds, periods, TARGET_ID, counts_t, tot_t),
                     _long(ids, kinds, periods, LEAD_A, counts_a, tot_a),
                     _long(ids, kinds, periods, LEAD_B, counts_b, tot_b)], ignore_index=True)
    truth = {"scenario": scenario, "name": SCENARIOS[scenario], "strength": k, "seed": int(seed),
             "lead_lag": lag, "snr": float(snr), "latent_target": x_t, "lead_signal": lead_signal,
             "entity_ids": ids, "kinds": kinds, "periods": [str(p)[:10] for p in periods],
             "duplicate_groups": groups}
    return Dataset(obs=obs, registry=make_registry(f), truth=truth)


def duplicate_detection(obs: pd.DataFrame, registry: dict, truth_groups: list[list[str]]) -> dict:
    fn = getattr(audit_module, "duplicate_groups", None)
    if fn is None:
        return {"detector": "not_available", "note": "src/training/audit.py has no duplicate_groups function"}
    try:
        found = fn(obs, registry, pd.Timestamp(obs["period"].max()))
    except Exception as exc:
        return {"detector": "duplicate_groups", "error": f"{type(exc).__name__}: {exc}"}
    detected = {frozenset(g["members"]) for g in found}
    truth = {frozenset(g) for g in truth_groups}
    return {"detector": "duplicate_groups", "truth_groups": len(truth), "detected_groups": len(detected),
            "groups_exact_match": detected == truth,
            "truth_members_detected": len(set().union(*truth) & set().union(*detected)) if truth and detected else 0,
            "false_members": len(set().union(*detected) - set().union(*truth)) if detected else 0}


def _num(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if (math.isnan(x) or math.isinf(x)) else x


def _gate_summary(folds: pd.DataFrame, gate: dict) -> dict:
    out = {"passed": bool(gate["passed"]), "reason": gate["reason"], "n_folds": int(len(folds)),
           "checks": {k: bool(v) for k, v in gate.get("checks", {}).items()}}
    if "model_mae" not in gate or folds.empty:
        return out
    w = folds["n_test"].astype(float)
    wmean = lambda c: float(np.average(folds[c], weights=w))
    base_cols = [c for c in folds.columns if c.startswith("mae_") and c != "mae_model"]
    best = gate["best_baseline"]
    out.update({
        "model_mae": _num(gate["model_mae"]), "best_baseline": best, "best_baseline_mae": _num(gate["best_baseline_mae"]),
        "improvement_pct": _num(gate["improvement_pct"]), "fold_win_rate": _num(gate["fold_win_rate"]),
        "folds_won": int((folds["mae_model"] < folds[f"mae_{best}"]).sum()),
        "coverage": _num(gate["coverage"]), "direction_accuracy": _num(wmean("dir_acc_model")),
        "baseline_mae": {c.removeprefix("mae_"): _num(wmean(c)) for c in base_cols},
    })
    return out


def _importance(feats: pd.DataFrame, cols: list[str], registry: dict) -> dict:
    train = usable(feats)
    train = train[train["y"].notna()]
    models = fit_models(train, cols)
    imp = source_contributions(models, train[cols].astype(float), column_source_map(cols, registry)).abs().mean()
    total = float(imp.sum())
    return {"mean_abs_shap": {k: _num(v) for k, v in imp.items()},
            "share": {k: (_num(v / total) if total else None) for k, v in imp.items()}}


def _noise_floor(feats: pd.DataFrame, truth: dict) -> dict | None:
    if "latent_target" not in truth or truth.get("latent_target") is None:
        return None
    rows = usable(feats)
    rows = rows[rows["y"].notna()]
    if rows.empty:
        return None
    x = truth["latent_target"]
    t_pos = {pd.Timestamp(p): i for i, p in enumerate(pd.to_datetime(truth["periods"]))}
    e_pos = {e: i for i, e in enumerate(truth["entity_ids"])}
    i = rows["period"].map(t_pos).to_numpy()
    j = rows["entity_id"].map(e_pos).to_numpy()
    h = C.HORIZON
    ok = (i + h) < x.shape[1]
    y = rows["y"].to_numpy()[ok]
    true_change = x[j[ok], i[ok] + h] - x[j[ok], i[ok]]
    return {"rows": int(ok.sum()), "mae_true_latent_change": _num(np.mean(np.abs(y - true_change))),
            "mae_zero_change": _num(np.mean(np.abs(y)))}


def evaluate(ds: Dataset) -> dict:
    obs, registry = ds.obs, ds.registry
    as_of = pd.Timestamp(obs["period"].max())
    summ = summarise_audit(signal_audit(obs, registry, as_of), registry).set_index("source_id")
    audit = {sid: {"entities": int(summ.loc[sid, "entities"]),
                   "median_dispersion_ratio": _num(summ.loc[sid, "median_dispersion_ratio"]),
                   "share_noise_like": _num(summ.loc[sid, "share_noise_like"])} for sid in summ.index}
    feats = build_feature_table(obs, registry, as_of)
    cols = feature_columns(feats)
    gates = {}
    for name in PROFILES:
        prof = C.GATE_PROFILES[name]
        folds, _ = walk_forward(feats, cols, profile=prof)
        gates[name] = _gate_summary(folds, evaluate_gate(folds, prof))
    imp = _importance(feats, cols, registry)
    abl = run_ablation(obs, registry, as_of, profile=C.GATE_PROFILES["demo"])
    ablation = {}
    for row in abl.to_dict("records"):
        ablation[row["config"]] = {"status": row["status"], "model_mae": _num(row.get("model_mae")),
                                   "best_baseline": row.get("best_baseline"),
                                   "best_baseline_mae": _num(row.get("best_baseline_mae")),
                                   "improvement_pct": _num(row.get("improvement_pct")),
                                   "n_features": None if pd.isna(row.get("n_features")) else int(row["n_features"])}
    floor = _noise_floor(feats, ds.truth)
    return {"audit": audit, "gates": gates, "importance": imp, "ablation": ablation, "noise_floor": floor}


def run_one(scenario: str, strength: float, seed: int, entities: int, periods: int, freq: str | None = None,
            snr: float = SNR) -> dict:
    t0 = time.perf_counter()
    ds = generate(scenario, strength, seed, entities=entities, periods=periods, freq=freq, snr=snr)
    if scenario == "S3":
        dup = duplicate_detection(ds.obs, ds.registry, ds.truth["duplicate_groups"])
        return {"scenario": scenario, "strength": float(strength), "seed": int(seed),
                "truth_duplicate_groups": len(ds.truth["duplicate_groups"]),
                "truth_duplicate_entities": sum(len(g) - 1 for g in ds.truth["duplicate_groups"]),
                "duplicate_check": dup, "runtime_s": round(time.perf_counter() - t0, 2)}
    res = evaluate(ds)
    return {"scenario": scenario, "strength": float(ds.truth["strength"]), "seed": int(seed),
            "runtime_s": round(time.perf_counter() - t0, 2), **res}


def plan(seeds: int, strengths: tuple[float, ...]) -> list[tuple[str, float, int]]:
    runs = [("S0", 0.0, s) for s in range(seeds)]
    runs += [("S1", float(k), s) for k in strengths if k != 0 for s in range(seeds)]
    runs += [("S2", float(k), s) for k in strengths for s in range(seeds)]
    runs += [("S3", 1.0, s) for s in range(seeds)]
    return runs


def _stat(values) -> dict:
    v = [float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n": len(v)}


def _rate(flags: list[bool]) -> dict:
    k, n = int(sum(flags)), len(flags)
    lo, hi = _wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [lo, hi]}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def aggregate(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        if r["scenario"] != "S3":
            groups.setdefault((r["scenario"], r["strength"]), []).append(r)
    rows = []
    for (scen, k), rs in sorted(groups.items()):
        row = {"scenario": scen, "name": SCENARIOS[scen], "strength": k, "n_runs": len(rs)}
        for p in PROFILES:
            g = [r["gates"][p] for r in rs]
            row[f"pass_{p}"] = _rate([x["passed"] for x in g])
            row[f"improvement_pct_{p}"] = _stat([x.get("improvement_pct") for x in g])
            row[f"model_mae_{p}"] = _stat([x.get("model_mae") for x in g])
            row[f"best_baseline_mae_{p}"] = _stat([x.get("best_baseline_mae") for x in g])
            row[f"best_baseline_{p}"] = Counter(x.get("best_baseline") for x in g if x.get("best_baseline")).most_common(1)[0][0] \
                if any(x.get("best_baseline") for x in g) else None
            row[f"folds_won_rate_{p}"] = _stat([x.get("fold_win_rate") for x in g])
            row[f"coverage_{p}"] = _stat([x.get("coverage") for x in g])
            row[f"direction_accuracy_{p}"] = _stat([x.get("direction_accuracy") for x in g])
            row[f"n_folds_{p}"] = _stat([x.get("n_folds") for x in g])
        row["noise_floor_mae"] = _stat([(r.get("noise_floor") or {}).get("mae_true_latent_change") for r in rs])
        row["zero_change_mae"] = _stat([(r.get("noise_floor") or {}).get("mae_zero_change") for r in rs])
        row["target_share_noise_like"] = _stat([r["audit"][TARGET_ID]["share_noise_like"] for r in rs])
        row["target_dispersion_ratio"] = _stat([r["audit"][TARGET_ID]["median_dispersion_ratio"] for r in rs])
        for sid in (TARGET_ID, *LEADS, "composite", "calendar_meta"):
            row[f"importance_{sid}"] = _stat([r["importance"]["share"].get(sid) for r in rs])
        row["lead_a_above_lead_b"] = _rate([r["importance"]["share"][LEAD_A] > r["importance"]["share"][LEAD_B]
                                            for r in rs])
        for cfg in CONFIGS:
            row[f"ablation_{cfg}_improvement_pct"] = _stat([r["ablation"].get(cfg, {}).get("improvement_pct")
                                                            for r in rs])
            row[f"ablation_{cfg}_model_mae"] = _stat([r["ablation"].get(cfg, {}).get("model_mae") for r in rs])
        rows.append(row)
    return rows



def _ms(stat: dict, nd=3, pct=False) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    sd = f" +- {stat['sd']:.{nd}f}" if stat.get("sd") is not None else ""
    if pct:
        return f"{stat['mean']:+.1f}%" + (f" +- {stat['sd']:.1f}" if stat.get("sd") is not None else "")
    return f"{stat['mean']:.{nd}f}{sd}"


def _pass_cell(rate: dict) -> str:
    lo, hi = rate["ci95"]
    return f"{rate['k']}/{rate['n']} ({rate['rate']:.0%}; 95% CI {lo:.2f}-{hi:.2f})"


def _bar(x: float | None, width: int = 30) -> str:
    if x is None:
        return ""
    n = round(x * width)
    return "#" * n + "." * (width - n)


def _gap_closed(row: dict) -> float | None:
    zero, floor, model = (row[k]["mean"] for k in ("zero_change_mae", "noise_floor_mae", "model_mae_demo"))
    if None in (zero, floor, model) or zero - floor <= 0:
        return None
    return (zero - model) / (zero - floor)


def _findings(agg: list[dict], runs: list[dict]) -> list[str]:
    out = []
    s0 = [r for r in agg if r["scenario"] == "S0"]
    if s0:
        s0 = s0[0]
        for p in PROFILES:
            out.append(f"Noise only (S0), {p} gate: passed {_pass_cell(s0[f'pass_{p}'])}. "
                       f"A pass here is a false positive.")
            hi = s0[f"pass_{p}"]["ci95"][1]
            if hi is not None and hi > 0.05:
                out.append(f"Noise only (S0), {p} gate: with {s0[f'pass_{p}']['n']} seeds the 95% upper bound on the "
                           f"false-positive rate is {hi:.0%}, so this run cannot show the rate is below 5%.")
    for scen in ("S1", "S2"):
        rows = [r for r in agg if r["scenario"] == scen]
        if not rows:
            continue
        for p in PROFILES:
            ok = [r for r in rows if r[f"pass_{p}"]["rate"] is not None and r[f"pass_{p}"]["rate"] >= 0.8]
            first = min(ok, key=lambda r: r["strength"])["strength"] if ok else None
            if first is None:
                out.append(f"{scen} ({SCENARIOS[scen]}), {p} gate: the pass rate never reached 80% on the strength grid.")
            else:
                out.append(f"{scen} ({SCENARIOS[scen]}), {p} gate: the pass rate first reached 80% at strength {first:g}.")
        top = max(rows, key=lambda r: r["strength"])
        closed = _gap_closed(top)
        if closed is not None:
            out.append(f"{scen} at the largest strength ({top['strength']:g}): the demo-gate model closes about "
                       f"{closed:.0%} of the gap between a no-change forecast and the error floor (the best achievable).")
        out.append(f"{scen} at the largest strength ({top['strength']:g}): demo-gate improvement over the best baseline "
                   f"{_ms(top['improvement_pct_demo'], pct=True)}; model MAE "
                   f"{_ms(top['model_mae_demo'], 4)} against best baseline {top['best_baseline_demo']} "
                   f"{_ms(top['best_baseline_mae_demo'], 4)}; error floor with perfect latent knowledge "
                   f"{_ms(top['noise_floor_mae'], 4)}.")
    s2 = [r for r in agg if r["scenario"] == "S2"]
    if s2:
        top = max(s2, key=lambda r: r["strength"])
        a, b = top[f"importance_{LEAD_A}"], top[f"importance_{LEAD_B}"]
        out.append(f"S2 at strength {top['strength']:g}: mean share of |SHAP| for the planted leading source "
                   f"{_ms(a, 3)} against the noise source {_ms(b, 3)}; the planted source has the larger share in "
                   f"{top[f'lead_a_above_lead_b']['k']}/{top[f'lead_a_above_lead_b']['n']} runs.")
        full, drop = top["ablation_all_sources_model_mae"], top[f"ablation_without_{LEAD_A}_model_mae"]
        if full["mean"] is not None and drop["mean"] is not None:
            out.append(f"S2 at strength {top['strength']:g}, leave-one-out: model MAE with all sources "
                       f"{full['mean']:.4f} against {drop['mean']:.4f} without the planted source. "
                       f"Improvement over the best baseline with all sources {_ms(top[f'ablation_all_sources_improvement_pct'], pct=True)}, "
                       f"without the planted source {_ms(top[f'ablation_without_{LEAD_A}_improvement_pct'], pct=True)}, "
                       f"target only {_ms(top['ablation_target_only_improvement_pct'], pct=True)}.")
    return out


def render_markdown(res: dict) -> str:
    m, agg, runs = res["meta"], res["aggregate"], res["runs"]
    L = []
    w = L.append
    w("# Synthetic simulation: positive and negative controls for the forecasting pipeline")
    w("")
    w("> **All data on this page is SYNTHETIC. It was generated from a known latent process and says NOTHING about "
      "the Malaysian labour market. It was never used to train, tune or validate the deployed model.**")
    w("")
    w(f"Generated {m['generated_at']} by `python3 -m src.validation.simulate` (code: `src/validation/simulate.py`). "
      f"Machine-readable results: `docs/simulation_results.json`. Runtime of this run: {m['runtime_s']:.0f} s.")
    w("")
    w("## 1. Purpose and why this is synthetic")
    w("")
    w("The Malaysian demo data is mostly noise, so the gate rejects the model, and that rejection cannot tell us "
      "whether the gate or the model is at fault. This simulation has known ground truth, so each question has an "
      "answer we can check:")
    w("")
    w("1. **False positives.** With no signal at all (S0), does the gate reject?")
    w("2. **Power.** With planted trends (S1) and a planted leading source (S2), does the model beat the baselines and "
      "pass the gate, and how much signal is needed?")
    w("3. **Use of the leading source.** Does the model learn the planted leading source (learned importance and the "
      "leave-one-source-out ablation)?")
    w("")
    w("The gate, the features, the baselines and the audit are the pipeline's own code, called unchanged. Only the "
      "data is generated. Nothing was tuned to make a result look better. If the model fails to detect a planted "
      "signal, the tables say so.")
    w("")
    w("## 2. Generator specification")
    w("")
    w(f"- Entities: {m['entities']} ({m['n_skill']} skills, {m['n_job']} jobs). Periods: {m['periods']} "
      f"{m['granularity']} periods from {m['start']}. Model grid: `PERIOD_FREQ={m['freq']}`, "
      f"horizon {m['horizon']} periods, gate profiles `production` and `demo` (both from `src/training/config.py`).")
    w(f"- Seeds: {m['seeds']} (0 to {m['seeds'] - 1}). Strengths: {', '.join(f'{s:g}' for s in m['strengths'])}. "
      f"Scenarios: S0 (no strength; k = 0), S1 and S2 on the strength grid, S3 generation only.")
    w("- Latent log-share of entity `e`, period `t`: `x[e,t] = a[e] + k * D[e,t]`. `a[e]` is a constant base share "
      "drawn uniformly from 1% to 4% of postings. `k` is the strength (0 in S0).")
    w("- `D` (S1, S2) is a damped trend with random regime changes. Velocity relaxes with persistence "
      f"PHI={PHI} toward a regime slope; the regime is redrawn with chance {REGIME_RATE:.4f} per period (about every "
      f"{1 / REGIME_RATE:.0f} periods) and the slope is drawn from N(0, {SIGMA_MU}). The level decays with RHO={RHO}.")
    w(f"- Observed share: counts are binomial draws, `count[e,t] ~ Binomial(N[t], exp(x[e,t]))`, with posting totals "
      f"`N[t]` = {POSTINGS} (target) or {LEAD_POSTINGS} (leading sources) times a log-normal jitter of sd "
      f"{TOTAL_JITTER}. Then `value = count / N[t]` and `support = count`. This keeps the sampling noise that the signal "
      "audit measures.")
    w("- Each entity's count is drawn independently. A posting can mention several skills, so shares need not sum to 1 "
      "(an approximation, stated for honesty).")
    w(f"- Leading source A (`{LEAD_A}`, labelled tier 1, reliability 0.8): in S2 it is a noisy copy that runs "
      f"{m['lead_lag']} periods ahead: `a[e] + k * D[e,t+{m['lead_lag']}] + eps`, `eps ~ N(0, {STD_UNIT / m['snr']:.2f}^2)` "
      f"(STD_UNIT={STD_UNIT} divided by snr={m['snr']}). In S0 and S1 it is pure sampling noise around `a[e]`.")
    w(f"- Leading source B (`{LEAD_B}`, tier 3, reliability 0.25): pure sampling noise around `a[e]` in every scenario.")
    w(f"- Target (`{TARGET_ID}`, tier 2, reliability 0.55): the counts described above.")
    w("- Registry: in memory, in the real schema (role, granularity, reliability, release lag, ffill, enabled). The "
      "labels are fixed across scenarios, so the simulation does not reveal which source carries the signal.")
    w(f"- Grid note: {m['grid_note']}")
    w(f"- S2 at k=0 is not a pure sampling-noise control for {LEAD_A}: its eps noise is added on the log scale, so the "
      "audit flags it as non-noise although it carries no information about the target.")
    w(f"- S3 (duplicates): {DUP_GROUPS} groups of {DUP_GROUP_SIZE} entities whose target counts are identical copies. "
      "Generation and the duplicate check only.")
    w("- Determinism: each component draws from its own stream keyed by (seed, component). The same seed gives "
      "the same draws in every scenario and strength (common random numbers), so strength changes only the trend and "
      "the planted leading signal.")
    w("")
    w("Gate profiles used (from the pipeline, unchanged):")
    w("")
    w("| Profile | min_train_periods | mae_margin | min_fold_win_rate | coverage range |")
    w("|---|---|---|---|---|")
    for p in PROFILES:
        c = C.GATE_PROFILES[p]
        w(f"| {p} | {c['min_train_periods']} | {c['mae_margin']} | {c['min_fold_win_rate']} | {tuple(c['coverage_range'])} |")
    w("")
    w("## 3. Results")
    w("")
    w("Mean +- sd over seeds unless stated. Pass rates are counts with a 95% Wilson interval. Improvement is "
      "`100 * (1 - model MAE / best baseline MAE)` on the walk-forward folds: positive means the model is better.")
    w("")
    w("### 3.1 Run inventory")
    w("")
    w("| Scenario | Strength | Runs | Folds (production / demo) | Target noise-like share | Target median dispersion |")
    w("|---|---|---|---|---|---|")
    for r in agg:
        w(f"| {r['scenario']} {r['name']} | {r['strength']:g} | {r['n_runs']} | "
          f"{_ms(r['n_folds_production'], 1)} / {_ms(r['n_folds_demo'], 1)} | "
          f"{_ms(r['target_share_noise_like'], 2)} | {_ms(r['target_dispersion_ratio'], 2)} |")
    w("")
    w("### 3.2 False-positive rate (S0, no signal)")
    w("")
    s0 = [r for r in agg if r["scenario"] == "S0"]
    if s0:
        s0 = s0[0]
        w("| Gate profile | Passed (false positives) | Best baseline (mode) | Improvement over best baseline | Folds won |")
        w("|---|---|---|---|---|")
        for p in PROFILES:
            w(f"| {p} | {_pass_cell(s0[f'pass_{p}'])} | {s0[f'best_baseline_{p}']} | "
              f"{_ms(s0[f'improvement_pct_{p}'], pct=True)} | {_ms(s0[f'folds_won_rate_{p}'], 2)} |")
    w("")
    w("### 3.3 Pass rate and improvement against strength")
    w("")
    w("| Scenario | Strength | Pass production | Pass demo | Improvement, demo folds | Model MAE (demo) | Best baseline MAE (demo) | Folds won | Coverage (demo) | Direction accuracy |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in agg:
        if r["scenario"] == "S0":
            continue
        w(f"| {r['scenario']} {r['name']} | {r['strength']:g} | {_pass_cell(r['pass_production'])} | "
          f"{_pass_cell(r['pass_demo'])} | {_ms(r['improvement_pct_demo'], pct=True)} | "
          f"{_ms(r['model_mae_demo'], 4)} | {r['best_baseline_demo']} {_ms(r['best_baseline_mae_demo'], 4)} | "
          f"{_ms(r['folds_won_rate_demo'], 2)} | {_ms(r['coverage_demo'], 2)} | {_ms(r['direction_accuracy_demo'], 2)} |")
    w("")
    w("### 3.3b Error floor (reference, not a forecaster)")
    w("")
    w("`Floor` is the MAE of predicting each observed 12-period target change with the TRUE latent change over the "
      "same horizon. That predictor uses the future, so it is unattainable. The floor is what sampling noise and "
      "smoothing leave even for a perfect forecaster of the latent. `Zero change` is the error of predicting no change.")
    w("")
    w("| Scenario | Strength | Floor (true latent change) | Zero change | Model MAE (demo) | Best baseline MAE (demo) | Gap closed by model |")
    w("|---|---|---|---|---|---|---|")
    for r in agg:
        closed = _gap_closed(r)
        w(f"| {r['scenario']} {r['name']} | {r['strength']:g} | {_ms(r['noise_floor_mae'], 4)} | "
          f"{_ms(r['zero_change_mae'], 4)} | {_ms(r['model_mae_demo'], 4)} | "
          f"{r['best_baseline_demo']} {_ms(r['best_baseline_mae_demo'], 4)} | "
          f"{'n/a' if closed is None else f'{closed:.0%}'} |")
    w("")
    w("`Gap closed` = (zero change - model) / (zero change - floor): the share of the reducible error that the demo "
      "model removes. 100% would match the floor.")
    w("")
    w("### 3.4 Power curve (demo gate, pass rate)")
    w("")
    w("Bars show the pass rate under the demo profile. The production rate is in table 3.3.")
    w("")
    w("```")
    for r in agg:
        if r["scenario"] == "S0":
            continue
        rate = r["pass_demo"]["rate"]
        w(f"{r['scenario']} k={r['strength']:<5g} |{_bar(rate)}| {rate:.0%}" if rate is not None else f"{r['scenario']} k={r['strength']:g} n/a")
    w("```")
    w("")
    w("### 3.5 Learned source importance (mean share of |SHAP|, in-sample on the full usable table)")
    w("")
    w("Shares sum to one across groups: target own history, the two leading sources, the composite of leading "
      "sources, and calendar and metadata columns. The composite is reported as the pipeline reports it.")
    w("")
    w(f"| Scenario | Strength | Target | {LEAD_A} (planted in S2) | {LEAD_B} (noise) | composite | calendar | {LEAD_A} > {LEAD_B} |")
    w("|---|---|---|---|---|---|---|---|")
    for r in agg:
        w(f"| {r['scenario']} {r['name']} | {r['strength']:g} | {_ms(r[f'importance_{TARGET_ID}'], 3)} | "
          f"{_ms(r[f'importance_{LEAD_A}'], 3)} | {_ms(r[f'importance_{LEAD_B}'], 3)} | "
          f"{_ms(r['importance_composite'], 3)} | {_ms(r['importance_calendar_meta'], 3)} | "
          f"{r['lead_a_above_lead_b']['k']}/{r['lead_a_above_lead_b']['n']} |")
    w("")
    w("### 3.6 Leave-one-source-out ablation (demo gate folds)")
    w("")
    w("Improvement over the best baseline of each configuration. `without_sim_lead_a` removes the planted source. "
      "Positive means that configuration beats the baselines on these folds.")
    w("")
    w("| Scenario | Strength | all sources | without lead A (planted) | without lead B (noise) | target only |")
    w("|---|---|---|---|---|---|")
    for r in agg:
        w(f"| {r['scenario']} {r['name']} | {r['strength']:g} | "
          f"{_ms(r['ablation_all_sources_improvement_pct'], pct=True)} | "
          f"{_ms(r[f'ablation_without_{LEAD_A}_improvement_pct'], pct=True)} | "
          f"{_ms(r[f'ablation_without_{LEAD_B}_improvement_pct'], pct=True)} | "
          f"{_ms(r['ablation_target_only_improvement_pct'], pct=True)} |")
    w("")
    w("### 3.7 Signal audit")
    w("")
    w("Dispersion ratio = observed variance of the share over the variance expected from sampling alone. About 1 "
      "means indistinguishable from noise. The noise-like flag uses the pipeline's threshold.")
    w("")
    w("| Scenario | Strength | Target median dispersion | Target noise-like share | Lead A dispersion | Lead B dispersion |")
    w("|---|---|---|---|---|---|")
    for r in agg:
        w(f"| {r['scenario']} {r['name']} | {r['strength']:g} | {_ms(r['target_dispersion_ratio'], 2)} | "
          f"{_ms(r['target_share_noise_like'], 2)} | "
          f"{_ms(_stat([x['audit'][LEAD_A]['median_dispersion_ratio'] for x in runs if x['scenario'] == r['scenario'] and x['strength'] == r['strength']]), 2)} | "
          f"{_ms(_stat([x['audit'][LEAD_B]['median_dispersion_ratio'] for x in runs if x['scenario'] == r['scenario'] and x['strength'] == r['strength']]), 2)} |")
    w("")
    s3 = [r for r in runs if r["scenario"] == "S3"]
    if s3:
        dups = s3[0]
        det = dups["duplicate_check"]
        w(f"**S3 duplicates.** Ground truth: {dups['truth_duplicate_groups']} groups, "
          f"{dups['truth_duplicate_entities']} duplicate entities per seed. Duplicate detection: "
          f"{'not run, the audit module has no duplicate_groups function' if det['detector'] == 'not_available' else json.dumps(det)}.")
        w("")
    w("## 4. Honest reading")
    w("")
    for line in _findings(agg, runs):
        w(f"- {line}")
    w("")
    w("What these controls can and cannot show:")
    w("")
    w("- **The generator is ours.** Every planted pattern matches the model's assumptions (damped trends, a leading "
      "source that is a lagged copy). Passing here shows the machinery works when its assumptions hold. It is not "
      "evidence that the forecast works on labour-market data.")
    w("- **The leading source's quality label is correlated with its truth** (lead A is both labelled higher quality "
      "and carries the signal). This flatters the composite. A real test would hide the labels.")
    w("- **Few seeds.** With the default 5 seeds, a false-positive rate has a resolution of 20 points and a 95% interval "
      "about +-40 points wide. Use the Wilson intervals, and raise `--seeds` for a finer estimate.")
    w("- **Importance is in-sample** on the full usable table, as the pipeline reports it. It shows what the fitted "
      "model relies on, not out-of-sample value. The ablation is the out-of-sample check.")
    w("- **Sample size.** 130 periods with 12-period horizon and 13-period folds leaves few production folds: "
      "the production profile needs 52 training periods, so only the earliest folds qualify. Table 3.1 shows the "
      "fold counts.")
    w("- **No tuning.** The model, features, baselines and gate are unchanged. The generator parameters were fixed "
      "before the run and are listed in section 2. Changing them changes the power curve, so the thresholds shown "
      "are for these settings only.")
    w("- **Reading the tables.** Read the pass rates together with the error floor. A model can fail the gate while still "
      "being far from the floor, and it can pass only by beating a baseline that itself is only a rough guess.")
    w("- **The planted lag is not identifiable from observed correlations.** The latent moves slowly, so the correlation "
      "of an observed leading source with the latent is nearly flat across lags 0 to 16. The lag is checked exactly "
      "on the noise-free planted lead (tests/test_simulation.py), not on the noisy observations.")
    w("- **Error floor.** The floor in table 3.3b is unattainable for any forecaster: it uses the future latent. The gap "
      "between the model and the best baseline is measured against it only to show how much error sampling noise "
      "leaves, not as a target.")
    w("- **Single entity count, single period grid, single leading lag** (lag 8 weeks). Power at other lags or with "
      "more entities is not tested.")
    w("")
    w("## 5. How to reproduce")
    w("")
    w("```bash")
    w("python3 -m src.validation.simulate                     # full run, writes docs/SIMULATION.md and docs/simulation_results.json")
    w("python3 -m src.validation.simulate --seeds 10 --out /path/to/elsewhere   # more seeds, other output folder")
    w("python3 -m src.validation.simulate --quick --out /path/to/elsewhere     # tiny grid, 2 seeds")
    w("python3 -m pytest -q tests/test_simulation.py")
    w("```")
    w("")
    w("The run is deterministic for fixed arguments. The gate and the features use the pipeline's default weekly grid "
      "unless `PERIOD_FREQ=M` is set in the environment.")
    return "\n".join(L) + "\n"


def _run_spec(args: tuple) -> dict:
    scen, k, seed, entities, periods, freq, snr = args
    return run_one(scen, k, seed, entities, periods, freq=freq, snr=snr)


def run_simulation(seeds: int = DEFAULT_SEEDS, entities: int = DEFAULT_ENTITIES, periods: int = DEFAULT_PERIODS,
                   strengths: tuple[float, ...] = DEFAULT_STRENGTHS, freq: str | None = None,
                   snr: float = SNR, progress: bool = False, workers: int = DEFAULT_WORKERS) -> dict:
    f = _check_freq(freq)
    t0 = time.perf_counter()
    specs = [(scen, k, s, entities, periods, f, snr) for scen, k, s in plan(seeds, tuple(strengths))]
    runs = []
    if workers <= 1:
        results = (_run_spec(sp) for sp in specs)
        for r in results:
            runs.append(r)
            if progress:
                print(f"  {r['scenario']} k={r['strength']:g} seed={r['seed']}: {r['runtime_s']:.1f} s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(_run_spec, specs):
                runs.append(r)
                if progress:
                    print(f"  {r['scenario']} k={r['strength']:g} seed={r['seed']}: {r['runtime_s']:.1f} s "
                          "(own wall time)", flush=True)
    if 0.0 in strengths:
        for r0 in [r for r in runs if r["scenario"] == "S0"]:
            runs.append({**r0, "scenario": "S1", "strength": 0.0, "reused_from": "S0", "runtime_s": 0.0})
    agg = aggregate(runs)
    n_skill = round(entities * SKILL_FRACTION)
    start = START[f]
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(), "runtime_s": round(time.perf_counter() - t0, 1),
            "entities": entities, "n_skill": n_skill, "n_job": entities - n_skill, "periods": periods,
            "granularity": GRANULARITY[f], "start": start, "freq": f, "horizon": C.HORIZON, "seeds": seeds,
            "strengths": list(strengths), "lead_lag": LEAD_LAG, "snr": snr, "std_unit": STD_UNIT,
            "phi": PHI, "rho": RHO, "regime_rate": REGIME_RATE, "sigma_mu": SIGMA_MU, "base_share": list(BASE_SHARE),
            "workers": workers, "grid_note": ("S1 at k=0 is not re-run; its records are the S0 records (identical by "
                                              "construction). Strength 0.25 was dropped to keep the run within budget."),
            "postings": POSTINGS, "lead_postings": LEAD_POSTINGS, "total_jitter": TOTAL_JITTER,
            "disclaimer": SYNTHETIC_DISCLAIMER, "scenarios": SCENARIOS,
            "gate_profiles": {p: {k: v for k, v in C.GATE_PROFILES[p].items() if k != "note"} for p in PROFILES}}
    return {"meta": meta, "aggregate": agg, "runs": runs}


def write_outputs(res: dict, out_dir: Path) -> tuple[Path, Path]:
    out = Path(out_dir).resolve()
    protected = [ROOT / "data", ROOT / "serving", ROOT / "src" / "models", C.DATA_DIR, C.MODELS_DIR]
    for p in protected:
        if out == p.resolve() or p.resolve() in out.parents or out in p.resolve().parents:
            raise ValueError(f"refusing to write simulation outputs into pipeline state: {out}")
    out.mkdir(parents=True, exist_ok=True)
    md, js = out / "SIMULATION.md", out / "simulation_results.json"
    js.write_text(json.dumps(_jsonable(res), indent=2))
    md.write_text(render_markdown(res))
    return md, js


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Synthetic simulation with known ground truth (positive and negative "
                                             "controls). All data is synthetic; nothing here is about Malaysia.")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--entities", type=int, default=DEFAULT_ENTITIES)
    ap.add_argument("--periods", type=int, default=DEFAULT_PERIODS)
    ap.add_argument("--freq", default=None, help=f"must match the pipeline grid (PERIOD_FREQ, now {C.PERIOD_FREQ})")
    ap.add_argument("--quick", action="store_true", help="tiny grid (strengths 0 and 1, 2 seeds) for tests")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="threads running independent runs")
    ap.add_argument("--out", default=None, help="output folder (default docs/; with --quick, nothing is written "
                                               "unless --out is given)")
    a = ap.parse_args(argv)
    seeds = QUICK_SEEDS if a.quick else a.seeds
    strengths = QUICK_STRENGTHS if a.quick else DEFAULT_STRENGTHS
    print(f"Synthetic simulation (all data synthetic): seeds={seeds} entities={a.entities} periods={a.periods} "
          f"strengths={list(strengths)} grid={C.PERIOD_FREQ}", flush=True)
    res = run_simulation(seeds=seeds, entities=a.entities, periods=a.periods, strengths=strengths, freq=a.freq,
                         progress=True, workers=a.workers)
    print("\n".join(_findings(res["aggregate"], res["runs"])))
    out = a.out if a.out else (None if a.quick else str(ROOT / "docs"))
    if out:
        md, js = write_outputs(res, Path(out))
        print(f"wrote {md}\nwrote {js}")


if __name__ == "__main__":
    main()
