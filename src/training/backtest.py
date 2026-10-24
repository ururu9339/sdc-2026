"""Walk-forward validation and the promotion gate.

The gate is fixed and simple: the candidate must beat the best naive baseline on time-ordered
held-out folds, under the constraints of a gate profile (config.GATE_PROFILES). The profile only
changes the numeric constraints, never the rule. The model never influences its own gate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.baselines import predict_baselines
from src.training.config import GATE_KEYS, HORIZON, active_gate_profile, shift_periods
from src.training.features import usable
from src.training.train import fit_models, predict


def walk_forward(df: pd.DataFrame, cols: list[str], horizon: int = HORIZON, profile: dict | None = None):
    """Returns (per-fold metrics, per-row predictions). Train rows must have their target known
    by the fold's information cutoff, so nothing from the test period leaks into training."""
    p = profile or active_gate_profile()
    rows = usable(df)
    labelled = rows[rows["y"].notna()]
    origins = sorted(labelled["period"].unique())
    step = p["fold_step"]
    folds, preds = [], []
    for i in range(p["n_folds"]):
        end = len(origins) - i * step
        start = end - step
        if start <= 0:
            break
        test_origins = origins[start:end]
        cutoff = shift_periods(test_origins[0], -1)
        train = labelled[labelled["period"] <= shift_periods(cutoff, -horizon)]
        if train["period"].nunique() < p["min_train_periods"]:
            continue
        test = labelled[labelled["period"].isin(test_origins)]
        models = fit_models(train, cols)
        pr = predict(models, test[cols])
        base = predict_baselines(test, horizon)
        res = pd.DataFrame({"entity_id": test["entity_id"].to_numpy(), "period": test["period"].to_numpy(),
                            "y": test["y"].to_numpy(), **{c: pr[c].to_numpy() for c in pr}})
        for name, b in base.items():
            res[name] = b
        res["fold"] = len(folds)
        preds.append(res)
        err = lambda c: float(np.mean(np.abs(res["y"] - res[c])))  # noqa: E731
        folds.append({
            "fold": len(folds), "cutoff": cutoff, "n_test": len(res), "n_train": len(train),
            "mae_model": err("p50"), **{f"mae_{n}": err(n) for n in base},
            "coverage": float(((res["y"] >= res["p10"]) & (res["y"] <= res["p90"])).mean()),
            "dir_acc_model": float((np.sign(res["p50"]) == np.sign(res["y"])).mean()),
        })
    return pd.DataFrame(folds), (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())


def evaluate_gate(folds: pd.DataFrame, profile: dict | None = None) -> dict:
    p = profile or active_gate_profile()
    head = {"profile": p["name"], "constraints": {k: p[k] for k in GATE_KEYS}}
    if len(folds) < p["min_folds"]:
        return {"passed": False, "reason": f"only {len(folds)} usable folds (need {p['min_folds']}); not enough history",
                **head}
    base_cols = [c for c in folds.columns if c.startswith("mae_") and c != "mae_model"]
    w = folds["n_test"]
    mean = lambda c: float(np.average(folds[c], weights=w))  # noqa: E731
    best = min(base_cols, key=mean)
    model_mae, best_mae = mean("mae_model"), mean(best)
    win_rate = float((folds["mae_model"] < folds[best]).mean())
    cov = float(np.average(folds["coverage"], weights=w))
    lo, hi = p["coverage_range"]
    checks = {
        "beats_baseline": model_mae <= (1 - p["mae_margin"]) * best_mae,
        "fold_win_rate": win_rate >= p["min_fold_win_rate"],
        "calibrated": lo <= cov <= hi,
    }
    return {
        **head, "passed": all(checks.values()), "checks": checks, "n_folds": len(folds),
        "model_mae": model_mae, "best_baseline": best.removeprefix("mae_"), "best_baseline_mae": best_mae,
        "improvement_pct": 100 * (1 - model_mae / best_mae) if best_mae else None,
        "fold_win_rate": win_rate, "coverage": cov,
        "reason": "ok" if all(checks.values()) else "failed: " + ", ".join(k for k, v in checks.items() if not v),
    }
