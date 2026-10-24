from __future__ import annotations

import pandas as pd

from src.training.audit import source_coverage
from src.training.backtest import evaluate_gate, walk_forward
from src.training.config import leading_sources, target_source
from src.training.features import build_feature_table, feature_columns


def run_ablation(obs: pd.DataFrame, registry: dict, as_of=None, profile: dict | None = None) -> pd.DataFrame:
    tgt = target_source(registry)
    cov = source_coverage(obs, registry, as_of)
    if as_of is None:
        as_of = cov.loc[cov["source_id"] == tgt, "last_period"].iloc[0]
    as_of = pd.Timestamp(as_of)
    cov = source_coverage(obs, registry, as_of).set_index("source_id")
    leads = [s["source_id"] for s in leading_sources(registry)]
    present = [s for s in leads if cov.loc[s, "periods"] > 0] if leads else []
    overlap = {s: int(cov.loc[s, "overlap_periods"]) if s in cov.index else 0 for s in leads}

    configs = [("all_sources", None, "none")]
    configs += [(f"without_{s}", [x for x in present if x != s], s) for s in leads]
    configs.append(("target_only", [], "all_leading"))

    rows = []
    for name, active, dropped in configs:
        row = {"config": name, "dropped": dropped, **{f"overlap_{s}": overlap[s] for s in leads}}
        if dropped not in ("none", "all_leading") and overlap[dropped] == 0:
            rows.append({**row, "status": "no_overlap", "n_features": None, "n_folds": None, "model_mae": None,
                         "best_baseline": None, "best_baseline_mae": None, "improvement_pct": None,
                         "passed": None, "gate_profile": None})
            continue
        feats = build_feature_table(obs, registry, as_of, active=None if active is None else set(active))
        cols = feature_columns(feats)
        folds, _ = walk_forward(feats, cols, profile=profile)
        gate = evaluate_gate(folds, profile)
        rows.append({**row, "status": "ok", "n_features": len(cols), "n_folds": len(folds),
                     "model_mae": gate.get("model_mae"), "best_baseline": gate.get("best_baseline"),
                     "best_baseline_mae": gate.get("best_baseline_mae"),
                     "improvement_pct": gate.get("improvement_pct"), "passed": bool(gate["passed"]),
                     "gate_profile": gate["profile"]})
    out = pd.DataFrame(rows)
    return out.astype({"n_features": "Int64", "n_folds": "Int64"})
