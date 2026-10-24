import numpy as np
import pandas as pd

from src.sources import loaders
from src.training.ablation import run_ablation
from src.training.audit import signal_audit, source_coverage, summarise_audit
from src.training.backtest import walk_forward
from src.training.config import GATE_PROFILES
from src.training.features import build_feature_table, feature_columns

START = "2021-01-04"

SMALL = {**GATE_PROFILES["demo"], "name": "test", "n_folds": 2, "fold_step": 4, "min_train_periods": 4, "min_folds": 1}
REG = {"sources": [
    {"source_id": "p", "role": "target", "granularity": "weekly", "reliability": 1.0},
    {"source_id": "g", "role": "leading", "enabled": True, "granularity": "weekly", "reliability": 0.8,
     "release_lag_days": 0, "max_staleness_days": 400},
    {"source_id": "j", "role": "leading", "enabled": True, "granularity": "weekly", "reliability": 0.8,
     "release_lag_days": 0, "max_staleness_days": 400},
]}


def _series(entity, shares, start=START, n=500, source="p", kind="skill"):
    weeks = pd.date_range(start, periods=len(shares), freq="W-MON")
    return pd.DataFrame({"entity_id": entity, "kind": kind, "period": weeks, "source_id": source,
                         "value": shares, "support": [round(s * n) for s in shares]}, columns=loaders.COLUMNS)


def test_audit_flags_noise_and_trend():
    noise = [0.10, 0.12, 0.09, 0.11] * 4
    trend = list(np.linspace(0.05, 0.30, 12))
    obs = pd.concat([_series("flat", noise, n=500), _series("rising", trend, n=500)], ignore_index=True)
    audit = signal_audit(obs, REG).set_index("entity_id")
    assert bool(audit.loc["flat", "noise_like"]) is True
    assert bool(audit.loc["rising", "noise_like"]) is False
    assert audit.loc["rising", "dispersion_ratio"] > audit.loc["flat", "dispersion_ratio"]
    assert audit.loc["rising", "abs_corr_with_time"] > 0.95
    summary = summarise_audit(signal_audit(obs, REG), REG).set_index("source_id")
    assert summary.loc["p", "entities"] == 2 and summary.loc["p", "share_noise_like"] == 0.5
    assert summary.loc["p", "periods_span"] == 16


def test_audit_ignores_data_after_as_of():
    obs = _series("rising", list(np.linspace(0.05, 0.30, 12)))
    early = signal_audit(obs, REG, as_of="2021-02-15")
    assert early.loc[0, "n_periods"] == 7 and early.loc[0, "last_period"] == "2021-02-15"


def test_coverage_counts_overlap_with_target_in_periods():
    obs = pd.concat([_series("a", [0.1] * 6, source="p"),
                     _series("a", [0.1] * 3, start="2021-01-18", source="g"),
                     _series("a", [0.1] * 3, start="2030-01-07", source="j")], ignore_index=True)
    cov = source_coverage(obs, REG).set_index("source_id")
    assert cov.loc["g", "overlap_periods"] == 3 and cov.loc["j", "overlap_periods"] == 0
    assert cov.loc["j", "periods"] == 3 and cov.loc["g", "first_period"] == "2021-01-18"


def test_walk_forward_reports_all_baselines():
    shares = {f"e{i}": [0.05 + 0.002 * t + 0.001 * ((t + i) % 3) for t in range(80)] for i in range(2)}
    obs = pd.concat([_series(e, v) for e, v in shares.items()], ignore_index=True)
    feats = build_feature_table(obs, REG, as_of="2022-06-06")
    assert "lvl_dev" in feature_columns(feats)
    folds, preds = walk_forward(feats, feature_columns(feats), profile=SMALL)
    assert len(folds) > 0
    for name in ("mae_persistence", "mae_trend", "mae_damped_trend", "mae_mean_reversion", "mae_half_reversion"):
        assert name in folds.columns
    assert "mean_reversion" in preds.columns and "half_reversion" in preds.columns


def test_ablation_rows_and_no_overlap_status():
    wiggle = np.where(np.arange(60) % 2 == 0, 0.004, -0.004)
    base = np.linspace(0.05, 0.12, 60) + wiggle
    obs = pd.concat([
        _series("e1", list(base)),
        _series("e2", list(base * 1.1)),
        _series("e1", [0.02] * 60, source="g"),
        _series("e1", [0.03] * 5, start="2030-01-07", source="j"),
    ], ignore_index=True)
    res = run_ablation(obs, REG, profile=SMALL).set_index("config")
    assert list(res.index) == ["all_sources", "without_g", "without_j", "target_only"]
    assert res.loc["without_j", "status"] == "no_overlap" and res.loc["without_j", "overlap_j"] == 0
    assert res.loc["without_g", "status"] == "ok" and res.loc["target_only", "status"] == "ok"
    assert res.loc["all_sources", "n_features"] > res.loc["target_only", "n_features"]
    assert set(res["gate_profile"].dropna()) == {"test"}
