import pandas as pd

from src.training import config as C
from src.training.backtest import evaluate_gate


def test_weekly_period_helpers():
    assert C.to_period_start("2024-01-03") == pd.Timestamp("2024-01-01")
    assert C.to_period_start("2024-01-07") == pd.Timestamp("2024-01-01")
    assert C.shift_periods("2024-01-03", 2) == pd.Timestamp("2024-01-15")
    assert C.shift_periods("2024-01-03", -1) == pd.Timestamp("2023-12-25")
    assert C.periods_between("2024-01-01", "2024-03-04") == 9
    assert list(C.period_range("2024-01-03", "2024-01-22")) == [pd.Timestamp(d) for d in
                                                                ("2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22")]
    assert C.period_label(pd.Timestamp("2024-01-29")) == "2024-01-29"
    assert C.parse_period_label("2024-01-05") == pd.Timestamp("2024-01-01")


def test_span_end_by_granularity():
    assert C.span_end(pd.Timestamp("2024-01-01"), "weekly") == pd.Timestamp("2024-01-08")
    assert C.span_end(pd.Timestamp("2024-01-01"), "monthly") == pd.Timestamp("2024-02-01")
    assert C.span_end(pd.Timestamp("2024-01-01"), "quarterly") == pd.Timestamp("2024-04-01")
    assert C.span_end(pd.Timestamp("2024-01-01"), "annual") == pd.Timestamp("2025-01-01")


def test_gate_profiles_differ_only_in_constraints():
    prod, demo = C.GATE_PROFILES["production"], C.GATE_PROFILES["demo"]
    assert set(prod) == set(demo)
    assert {k for k in prod if prod[k] != demo[k]} == {"min_train_periods", "mae_margin", "min_fold_win_rate",
                                                       "coverage_range", "note", "name"}
    assert C.ACTIVE_GATE_PROFILE in C.GATE_PROFILES


def _folds(model, base, coverage):
    n = len(model)
    return pd.DataFrame({
        "fold": range(n), "n_test": [100] * n, "mae_model": model, "mae_persistence": base,
        "mae_trend": base, "mae_damped_trend": base, "mae_mean_reversion": base, "mae_half_reversion": base,
        "coverage": coverage,
    })


def test_same_candidate_passes_demo_and_fails_production():

    folds = _folds(model=[0.9, 0.9, 1.2], base=[1.0, 1.0, 1.0], coverage=[0.6, 0.6, 0.6])
    demo = evaluate_gate(folds, C.GATE_PROFILES["demo"])
    prod = evaluate_gate(folds, C.GATE_PROFILES["production"])
    assert demo["profile"] == "demo" and demo["passed"] is True
    assert prod["profile"] == "production" and prod["passed"] is False
    assert prod["constraints"]["mae_margin"] == 0.03 and demo["constraints"]["mae_margin"] == 0.0

    assert prod["checks"] == {"beats_baseline": False, "fold_win_rate": True, "calibrated": False}


def test_gate_rejects_too_few_folds_under_any_profile():
    res = evaluate_gate(_folds([0.5], [1.0], [0.8]).iloc[:1], C.GATE_PROFILES["demo"])
    assert res["passed"] is False and "usable folds" in res["reason"] and res["profile"] == "demo"
