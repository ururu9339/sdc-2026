"""Hand-written fixtures for the read-only forecast API: a tiny registry, observations, job-skill graph,
version log, forecast and preview logs (optional). Entity ids use the taxonomy form ('job:...', 'skill:...').
Everything lives in tmp_path and the config paths are pointed at it with monkeypatch, so the real data/ and
src/models/ are never read."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.pipeline.store import AppendOnlyLog
from src.training import config as C

TEST_DISCLAIMER = "Test disclaimer: fixture data, for demonstration only."
GATE_FAIL = {
    "passed": False, "reason": "failed: beats_baseline, fold_win_rate", "profile": "demo",
    "checks": {"beats_baseline": False, "fold_win_rate": False, "calibrated": True},
    "constraints": {"n_folds": 6, "fold_step": 13, "min_train_periods": 26, "min_folds": 3,
                    "mae_margin": 0.0, "min_fold_win_rate": 0.5, "coverage_range": [0.55, 0.98]},
    "n_folds": 6, "model_mae": 0.031, "best_baseline": "persistence", "best_baseline_mae": 0.027,
    "improvement_pct": -14.8, "fold_win_rate": 0.33, "coverage": 0.81,
}
GATE_PASS = {**GATE_FAIL, "passed": True, "reason": "ok", "checks": {"beats_baseline": True,
             "fold_win_rate": True, "calibrated": True}, "model_mae": 0.024, "improvement_pct": 11.1,
             "fold_win_rate": 0.67}

REGISTRY = {
    "version": 1,
    "disclaimer": TEST_DISCLAIMER,
    "sources": [
        {"source_id": "job_postings", "name": "Fixture postings", "role": "target", "enabled": True, "tier": 2,
         "reliability": 0.55, "granularity": "weekly", "release_lag_days": 0, "license": "fixture licence",
         "attribution": "fixture", "source_url": "https://example.test/postings", "loader": {"type": "csv"}},
        {"source_id": "market_signal", "name": "Fixture survey", "role": "leading", "enabled": True, "tier": 3,
         "reliability": 0.25, "granularity": "monthly", "release_lag_days": 0, "license": "fixture licence",
         "attribution": "fixture", "source_url": "https://example.test/survey", "loader": {"type": "csv"}},
        {"source_id": "jobstreet_my", "name": "Fixture listings", "role": "leading", "enabled": True, "tier": 1,
         "reliability": 0.8, "granularity": "weekly", "release_lag_days": 0, "license": "CC BY 4.0",
         "attribution": "fixture", "source_url": "https://example.test/listings", "loader": {"type": "csv"}},
    ],
}

WEEKS = pd.date_range("2026-07-06", periods=6, freq="W-MON")  # six Mondays
MONTHS = pd.to_datetime(["2026-06-01", "2026-07-01", "2026-08-01"])


def _observations() -> pd.DataFrame:
    rows = []

    def add(ent, kind, periods, values, supports, source):
        for p, v, s in zip(periods, values, supports):
            rows.append({"entity_id": ent, "kind": kind, "period": p.strftime("%Y-%m-%d"),
                         "source_id": source, "value": v, "support": s})

    add("job:accountant", "job", WEEKS, [0.010, 0.011, 0.012, 0.011, 0.010, 0.012], [100, 110, 120, 115, 105, 118],
        "job_postings")
    add("job:ai_engineer", "job", WEEKS, [0.004, 0.005, 0.007, 0.009, 0.011, 0.013], [60, 62, 70, 80, 90, 95],
        "job_postings")
    add("job:business_analyst", "job", WEEKS, [0.020, 0.019, 0.021, 0.020, 0.022, 0.021],
        [200, 190, 205, 199, 210, 208], "job_postings")
    add("skill:accounting", "skill", WEEKS, [0.030, 0.031, 0.029, 0.030, 0.032, 0.031], [300, 310, 290, 300, 320, 310],
        "job_postings")
    add("skill:python", "skill", WEEKS, [0.040, 0.041, 0.043, 0.045, 0.046, 0.048], [400, 410, 430, 450, 460, 480],
        "job_postings")
    add("job:accountant", "job", MONTHS, [0.05, 0.06, 0.055], [400, 420, 410], "market_signal")
    return pd.DataFrame(rows)


GRAPH = pd.DataFrame([
    ("job:accountant", "skill:accounting", 10, 10, 1.0),
    ("job:accountant", "skill:excel", 7, 10, 0.7),
    ("job:accountant", "skill:python", 4, 10, 0.4),
    ("job:ai_engineer", "skill:python", 9, 9, 1.0),
    ("job:ai_engineer", "skill:accounting", 1, 9, 0.11),
], columns=["job", "skill", "n_cooccur", "n_job_postings", "weight"])


def _record(**kw) -> dict:
    base = {"model_version": "v1", "as_of": "2026-09-14", "target_period": "2026-12-07", "horizon_periods": 12,
            "kind": "job", "gate_profile": "demo", "disclaimer": TEST_DISCLAIMER,
            "baseline_persistence": 0.01, "baseline_trend": 0.02}
    return {**base, **kw}


def _forecast_records() -> list[dict]:
    older = _record(forecast_id="v1:job:accountant:2026-09-07", as_of="2026-09-07", entity_id="job:accountant",
                    status="ok", reason=None, p50=0.90, p10=0.80, p90=1.00, direction="emerging",
                    confidence="high", signal_quality="trend_detected", dispersion_ratio=2.0,
                    source_contributions={"market_signal": 0.9})
    stable = _record(forecast_id="v1:job:accountant:2026-09-14", entity_id="job:accountant", status="ok",
                     reason=None, p50=0.01, p10=-0.08, p90=0.10, direction="stable", confidence="low",
                     signal_quality="noise_like", dispersion_ratio=1.02,
                     source_contributions={"job_postings": 0.006, "market_signal": 0.004})
    emerging = _record(forecast_id="v1:job:ai_engineer:2026-09-14", entity_id="job:ai_engineer", status="ok",
                       reason=None, p50=0.25, p10=0.10, p90=0.45, direction="emerging", confidence="high",
                       signal_quality="trend_detected", dispersion_ratio=2.3,
                       source_contributions={"job_postings": 0.12, "market_signal": 0.05})
    insufficient = _record(forecast_id="v1:job:business_analyst:2026-09-14", entity_id="job:business_analyst",
                           status="insufficient_data", reason="history_too_short (6 < 26 periods)",
                           signal_quality="unknown", dispersion_ratio=None)
    return [older, stable, emerging, insufficient]


PREVIEW_REASON = "failed: beats_baseline, fold_win_rate"


def _preview_records() -> list[dict]:
    """Unvalidated candidate forecasts, as the agent loop writes them to the preview log."""
    common = dict(model_version="preview:v1", feed="preview", validated=False, preview_reason=PREVIEW_REASON,
                  gate_profile="demo")
    return [
        _record(**common, forecast_id="preview:v1:job:accountant:2026-09-14", as_of="2026-09-14",
                entity_id="job:accountant", status="ok", reason=None, p50=0.02, p10=-0.05, p90=0.09,
                direction="stable", confidence="low", signal_quality="noise_like", dispersion_ratio=1.1,
                source_contributions={"job_postings": 0.01, "market_signal": 0.01}),
        _record(**common, forecast_id="preview:v1:job:ai_engineer:2026-09-14", as_of="2026-09-14",
                entity_id="job:ai_engineer", status="ok", reason=None, p50=0.25, p10=0.10, p90=0.45,
                direction="emerging", confidence="high", signal_quality="trend_detected", dispersion_ratio=2.3,
                source_contributions={"job_postings": 0.12, "market_signal": 0.05}),
        _record(**common, forecast_id="preview:v1:job:business_analyst:2026-09-14", as_of="2026-09-14",
                entity_id="job:business_analyst", status="insufficient_data",
                reason="history_too_short (6 < 26 periods)", signal_quality="unknown", dispersion_ratio=None),
    ]


def build_env(tmp_path, monkeypatch, live: bool, preview: bool = False) -> dict:
    data = tmp_path / "data"
    processed = data / "processed"
    models = tmp_path / "models"
    processed.mkdir(parents=True)
    models.mkdir()
    reg_path = tmp_path / "sources_registry.json"
    reg_path.write_text(json.dumps(REGISTRY))
    obs_path = processed / "observations.csv"
    _observations().to_csv(obs_path, index=False)
    GRAPH.to_csv(processed / "job_skill_graph.csv", index=False)
    vlog = {"live": "v1" if live else None, "history": [
        {"created_at": "2026-09-14T00:00:00+00:00", "as_of": "2026-09-14", "gate_profile": "demo",
         "gate": GATE_PASS if live else GATE_FAIL, "folds": [], "sources_used": ["market_signal"], "audit": [
            {"source_id": "job_postings", "entities": 3, "median_dispersion_ratio": 1.02,
             "share_noise_like": 1.0, "first_period": "2026-07-06", "last_period": "2026-08-10", "periods_span": 6}],
         "version": "v1" if live else None, "status": "promoted" if live else "rejected",
         **({"source_importance": {"job_postings": 0.02, "market_signal": 0.005}, "n_train_rows": 300} if live else {})}
    ]}
    if preview:
        vlog["preview"] = "preview_v1"
        vlog["history"].append({"created_at": "2026-09-14T00:00:00+00:00", "as_of": "2026-09-14",
                                "gate_profile": "demo", "gate": GATE_FAIL, "folds": [], "version": None,
                                "status": "rejected_preview", "preview_version": "preview:v1",
                                "preview_reason": PREVIEW_REASON, "model_dir": "preview_v1"})
    vlog_path = models / "version_log.json"
    vlog_path.write_text(json.dumps(vlog))
    report_path = data / "model_report.json"
    report_path.write_text(json.dumps({"generated_at": "2026-09-14T00:00:00+00:00",
                                       "timeline": {"all_data": [{"source_id": "job_postings"}]}}))
    forecast_path = data / "forecast_log.jsonl"
    preview_path = data / "preview_log.jsonl"
    if live:
        (models / "forecaster_v1").mkdir()
        (models / "forecaster_v1" / "meta.json").write_text(json.dumps({
            "version": "v1", "feature_columns": ["lvl", "g4", "g13"], "horizon_periods": 12,
            "trained_as_of": "2026-09-14", "gate_profile": "demo", "gate": GATE_PASS,
            "disclaimer": TEST_DISCLAIMER}))
        AppendOnlyLog(forecast_path).append_many(_forecast_records())
    if preview:
        AppendOnlyLog(preview_path).append_many(_preview_records())

    monkeypatch.setattr(C, "DATA_DIR", data)
    monkeypatch.setattr(C, "PROCESSED_DIR", processed)
    monkeypatch.setattr(C, "OBSERVATIONS_PATH", obs_path)
    monkeypatch.setattr(C, "REGISTRY_PATH", reg_path)
    monkeypatch.setattr(C, "VERSION_LOG_PATH", vlog_path)
    monkeypatch.setattr(C, "MODELS_DIR", models)
    monkeypatch.setattr(C, "REPORT_PATH", report_path)
    monkeypatch.setattr(C, "FORECAST_LOG_PATH", forecast_path)
    monkeypatch.setattr(C, "PREVIEW_LOG_PATH", preview_path)
    monkeypatch.setattr(C, "PREVIEW_ENABLED", True)
    return {"tmp": tmp_path, "forecast_path": forecast_path, "preview_path": preview_path, "vlog": vlog,
            "obs_path": obs_path}


@pytest.fixture
def no_model_env(tmp_path, monkeypatch):
    """Today's real shape: gate rejected every candidate, no live model, no forecast log."""
    return build_env(tmp_path, monkeypatch, live=False)


@pytest.fixture
def live_model_env(tmp_path, monkeypatch):
    """A promoted model with four logged forecast records (one older run, three in the newest run)."""
    return build_env(tmp_path, monkeypatch, live=True)


@pytest.fixture
def preview_env(tmp_path, monkeypatch):
    """No live model, but the gate-rejected candidate's preview forecasts exist (validated: false)."""
    return build_env(tmp_path, monkeypatch, live=False, preview=True)
