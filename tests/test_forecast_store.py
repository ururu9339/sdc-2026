"""Service-level tests for the read-only forecast store, on hand-written fixtures (see conftest.py)."""
import math


from src.backend.services import forecast_store as store


def test_no_live_model_state(no_model_env):
    assert store.live_version() is None
    assert store.latest_forecasts() == []
    assert store.forecast_for("job:accountant") is None
    assert store.select_forecasts() == []
    assert store.no_live_reason() == "failed: beats_baseline, fold_win_rate"
    info = store.model_info()
    assert info["live"] is None and info["live_version"] is None and info["attempts"] == 1
    assert info["latest_attempt"]["passed"] is False
    assert info["timeline"] is None or "all_data" in info["timeline"]


def test_no_live_model_explanation_does_not_invent_numbers(no_model_env):
    out = store.explain("job:accountant")
    assert out["status"] == "no_live_model"
    assert "no model has passed the gate" in out["summary"]
    assert "failed: beats_baseline, fold_win_rate" in out["summary"]
    assert "%" not in out["summary"]
    assert out["drivers"]["source_contributions"] is None
    assert out["caveats"][0] == "Test disclaimer: fixture data, for demonstration only."


def test_live_state_latest_run_only(live_model_env):
    recs = store.latest_forecasts()
    assert {r["entity_id"] for r in recs} == {"job:accountant", "job:ai_engineer", "job:business_analyst"}
    assert all(r["as_of"] == "2026-09-14" for r in recs)
    acc = store.forecast_for("job:accountant")
    assert acc["as_of"] == "2026-09-14" and acc["p50"] == 0.01  # not the older 2026-09-07 run
    assert store.forecast_for("unknown_entity") is None


def test_live_state_filters(live_model_env):
    assert [r["entity_id"] for r in store.select_forecasts(direction="emerging")] == ["job:ai_engineer"]
    assert [r["entity_id"] for r in store.select_forecasts(status="insufficient_data")] == ["job:business_analyst"]
    assert store.select_forecasts(kind="skill") == []
    assert len(store.select_forecasts(limit=1)) == 1
    assert store.forecast_envelope()["model_version"] == "v1"


def test_explain_percent_conversion_and_caveats(live_model_env):
    out = store.explain("job:ai_engineer")
    assert out["status"] == "ok"
    lo, hi = math.exp(0.10) - 1, math.exp(0.45) - 1
    assert f"{100 * lo:+.1f}% to {100 * hi:+.1f}%" in out["summary"]
    assert "12 weeks" in out["summary"] and "to 2026-12-07" in out["summary"]
    assert "trend_detected" not in out["summary"]  # signal quality is spoken, not dumped as a code
    assert out["caveats"][0] == "Test disclaimer: fixture data, for demonstration only."
    assert any("'demo'" in c for c in out["caveats"])
    assert not any("random variation" in c for c in out["caveats"])


def test_explain_noise_like_caveat(live_model_env):
    out = store.explain("job:accountant")
    assert "indistinguishable from random variation" in out["summary"]
    assert any("indistinguishable from random variation" in c for c in out["caveats"])
    assert any("'demo'" in c for c in out["caveats"])
    assert out["drivers"]["source_contributions"] == {"job_postings": 0.006, "market_signal": 0.004}


def test_explain_insufficient_data_gives_reason(live_model_env):
    out = store.explain("job:business_analyst")
    assert out["status"] == "insufficient_data"
    assert "history_too_short" in out["summary"]
    assert out["drivers"]["source_contributions"] is None


def test_graph_neighbours_order(live_model_env):
    job = store.graph_neighbours("job:accountant", top_k=2)
    assert [n["entity_id"] for n in job] == ["skill:accounting", "skill:excel"]
    assert all(n["kind"] == "skill" for n in job)
    skill = store.graph_neighbours("skill:python")
    assert [n["entity_id"] for n in skill] == ["job:ai_engineer", "job:accountant"]
    assert store.graph_neighbours("no_such_entity") == []


def test_history_series_and_audit(live_model_env):
    h = store.history("job:accountant", n_periods=3)
    assert h["source_id"] == "job_postings" and h["granularity"] == "weekly"
    assert len(h["series"]) == 3
    assert h["series"][-1]["period"] == "2026-08-10"
    assert h["series"][-1]["value"] == 0.012 and h["series"][-1]["support"] == 118
    assert h["audit"]["entity_id"] == "job:accountant" and h["audit"]["n_periods"] == 6
    assert store.history("unknown_entity") is None


def test_sources_info_reports_coverage(live_model_env):
    by = {s["source_id"]: s for s in store.sources_info()}
    assert by["job_postings"]["has_data"] is True and by["job_postings"]["periods"] == 6
    assert by["market_signal"]["has_data"] is True and by["market_signal"]["periods"] == 3
    assert by["jobstreet_my"]["has_data"] is False and by["jobstreet_my"]["periods"] == 0
    assert by["job_postings"]["license"] == "fixture licence"


def test_model_info_live(live_model_env):
    info = store.model_info()
    assert info["live_version"] == "v1"
    assert info["live"]["gate_profile"] == "demo"
    assert info["live"]["source_importance"] == {"job_postings": 0.02, "market_signal": 0.005}
    assert info["latest_attempt"]["passed"] is True
    assert info["report_generated_at"] == "2026-09-14T00:00:00+00:00"
    assert info["audit_summary"][0]["source_id"] == "job_postings"
