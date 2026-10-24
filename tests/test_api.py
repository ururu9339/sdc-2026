"""HTTP tests for the read-only forecast API. TestClient is used without the startup hook, so the SQLite
candidates database is never touched. Fixtures live in conftest.py."""
from fastapi.testclient import TestClient

from src.backend.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


# ---- no live model: the real shape of today ----------------------------------------------
def test_forecast_no_live_model_is_200_not_error(no_model_env):
    r = client.get("/forecast")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_live_model"
    assert body["reason"] == "failed: beats_baseline, fold_win_rate"
    assert body["gate"]["passed"] is False and body["gate"]["profile"] == "demo"
    assert body["forecasts"] == []
    assert body["disclaimer"].startswith("Test disclaimer")


def test_forecast_entity_no_live_model(no_model_env):
    body = client.get("/forecast/accountant").json()
    assert body["status"] == "no_live_model" and body["forecast"] is None
    assert "failed: beats_baseline" in body["reason"]


def test_forecast_unknown_entity_is_404(no_model_env):
    assert client.get("/forecast/not_a_real_entity").status_code == 404
    assert client.get("/history/not_a_real_entity").status_code == 404
    assert client.get("/explain/not_a_real_entity").status_code == 404


def test_explain_no_live_model_has_no_numbers(no_model_env):
    body = client.get("/explain/accountant").json()
    assert body["status"] == "no_live_model"
    assert "%" not in body["summary"]
    assert body["drivers"]["source_contributions"] is None
    # related skills come from the job-skill graph, not from a forecast
    assert [n["entity_id"] for n in body["drivers"]["related"]] == ["accounting", "excel", "python"]
    assert any("No live model" in c for c in body["caveats"])


def test_model_no_live(no_model_env):
    body = client.get("/model").json()
    assert body["live"] is None and body["live_version"] is None
    assert body["latest_attempt"]["reason"] == "failed: beats_baseline, fold_win_rate"


def test_history_endpoint_no_model_still_works(no_model_env):
    body = client.get("/history/accountant", params={"n_periods": 2}).json()
    assert body["entity_id"] == "accountant" and body["n_periods_returned"] == 2
    assert [p["period"] for p in body["series"]] == ["2026-08-03", "2026-08-10"]


def test_sources_endpoint(no_model_env):
    body = client.get("/sources").json()
    assert [s["source_id"] for s in body] == ["job_postings", "market_signal", "jobstreet_my"]
    assert body[2]["has_data"] is False


def test_history_rejects_bad_n_periods(no_model_env):
    assert client.get("/history/accountant", params={"n_periods": 0}).status_code == 422


# ---- live model: three hand-written forecast records (ok/stable noise_like, ok/emerging, insufficient) --
def test_forecast_list_live(live_model_env):
    body = client.get("/forecast").json()
    assert body["status"] == "ok" and body["model_version"] == "v1" and body["gate_profile"] == "demo"
    assert body["as_of"] == "2026-09-14" and body["target_period"] == "2026-12-07"
    assert body["horizon_periods"] == 12 and body["count"] == 3
    ids = {f["entity_id"] for f in body["forecasts"]}
    assert ids == {"accountant", "ai_engineer", "business_analyst"}
    assert all(f["hash"] and f["logged_at"] for f in body["forecasts"])


def test_forecast_list_filters(live_model_env):
    body = client.get("/forecast", params={"direction": "emerging"}).json()
    assert [f["entity_id"] for f in body["forecasts"]] == ["ai_engineer"]
    body = client.get("/forecast", params={"status": "insufficient_data"}).json()
    assert [f["entity_id"] for f in body["forecasts"]] == ["business_analyst"]
    body = client.get("/forecast", params={"kind": "job", "limit": 1}).json()
    assert body["count"] == 1
    assert client.get("/forecast", params={"direction": "up"}).status_code == 422


def test_forecast_entity_ok_and_insufficient(live_model_env):
    ok = client.get("/forecast/ai_engineer").json()
    assert ok["status"] == "ok" and ok["forecast"]["p50"] == 0.25 and ok["forecast"]["direction"] == "emerging"
    thin = client.get("/forecast/business_analyst").json()
    assert thin["status"] == "insufficient_data" and thin["forecast"]["status"] == "insufficient_data"
    assert thin["reason"] == "history_too_short (6 < 26 periods)"


def test_explain_caveats_present_live(live_model_env):
    body = client.get("/explain/accountant").json()
    assert body["status"] == "ok"
    assert body["caveats"][0].startswith("Test disclaimer")
    assert any("'demo'" in c for c in body["caveats"])
    assert any("indistinguishable from random variation" in c for c in body["caveats"])
    assert "indistinguishable from random variation" in body["summary"]
    assert "%" in body["summary"] and "12 weeks" in body["summary"]


def test_explain_emerging_has_no_noise_caveat(live_model_env):
    body = client.get("/explain/ai_engineer").json()
    assert body["status"] == "ok"
    assert not any("random variation" in c for c in body["caveats"])
    assert body["drivers"]["source_contributions"] == {"job_postings": 0.12, "market_signal": 0.05}


def test_explain_insufficient_live(live_model_env):
    body = client.get("/explain/business_analyst").json()
    assert body["status"] == "insufficient_data"
    assert "history_too_short" in body["summary"]


def test_graph_neighbour_order_in_explain(live_model_env):
    body = client.get("/explain/accountant").json()
    assert [n["entity_id"] for n in body["drivers"]["related"]] == ["accounting", "excel", "python"]
    assert body["drivers"]["related"][0]["weight"] == 1.0


def test_history_endpoint_live(live_model_env):
    body = client.get("/history/ai_engineer", params={"n_periods": 52}).json()
    assert body["n_periods_returned"] == 6 and body["series"][-1]["value"] == 0.013
    assert body["audit"]["entity_id"] == "ai_engineer"


def test_model_live(live_model_env):
    body = client.get("/model").json()
    assert body["live"]["version"] == "v1" and body["live"]["gate_profile"] == "demo"
    assert body["live"]["source_importance"]["job_postings"] == 0.02
    assert body["timeline"] == {"all_data": [{"source_id": "job_postings"}]}
    assert body["disclaimer"].startswith("Test disclaimer")
