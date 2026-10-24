from datetime import date

from fastapi.testclient import TestClient

from src.backend.main import app
from src.backend.services import forecast_store as store
from src.backend.services import report as service
from src.pipeline.store import AppendOnlyLog
from src.training import config as C
from tests.conftest import PREVIEW_REASON, _record

client = TestClient(app)


def _today(monkeypatch, d: str):
    monkeypatch.setattr(C, "utc_today", lambda: date.fromisoformat(d))


def test_age_and_window_are_computed_from_today_not_stored(monkeypatch):
    _today(monkeypatch, "2026-10-25")
    s = store.staleness("2023-09-11", "2023-12-04")
    assert s["data_age_days"] == (date(2026, 10, 25) - date(2023, 9, 11)).days
    assert s["stale"] is True
    assert s["forecast_window"] == "made as of 2023-09-11 for 2023-12-04"


def test_stale_threshold_is_strictly_more_than_stale_days(monkeypatch):
    assert C.STALE_DAYS == 60
    _today(monkeypatch, "2026-09-14")
    assert store.staleness("2026-09-14")["stale"] is False
    _today(monkeypatch, "2026-11-12")
    assert store.staleness("2026-09-14")["data_age_days"] == 59 and store.staleness("2026-09-14")["stale"] is False
    _today(monkeypatch, "2026-11-13")
    assert store.staleness("2026-09-14")["stale"] is False
    _today(monkeypatch, "2026-11-14")
    assert store.staleness("2026-09-14")["stale"] is True


def test_no_forecast_date_gives_no_age(monkeypatch):
    assert store.staleness(None) == {"data_age_days": None, "stale": None, "forecast_window": None}


def test_preview_banner_carries_both_dates_and_the_age_when_stale(monkeypatch):
    _today(monkeypatch, "2026-10-25")
    fresh = store.preview_banner("r", "2026-10-20", "2026-12-07")
    assert fresh == ("PREVIEW: unvalidated candidate forecasts; the model did not pass the validation gate (r). "
                     "Forecast made as of 2026-10-20 for 2026-12-07. Not for decisions.")
    stale = store.preview_banner("r", "2023-09-11", "2023-12-04")
    assert "Forecast made as of 2023-09-11 for 2023-12-04." in stale
    assert f"Data is {(date(2026, 10, 25) - date(2023, 9, 11)).days} days old." in stale
    assert stale.endswith("Not for decisions.")


def test_live_banner_only_when_stale(monkeypatch):
    _today(monkeypatch, "2026-09-20")
    assert store.live_banner("2026-09-14", "2026-12-07") is None
    _today(monkeypatch, "2026-12-01")
    text = store.live_banner("2026-09-14", "2026-12-07")
    assert text.startswith("STALE: validated forecast made as of 2026-09-14 for 2026-12-07.")
    assert "Data is 78 days old" in text


def test_forecast_list_and_records_carry_the_fields_and_the_banner(preview_env, monkeypatch):
    _today(monkeypatch, "2026-12-01")
    body = client.get("/forecast").json()
    assert body["data_age_days"] == 78 and body["stale"] is True
    assert body["forecast_window"] == "made as of 2026-09-14 for 2026-12-07"
    assert "Data is 78 days old." in body["banner"]
    rec = body["forecasts"][0]
    assert rec["data_age_days"] == 78 and rec["stale"] is True and rec["forecast_window"] == body["forecast_window"]


def test_entity_and_explain_say_how_old_the_forecast_is(preview_env, monkeypatch):
    _today(monkeypatch, "2026-12-01")
    ent = client.get("/forecast/job:ai_engineer").json()
    assert ent["stale"] is True and ent["data_age_days"] == 78 and ent["forecast_window"].startswith("made as of")
    assert "Data is 78 days old." in ent["banner"]
    ex = client.get("/explain/job:ai_engineer").json()
    assert ex["stale"] is True and ex["data_age_days"] == 78
    assert "Data is 78 days old." in ex["summary"]
    assert "Forecast made as of 2026-09-14" in ex["summary"] or "made as of 2026-09-14" in ex["summary"]


def test_fresh_forecast_has_no_stale_sentence(preview_env):
    ex = client.get("/explain/job:ai_engineer").json()
    assert ex["stale"] is False and "days old" not in ex["summary"]


def test_health_reports_forecast_age_and_stale_flag(preview_env, monkeypatch):
    _today(monkeypatch, "2026-09-20")
    d = client.get("/health").json()["data"]
    assert d["forecast_as_of"] == "2026-09-14" and d["forecast_age_days"] == 6 and d["stale"] is False
    _today(monkeypatch, "2026-12-01")
    d = client.get("/health").json()["data"]
    assert d["forecast_age_days"] == 78 and d["stale"] is True


def test_health_has_null_age_with_no_forecast(no_model_env):
    d = client.get("/health").json()["data"]
    assert d["forecast_as_of"] is None and d["forecast_age_days"] is None and d["stale"] is None


def test_report_carries_dates_age_and_why_mentions_the_as_of(preview_env, monkeypatch):
    _today(monkeypatch, "2026-12-01")

    AppendOnlyLog(preview_env["preview_path"]).append_many([_record(
        model_version="preview:v1", feed="preview", validated=False, preview_reason=PREVIEW_REASON, gate_profile="demo",
        forecast_id="preview:v1:skill:python:2026-09-14", as_of="2026-09-14", entity_id="skill:python", kind="skill",
        status="ok", reason=None, p50=0.1, p10=0.05, p90=0.15, direction="emerging", confidence="low",
        signal_quality="noise_like", dispersion_ratio=1.0, source_contributions={"job_postings": 0.0})])
    r = service.build_report(["skill:accounting", "skill:excel"])
    assert r["forecast_as_of"] == "2026-09-14" and r["forecast_target_period"] == "2026-12-07"
    assert r["data_age_days"] == 78 and r["stale"] is True
    assert r["forecast_window"] == "made as of 2026-09-14 for 2026-12-07"
    assert "Data is 78 days old." in r["banner"]
    assert any("Outlook figures are dated" in c for c in r["caveats"])
    gap = r["skill_gaps"][0]
    assert gap["skill_id"] == "skill:python" and gap["as_of"] == "2026-09-14"
    assert "outlook: emerging (+11%, unvalidated preview, made as of 2026-09-14)" in gap["why"]


def test_report_has_no_dates_when_nothing_is_served(no_model_env):
    r = service.build_report(["skill:accounting"])
    assert r["forecast_as_of"] is None and r["data_age_days"] is None and r["stale"] is None
    assert r["forecast_window"] is None


def test_serving_does_not_change_the_stored_logs(preview_env, monkeypatch):
    before = preview_env["preview_path"].read_bytes() if preview_env["preview_path"].exists() else b""
    _today(monkeypatch, "2026-12-01")
    client.get("/forecast")
    client.get("/forecast/job:ai_engineer")
    client.get("/explain/job:ai_engineer")
    client.get("/health")
    assert preview_env["preview_path"].read_bytes() == before
    assert "data_age_days" not in before.decode()


def test_stale_flag_follows_the_clock_without_a_new_record(preview_env, monkeypatch):
    _today(monkeypatch, "2026-09-20")
    assert client.get("/forecast").json()["stale"] is False
    _today(monkeypatch, "2026-11-20")
    assert client.get("/forecast").json()["stale"] is True
