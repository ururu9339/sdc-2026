"""API: the preview feed (unvalidated, banner, validated: false) and the descriptive /snapshot.
Fixtures are hand-written and live in tmp_path (see conftest.py)."""
import pandas as pd
from fastapi.testclient import TestClient

from src.backend.main import app
from src.sources.loaders import COLUMNS
from src.training import config as C
from tests.conftest import PREVIEW_REASON, build_env

client = TestClient(app)
BANNER = ("PREVIEW: unvalidated candidate forecasts; the model did not pass the validation gate "
          f"({PREVIEW_REASON}). Not for decisions.")


# ---- preview feed ------------------------------------------------------------------------------
def test_forecast_list_is_preview_when_no_model_is_live(preview_env):
    body = client.get("/forecast").json()
    assert body["status"] == "preview" and body["feed"] == "preview" and body["validated"] is False
    assert body["banner"] == BANNER and body["reason"] == PREVIEW_REASON
    assert body["count"] == 3 and body["model_version"] == "preview:v1"
    assert all(f["validated"] is False and f["feed"] == "preview" for f in body["forecasts"])
    assert all(f["preview_reason"] == PREVIEW_REASON for f in body["forecasts"])


def test_preview_list_honours_filters(preview_env):
    body = client.get("/forecast", params={"direction": "emerging"}).json()
    assert [f["entity_id"] for f in body["forecasts"]] == ["job:ai_engineer"]
    assert client.get("/forecast", params={"status": "insufficient_data"}).json()["count"] == 1


def test_preview_entity_is_flagged_and_colon_ids_work(preview_env):
    body = client.get("/forecast/job:ai_engineer").json()
    assert body["status"] == "preview" and body["feed"] == "preview" and body["validated"] is False
    assert body["banner"] == BANNER
    assert body["forecast"]["p50"] == 0.25 and body["forecast"]["validated"] is False


def test_preview_entity_insufficient_is_not_dressed_up(preview_env):
    body = client.get("/forecast/job:business_analyst").json()
    assert body["status"] == "insufficient_data" and body["forecast"]["status"] == "insufficient_data"
    assert body["validated"] is False and body["banner"] == BANNER


def test_explain_starts_with_the_preview_sentence_and_keeps_caveats(preview_env):
    body = client.get("/explain/job:ai_engineer").json()
    assert body["status"] == "preview" and body["feed"] == "preview" and body["validated"] is False
    assert body["summary"].startswith(BANNER)
    assert body["caveats"][0].startswith("Test disclaimer")
    assert any(c.startswith("PREVIEW:") for c in body["caveats"])
    assert any("'demo'" in c for c in body["caveats"])
    assert "%" in body["summary"] and "12 weeks" in body["summary"]


def test_explain_noise_caveat_survives_in_preview(preview_env):
    body = client.get("/explain/job:accountant").json()
    assert body["summary"].startswith(BANNER)
    assert "indistinguishable from random variation" in body["summary"]


def test_explain_name_comes_from_the_taxonomy(preview_env):
    body = client.get("/explain/skill:python").json()
    assert body["summary"].startswith(BANNER)
    assert "Python" in body["summary"]  # taxonomy name, not the raw id
    assert body["status"] == "insufficient_data"  # no preview record for this skill in the fixture


def test_no_model_no_preview_is_feed_none(no_model_env):
    body = client.get("/forecast").json()
    assert body["status"] == "no_live_model" and body["feed"] == "none" and body["validated"] is False
    ent = client.get("/forecast/job:accountant").json()
    assert ent["feed"] == "none" and ent["validated"] is False and ent["status"] == "no_live_model"


def test_preview_disabled_hides_preview_records(preview_env, monkeypatch):
    monkeypatch.setattr(C, "PREVIEW_ENABLED", False)
    body = client.get("/forecast").json()
    assert body["status"] == "no_live_model" and body["feed"] == "none"


def test_live_model_wins_over_stale_preview_records(tmp_path, monkeypatch):
    build_env(tmp_path, monkeypatch, live=True, preview=True)
    body = client.get("/forecast").json()
    assert body["status"] == "ok" and body["feed"] == "live" and body["validated"] is True
    assert body["banner"] is None and body["count"] == 3
    assert all(f["feed"] == "live" and f["validated"] is True for f in body["forecasts"])
    ent = client.get("/forecast/job:ai_engineer").json()
    assert ent["status"] == "ok" and ent["feed"] == "live" and ent["banner"] is None
    explain = client.get("/explain/job:ai_engineer").json()
    assert explain["status"] == "ok" and explain["feed"] == "live" and explain["banner"] is None
    assert not explain["summary"].startswith("PREVIEW")


# ---- snapshot (descriptive, not a forecast) -----------------------------------------------------
def _write_raw(tmp_path, monkeypatch, rows):
    """Long-format raw files, one per source, read by the loaders (the snapshot reads sources, not the store)."""
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    monkeypatch.setattr(C, "RAW_DIR", raw)
    monkeypatch.setattr(C, "SNAPSHOT_MIN_BLOCK_PERIODS", 2)  # fixtures use 2-3 period blocks
    df = pd.DataFrame(rows, columns=COLUMNS)
    for sid, part in df.groupby("source_id"):
        part.to_csv(raw / f"{sid}.csv", index=False)


def _listing_rows():
    """jobstreet_my (tier 1): three weeks, a two-week gap, then two weeks. Values are shares."""
    first = ["2024-01-01", "2024-01-08", "2024-01-15"]
    last = ["2024-03-04", "2024-03-11"]
    rows = []
    for ent, kind, vals in [("skill:python", "skill", [0.10, 0.20, 0.30]), ("job:accountant", "job", [0.02, 0.02, 0.02])]:
        for d, v in zip(first, vals):
            rows.append([ent, kind, d, "jobstreet_my", v, 10])
        for d, v in zip(last, [0.5 if kind == "skill" else 0.01, 0.6 if kind == "skill" else 0.01]):
            rows.append([ent, kind, d, "jobstreet_my", v, 20])
    rows.append(["job:accountant", "job", "2024-01-01", "job_postings", 0.5, 100])  # target, not a leading source
    return rows


def test_snapshot_uses_the_best_tier_leading_source_with_data(preview_env, tmp_path, monkeypatch):
    _write_raw(tmp_path, monkeypatch, _listing_rows() + [["skill:python", "skill", "2023-01-02", "market_signal", 0.01, 5]])
    body = client.get("/snapshot").json()
    assert body["status"] == "ok" and body["type"] == "descriptive_not_forecast"
    assert body["banner"].startswith("DESCRIPTIVE SNAPSHOT, NOT A FORECAST")
    assert [s["source_id"] for s in body["sources"]] == ["jobstreet_my"]  # tier 1 beats tier 3 market_signal
    assert {i["type"] for i in body["items"]} == {"descriptive_not_forecast"}
    py = next(i for i in body["items"] if i["entity_id"] == "skill:python")
    assert py["name"] == "Python" and py["kind"] == "skill"
    assert abs(py["mean_share_first"] - 0.20) < 1e-9 and abs(py["mean_share_last"] - 0.55) < 1e-9
    assert abs(py["percent_change"] - 175.0) < 1e-6
    assert py["support"] == 10 * 3 + 20 * 2
    assert py["first_block"] == "2024-01-01..2024-01-15" and py["last_block"] == "2024-03-04..2024-03-11"


def test_snapshot_kind_filter_and_limit(preview_env, tmp_path, monkeypatch):
    _write_raw(tmp_path, monkeypatch, _listing_rows())
    skills = client.get("/snapshot", params={"kind": "skill"}).json()
    assert [i["entity_id"] for i in skills["items"]] == ["skill:python"]
    assert client.get("/snapshot", params={"limit": 1}).json()["count"] == 2
    assert client.get("/snapshot", params={"limit": 0}).status_code == 422
    assert client.get("/snapshot", params={"kind": "other"}).status_code == 422


def test_snapshot_ignores_blocks_shorter_than_the_minimum(preview_env, tmp_path, monkeypatch):
    _write_raw(tmp_path, monkeypatch, _listing_rows())
    monkeypatch.setattr(C, "SNAPSHOT_MIN_BLOCK_PERIODS", 4)  # both fixture blocks are shorter than this
    body = client.get("/snapshot").json()
    assert body["status"] == "no_data" and body["items"] == []
    assert "at least 4 periods" in body["reason"]


def test_snapshot_single_block_is_explained_not_faked(preview_env, tmp_path, monkeypatch):
    rows = [r for r in _listing_rows() if r[3] == "jobstreet_my" and r[2] < "2024-02"]  # only the first block
    _write_raw(tmp_path, monkeypatch, rows)
    body = client.get("/snapshot").json()
    assert body["status"] == "no_data" and body["items"] == []
    assert "no first-versus-last comparison" in body["reason"]


def test_snapshot_without_any_leading_data_is_an_empty_explanation(tmp_path, monkeypatch):
    build_env(tmp_path, monkeypatch, live=False)
    _write_raw(tmp_path, monkeypatch, [r for r in _listing_rows() if r[3] == "job_postings"])
    body = client.get("/snapshot").json()
    assert body["status"] == "no_data" and body["items"] == [] and body["sources"] == []
    assert body["reason"] == "no enabled leading source has data"


def test_snapshot_never_forecasts(preview_env, tmp_path, monkeypatch):
    _write_raw(tmp_path, monkeypatch, _listing_rows())
    body = client.get("/snapshot").json()
    assert all("p50" not in i and "direction" not in i for i in body["items"])
