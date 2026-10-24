"""Serving bundle: contents, manifest hashes, size limit, chain validity, failure without damage, and the API
serving a bundle with the same answers as the source state. Fixtures only (conftest.build_env)."""
import hashlib
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.backend.main import app
from src.pipeline import export_bundle
from src.pipeline.store import AppendOnlyLog
from src.training import config as C
from tests.conftest import build_env
from tests.test_refresh import _patch_logs

client = TestClient(app)
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
ALLOWED = {"manifest.json", "preview_log.jsonl", "forecast_log.jsonl", "processed/job_skill_graph.csv",
           "processed/observations.csv", "model_report.json", "models/version_log.json"}


@pytest.fixture
def preview_source(tmp_path, monkeypatch):
    build_env(tmp_path, monkeypatch, live=False, preview=True)
    _patch_logs(monkeypatch, tmp_path)
    (C.DATA_DIR / "processed" / "job_skill_graph.csv").write_text("job,skill,n_cooccur,n_job_postings,weight\n")
    return tmp_path


def _bundle_files(out):
    return {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}


def test_manifest_hashes_and_sizes_match_the_files(preview_source):
    out = preview_source / "serving"
    m = export_bundle.build_bundle(out, now=NOW)
    assert m["feed"] == "preview" and m["validated"] is False and m["as_of"] == "2026-09-14"
    assert m["model_version"] == "preview:v1"
    assert m["note"].startswith("snapshot for serving")
    for entry in m["files"]:
        data = (out / entry["path"]).read_bytes()
        assert entry["bytes"] == len(data)
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    listed = {e["path"] for e in m["files"]}
    assert listed == _bundle_files(out) - {"manifest.json"}
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["total_bytes"] == sum(p.stat().st_size for p in out.rglob("*") if p.is_file())


def test_bundle_holds_only_what_the_api_reads(preview_source):
    out = preview_source / "serving"
    export_bundle.build_bundle(out, now=NOW)
    assert _bundle_files(out) <= ALLOWED
    assert not any("src/data" in p or "candidate" in p.lower() for p in _bundle_files(out))


def test_only_the_newest_as_of_is_kept_and_the_chain_verifies(preview_source):
    out = preview_source / "serving"
    m = export_bundle.build_bundle(out, now=NOW)
    records = AppendOnlyLog(out / "preview_log.jsonl").read()
    assert len(records) == 3 and {r["as_of"] for r in records} == {"2026-09-14"}
    assert AppendOnlyLog(out / "preview_log.jsonl").verify() is True
    assert m["row_counts"]["forecasts"] == 3
    # the original logged_at timestamps are kept, not rewritten at export time
    original = {r["forecast_id"]: r["logged_at"] for r in AppendOnlyLog(C.PREVIEW_LOG_PATH).read()}
    assert all(original[r["forecast_id"]] == r["logged_at"] for r in records)


def test_tampering_breaks_the_chain(preview_source):
    out = preview_source / "serving"
    export_bundle.build_bundle(out, now=NOW)
    path = out / "preview_log.jsonl"
    path.write_text(path.read_text().replace('"p50": 0.25', '"p50": 0.99', 1))
    assert AppendOnlyLog(path).verify() is False


def test_version_log_is_trimmed_and_keeps_the_preview_entry(preview_source):
    out = preview_source / "serving"
    m = export_bundle.build_bundle(out, now=NOW)
    vlog = json.loads((out / "models" / "version_log.json").read_text())
    assert any(h.get("preview_version") for h in vlog["history"])
    assert vlog["history_kept"] == len(vlog["history"]) <= export_bundle.KEEP_VERSIONS + 1
    assert m["gate"]["passed"] is False


def test_over_the_size_limit_fails_and_writes_nothing(preview_source):
    out = preview_source / "serving"
    with pytest.raises(export_bundle.BundleError, match="over the"):
        export_bundle.build_bundle(out, now=NOW, max_bytes=200)
    assert not out.exists()
    assert not any(p.name.startswith(".serving-") for p in preview_source.iterdir())


def test_failed_rebuild_keeps_the_previous_bundle(preview_source):
    out = preview_source / "serving"
    first = export_bundle.build_bundle(out, now=NOW)
    with pytest.raises(export_bundle.BundleError):
        export_bundle.build_bundle(out, now=datetime(2026, 11, 1, tzinfo=timezone.utc), max_bytes=200)
    assert json.loads((out / "manifest.json").read_text())["generated_at"] == first["generated_at"]


def test_no_feed_writes_no_forecast_file(tmp_path, monkeypatch):
    build_env(tmp_path, monkeypatch, live=False, preview=False)
    _patch_logs(monkeypatch, tmp_path)
    out = tmp_path / "serving"
    m = export_bundle.build_bundle(out, now=NOW)
    assert m["feed"] == "none" and m["validated"] is False
    assert not (out / "preview_log.jsonl").exists() and not (out / "forecast_log.jsonl").exists()


def test_live_feed_is_marked_validated_and_ships_the_meta(tmp_path, monkeypatch):
    build_env(tmp_path, monkeypatch, live=True, preview=False)
    _patch_logs(monkeypatch, tmp_path)
    out = tmp_path / "serving"
    m = export_bundle.build_bundle(out, now=NOW)
    assert m["feed"] == "live" and m["validated"] is True and m["model_version"] == "v1"
    assert (out / "forecast_log.jsonl").exists() and (out / "models" / "forecaster_v1" / "meta.json").exists()
    assert AppendOnlyLog(out / "forecast_log.jsonl").verify() is True


def test_api_serves_the_bundle_with_the_same_answers(preview_source, monkeypatch):
    before = {p: client.get(p).json() for p in ("/api/v1/forecast", "/api/v1/forecast/job:ai_engineer",
                                                "/api/v1/explain/job:ai_engineer", "/api/v1/model")}
    out = preview_source / "serving"
    export_bundle.build_bundle(out, now=NOW)
    # re-point every runtime path at the bundle, as SDC_DATA_DIR / SDC_MODELS_DIR would
    monkeypatch.setattr(C, "DATA_DIR", out)
    monkeypatch.setattr(C, "PROCESSED_DIR", out / "processed")
    monkeypatch.setattr(C, "RAW_DIR", out / "raw")
    monkeypatch.setattr(C, "MODELS_DIR", out / "models")
    monkeypatch.setattr(C, "VERSION_LOG_PATH", out / "models" / "version_log.json")
    monkeypatch.setattr(C, "OBSERVATIONS_PATH", out / "processed" / "observations.csv")
    monkeypatch.setattr(C, "PREVIEW_LOG_PATH", out / "preview_log.jsonl")
    monkeypatch.setattr(C, "FORECAST_LOG_PATH", out / "forecast_log.jsonl")
    monkeypatch.setattr(C, "REPORT_PATH", out / "model_report.json")
    after = {p: client.get(p).json() for p in before}
    for p in before:
        assert after[p] == before[p], p
    monkeypatch.setattr(C, "SERVING_DIR", out)
    health = client.get("/api/v1/health").json()["data"]
    assert health["bundle"]["feed"] == "preview" and health["bundle"]["validated"] is False
