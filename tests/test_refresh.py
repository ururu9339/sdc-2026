import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.pipeline import refresh
from src.pipeline.agent_loop import AgentLoop
from src.pipeline.store import AppendOnlyLog
from src.training import config as C
from tests.conftest import build_env


def _patch_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "OUTCOME_LOG_PATH", C.DATA_DIR / "forecast_outcomes.jsonl")
    monkeypatch.setattr(C, "PREVIEW_OUTCOME_LOG_PATH", C.DATA_DIR / "preview_outcomes.jsonl")
    monkeypatch.setattr(C, "RUN_LOG_PATH", C.DATA_DIR / "agent_runs.jsonl")
    monkeypatch.setattr(C, "STATE_PATH", C.DATA_DIR / "agent_state.json")
    monkeypatch.setattr(C, "RAW_DIR", tmp_path / "raw")
    (tmp_path / "raw").mkdir(exist_ok=True)


def test_lock_is_created_and_removed(tmp_path):
    path = tmp_path / "refresh.lock"
    with refresh.RefreshLock(path):
        assert json.loads(path.read_text())["pid"] > 0
    assert not path.exists()


def test_fresh_lock_held_by_a_live_process_refuses(tmp_path):
    path = tmp_path / "refresh.lock"
    path.write_text(json.dumps({"pid": __import__("os").getpid(), "started_at": datetime.now(timezone.utc).isoformat()}))
    with pytest.raises(refresh.LockHeld):
        with refresh.RefreshLock(path):
            pass
    assert path.exists()


def test_lock_of_a_dead_process_is_taken_over(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    path = tmp_path / "refresh.lock"
    path.write_text(json.dumps({"pid": proc.pid, "started_at": datetime.now(timezone.utc).isoformat()}))
    with refresh.RefreshLock(path):
        assert json.loads(path.read_text())["pid"] != proc.pid
    assert not path.exists()


def test_lock_older_than_the_limit_is_stale_even_if_the_pid_lives(tmp_path):
    import os
    path = tmp_path / "refresh.lock"
    old = datetime.now(timezone.utc) - timedelta(hours=refresh.STALE_HOURS + 1)
    path.write_text(json.dumps({"pid": os.getpid(), "started_at": old.isoformat()}))
    with refresh.RefreshLock(path):
        pass
    assert not path.exists()


def test_unreadable_lock_goes_stale_by_age(tmp_path):
    path = tmp_path / "refresh.lock"
    path.write_text("{not json")
    assert refresh._is_stale(path, datetime.now(timezone.utc)) is False
    old_time = (datetime.now(timezone.utc) - timedelta(hours=refresh.STALE_HOURS + 1)).timestamp()
    import os
    os.utime(path, (old_time, old_time))
    assert refresh._is_stale(path, datetime.now(timezone.utc)) is True


def test_decide_reuses_the_preview_fitted_for_the_same_as_of():
    loop = AgentLoop.__new__(AgentLoop)
    as_of = pd.Timestamp("2026-09-14")
    vlog = {"live": None, "preview": "preview_v1", "history": []}
    assert loop.decide(as_of, {"last_trained_as_of": "2026-09-14"}, vlog, {}) == \
        (False, "preview model already fitted for this as_of")
    assert loop.decide(pd.Timestamp("2026-09-21"), {"last_trained_as_of": "2026-09-14"}, vlog, {})[0] is True
    assert loop.decide(as_of, {"last_trained_as_of": "2026-09-14"}, {"live": None, "history": []}, {})[0] is True


def test_issuing_the_same_origin_twice_adds_no_forecast_records(tmp_path, monkeypatch):
    build_env(tmp_path, monkeypatch, live=False, preview=True)
    _patch_logs(monkeypatch, tmp_path)
    import src.pipeline.agent_loop as al
    as_of = pd.Timestamp("2026-09-14")
    meta = {"version": "preview:v1", "feature_columns": ["lvl"], "gate_profile": "demo", "preview_reason": "x",
            "horizon_periods": 12, "trained_as_of": "2026-09-14"}
    monkeypatch.setattr(al, "load_models", lambda path: (None, meta))

    def fake_forecasts(models, meta_, feats, as_of_, registry, audit=None, duplicates=None):
        return [{"forecast_id": f"{meta_['version']}:job:new_entity:{C.period_label(as_of_)}", "status": "ok",
                 "signal_quality": "unknown", "entity_id": "job:new_entity", "as_of": C.period_label(as_of_)}]
    monkeypatch.setattr(al, "make_forecasts", fake_forecasts)
    loop = AgentLoop()
    feats = pd.DataFrame({"lvl": [1.0]})
    before = len(AppendOnlyLog(C.PREVIEW_LOG_PATH).read())
    loop._issue({}, "preview", "preview_v1", as_of, feats, pd.DataFrame())
    after_first = len(AppendOnlyLog(C.PREVIEW_LOG_PATH).read())
    loop._issue({}, "preview", "preview_v1", as_of, feats, pd.DataFrame())
    assert (before, after_first) == (3, 4)
    assert len(AppendOnlyLog(C.PREVIEW_LOG_PATH).read()) == after_first


def test_reconciling_twice_adds_no_outcomes(tmp_path, monkeypatch):
    build_env(tmp_path, monkeypatch, live=False, preview=False)
    _patch_logs(monkeypatch, tmp_path)
    AppendOnlyLog(C.PREVIEW_LOG_PATH).append_many([{
        "forecast_id": "preview:v1:job:a:2026-01-05", "model_version": "preview:v1", "gate_profile": "demo",
        "entity_id": "job:a", "as_of": "2026-01-05", "target_period": "2026-02-02", "status": "ok",
        "p50": 0.1, "p10": 0.0, "p90": 0.2, "baseline_trend": 0.0}])
    feats = pd.DataFrame({"entity_id": ["job:a", "job:a"],
                          "period": pd.to_datetime(["2026-01-05", "2026-02-02"]), "lvl": [0.0, 0.5]})
    loop = AgentLoop()
    first = loop.reconcile(feats, pd.Timestamp("2026-02-02"), feed="preview")
    second = loop.reconcile(feats, pd.Timestamp("2026-02-02"), feed="preview")
    assert len(first) == 1 and second == []
    assert len(AppendOnlyLog(C.PREVIEW_OUTCOME_LOG_PATH).read()) == 1


class _FakeLoop:
    result: dict = {}
    calls: list = []

    def run_once(self, as_of=None):
        _FakeLoop.calls.append(as_of)
        return dict(_FakeLoop.result)


OK_RUN = {"status": "ok", "as_of": "2026-09-14", "feed": "preview", "decision": {"retrain": False,
          "why": "preview model already fitted for this as_of"}, "added_rows": {}, "outcomes_reconciled": 0,
          "preview_outcomes_reconciled": 0, "n_forecasts": 0}


def _refresh_env(tmp_path, monkeypatch, preview=True):
    build_env(tmp_path, monkeypatch, live=False, preview=preview)
    _patch_logs(monkeypatch, tmp_path)
    monkeypatch.setattr(refresh, "AgentLoop", _FakeLoop)
    _FakeLoop.calls = []
    _FakeLoop.result = dict(OK_RUN)
    return tmp_path / "bundle"


def test_refresh_writes_export_bundle_and_record(tmp_path, monkeypatch):
    bundle = _refresh_env(tmp_path, monkeypatch)
    now = datetime(2026, 10, 25, tzinfo=timezone.utc)
    rec = refresh.run_refresh(None, bundle, now=now)
    assert rec["status"] == "ok" and rec["feed"] == "preview" and rec["as_of"] == "2026-09-14"
    assert (C.DATA_DIR / "handoff" / "entity_forecast_preview.csv").exists()
    assert (bundle / "manifest.json").exists() and rec["bundle"]["manifest_sha256"]
    saved = json.loads((C.DATA_DIR / "last_refresh.json").read_text())
    assert saved["status"] == "ok" and saved["bundle"]["files"] == rec["bundle"]["files"]
    assert not (C.DATA_DIR / "refresh.lock").exists()


def test_refresh_twice_for_the_same_as_of_adds_no_records(tmp_path, monkeypatch):
    bundle = _refresh_env(tmp_path, monkeypatch)
    rec1 = refresh.run_refresh("2026-09-14", bundle)
    rec2 = refresh.run_refresh("2026-09-14", bundle)
    assert rec1["log_records"]["added"] == {"forecasts": 0, "outcomes": 0, "preview_forecasts": 0,
                                            "preview_outcomes": 0}
    assert rec2["log_records"]["added"] == rec1["log_records"]["added"]
    assert rec2["bundle"]["manifest_sha256"] is not None
    assert _FakeLoop.calls == ["2026-09-14", "2026-09-14"]


def test_failed_loop_writes_failed_record_releases_lock_and_exits_1(tmp_path, monkeypatch):
    bundle = _refresh_env(tmp_path, monkeypatch)
    _FakeLoop.result = {"status": "no_data", "reason": "no 'job_postings' data yet"}
    assert refresh.main(["--bundle", str(bundle)]) == 1
    saved = json.loads((C.DATA_DIR / "last_refresh.json").read_text())
    assert saved["status"] == "failed" and saved["error"]["type"] == "RuntimeError"
    assert not (C.DATA_DIR / "refresh.lock").exists()
    assert not bundle.exists()


def test_held_lock_exits_2_and_is_left_alone(tmp_path, monkeypatch):
    bundle = _refresh_env(tmp_path, monkeypatch)
    import os
    lock = C.DATA_DIR / "refresh.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}))
    assert refresh.main(["--bundle", str(bundle)]) == 2
    assert lock.exists() and _FakeLoop.calls == []


def test_refresh_refuses_to_run_from_the_serving_bundle(tmp_path, monkeypatch):
    _refresh_env(tmp_path, monkeypatch)
    monkeypatch.setattr(C, "DATA_LAYOUT", "bundle")
    assert refresh.main([]) == 2


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    bundle = _refresh_env(tmp_path, monkeypatch)
    assert refresh.main(["--dry-run", "--bundle", str(bundle)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["active_feed_now"] == "preview" and out["lock_present"] is False
    assert not (C.DATA_DIR / "last_refresh.json").exists() and not bundle.exists()
    assert _FakeLoop.calls == []


def test_no_model_is_a_valid_outcome_not_a_failure(tmp_path, monkeypatch):
    bundle = _refresh_env(tmp_path, monkeypatch, preview=False)
    _FakeLoop.result = {"status": "no_live_model", "feed": "none", "reason": "gate not passed yet"}
    rec = refresh.run_refresh(None, bundle)
    assert rec["status"] == "ok" and rec["export"]["rows"] == 0 and rec["feed"] == "none"
    assert refresh.main(["--bundle", str(bundle)]) == 0
