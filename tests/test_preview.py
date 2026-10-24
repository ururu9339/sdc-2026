import json

import numpy as np
import pandas as pd
import pytest

from src.pipeline import agent_loop as AL
from src.pipeline.store import AppendOnlyLog
from src.training import config as C

REASON = "failed: beats_baseline (test seam)"


@pytest.fixture
def toy(tmp_path, monkeypatch):
    data, raw, models = tmp_path / "data", tmp_path / "data" / "raw", tmp_path / "models"
    raw.mkdir(parents=True)
    models.mkdir()
    weeks = pd.date_range("2023-01-02", periods=60, freq="W-MON")
    rng = np.random.RandomState(0)
    rows = []
    for k, ent in enumerate(["skill:alpha", "skill:beta", "job:gamma"]):
        for i, p in enumerate(weeks):
            rows.append({"entity_id": ent, "kind": ent.split(":")[0], "period": p.strftime("%Y-%m-%d"),
                         "source_id": "toy", "value": 0.05 + 0.002 * i + 0.003 * k + rng.normal(0, 0.002),
                         "support": 300 + 10 * i})
    pd.DataFrame(rows).to_csv(raw / "toy.csv", index=False)
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"version": 5, "disclaimer": "toy disclaimer", "sources": [
        {"source_id": "toy", "role": "target", "enabled": True, "reliability": 0.9, "granularity": "weekly",
         "release_lag_days": 0, "min_period_postings": 1, "min_entity_total": 1, "dense": True,
         "loader": {"type": "csv"}, "scope": "entity"}]}))
    paths = {"DATA_DIR": data, "PROCESSED_DIR": data / "processed", "RAW_DIR": raw, "MODELS_DIR": models,
             "REGISTRY_PATH": reg, "OBSERVATIONS_PATH": data / "processed" / "observations.csv",
             "FORECAST_LOG_PATH": data / "forecast_log.jsonl", "OUTCOME_LOG_PATH": data / "forecast_outcomes.jsonl",
             "PREVIEW_LOG_PATH": data / "preview_log.jsonl",
             "PREVIEW_OUTCOME_LOG_PATH": data / "preview_outcomes.jsonl", "RUN_LOG_PATH": data / "agent_runs.jsonl",
             "STATE_PATH": data / "agent_state.json", "VERSION_LOG_PATH": models / "version_log.json",
             "PREVIEW_ENABLED": True}
    for name, value in paths.items():
        monkeypatch.setattr(C, name, value)
    monkeypatch.setattr(AL, "walk_forward", lambda feats, cols, profile=None: (pd.DataFrame(), None))
    state = {"passed": False}

    def fake_gate(folds, profile=None):
        return {"passed": state["passed"], "reason": "ok" if state["passed"] else REASON,
                "profile": "demo", "checks": {}, "constraints": {}}

    monkeypatch.setattr(AL, "evaluate_gate", fake_gate)
    toy.state = state
    toy.weeks = weeks
    toy.paths = paths
    return toy


def _replay(start_idx, end_idx):
    loop = AL.AgentLoop()
    return loop.replay(str(toy_start(start_idx)), str(toy_start(end_idx)), 1)


_WEEKS = pd.date_range("2023-01-02", periods=60, freq="W-MON")


def toy_start(i):
    return _WEEKS[i].date()


def test_rejected_gate_issues_preview_forecasts_only(toy):
    runs = _replay(30, 34)
    preview = AppendOnlyLog(C.PREVIEW_LOG_PATH).read()
    assert preview, "expected preview forecasts"
    assert all(r["validated"] is False and r["feed"] == "preview" for r in preview)
    assert all(r["preview_reason"] == REASON and r["gate_profile"] == "demo" for r in preview)

    assert all(r["model_version"].startswith("preview:v") for r in preview)
    assert all(r["forecast_id"].startswith(r["model_version"] + ":") for r in preview)
    assert any(r["status"] == "ok" for r in preview)
    assert AppendOnlyLog(C.PREVIEW_LOG_PATH).verify()

    assert AppendOnlyLog(C.FORECAST_LOG_PATH).read() == []
    vlog = json.loads(C.VERSION_LOG_PATH.read_text())
    assert vlog["live"] is None
    statuses = [h["status"] for h in vlog["history"]]
    assert statuses and set(statuses) == {"rejected_preview"}
    assert all(h["version"] is None for h in vlog["history"])
    dirs = [h["model_dir"] for h in vlog["history"]]
    assert dirs == [f"preview_v{i + 1}" for i in range(len(dirs))]
    assert vlog["preview"] == dirs[-1]
    meta = json.loads((C.MODELS_DIR / vlog["preview"] / "meta.json").read_text())
    assert meta["version"] == f"preview:v{len(dirs)}" and meta["validated"] is False
    assert meta["preview_reason"] == REASON
    assert not any((C.MODELS_DIR / f"forecaster_v{i}").exists() for i in range(1, 5))

    feeds = [r["feed"] for r in AppendOnlyLog(C.RUN_LOG_PATH).read()]
    assert "preview" in feeds and "live" not in feeds
    assert all(r["status"] == "ok" for r in runs if r["feed"] == "preview")


def test_preview_outcomes_go_to_their_own_log(toy):
    _replay(30, 50)
    outcomes = AppendOnlyLog(C.PREVIEW_OUTCOME_LOG_PATH).read()
    assert outcomes and all(o["feed"] == "preview" and o["validated"] is False for o in outcomes)
    assert AppendOnlyLog(C.PREVIEW_OUTCOME_LOG_PATH).verify()
    assert not C.OUTCOME_LOG_PATH.exists()


def test_preview_disabled_issues_nothing(toy, monkeypatch):
    monkeypatch.setattr(C, "PREVIEW_ENABLED", False)
    runs = _replay(30, 33)
    assert not C.PREVIEW_LOG_PATH.exists()
    assert all(r["status"] == "no_live_model" and r["feed"] == "none" for r in runs)
    assert json.loads(C.VERSION_LOG_PATH.read_text()).get("preview") is None


def test_passing_gate_is_live_and_never_writes_preview(toy):
    toy.state["passed"] = True
    runs = _replay(30, 32)
    live = AppendOnlyLog(C.FORECAST_LOG_PATH).read()
    assert live and all("validated" not in r and "feed" not in r for r in live)
    assert all(r["feed"] == "live" for r in runs if r.get("status") == "ok")
    assert not C.PREVIEW_LOG_PATH.exists()
    vlog = json.loads(C.VERSION_LOG_PATH.read_text())
    assert vlog["live"] == "v1" and vlog.get("preview") is None
    assert not any(h.get("preview_version") for h in vlog["history"])


def test_preview_forecasts_stop_when_a_live_model_appears(toy):
    _replay(30, 31)
    before = len(AppendOnlyLog(C.PREVIEW_LOG_PATH).read())
    toy.state["passed"] = True
    _replay(32, 33)
    assert len(AppendOnlyLog(C.PREVIEW_LOG_PATH).read()) == before
    assert AppendOnlyLog(C.FORECAST_LOG_PATH).read()
