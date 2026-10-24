"""Monthly model grid (PERIOD_FREQ=M) and isolated data paths (SDC_* overrides).

Configuration is read at import time, so each check runs in a fresh interpreter with the environment set
explicitly. The toy data here is hand-written and lives in tmp_path; the real repo data/ is never touched."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ISOLATION_VARS = ("SDC_DATA_DIR", "SDC_MODELS_DIR", "SDC_REGISTRY", "PERIOD_FREQ", "GATE_PROFILE")


def _run(code: str, **env_extra) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ISOLATION_VARS}
    env.update({k: str(v) for k, v in env_extra.items()})
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.strip().splitlines()[-1]


MONTHLY_CHECK = """
import json
from src.training import config as C
from src.training.features import first_visible_period
import pandas as pd
lab = C.period_label
out = {
    "freq": C.PERIOD_FREQ, "horizon": C.HORIZON, "growth": list(C.GROWTH_LAGS), "z": list(C.Z_LAGS),
    "vol": C.VOL_WINDOW, "min_history": C.MIN_HISTORY, "support": C.SUPPORT_WINDOW,
    "retrain": C.RETRAIN_EVERY_PERIODS, "abstain_min_obs": C.ABSTAIN["min_obs_periods"],
    "audit_min": C.AUDIT["min_periods"], "smooth": C.SMOOTH_WINDOW,
    "active": C.active_gate_profile(), "production": C.GATE_PROFILES["production"],
    "demo": C.GATE_PROFILES["demo"],
    "to_start": lab(C.to_period_start("2024-02-17")),
    "shift_back3": lab(C.shift_periods("2024-02-17", -3)),
    "range": [str(x.date()) for x in C.period_range("2024-01-15", "2024-03-02")],
    "between": C.periods_between("2024-01-01", "2024-04-30"),
    "span": lab(C.span_end("2024-01-01", "monthly")),
    "label": C.period_label("2024-02-01"),
    "ffill_31": C.ffill_periods(31),
    "visible_exact_lag0": [str(x.date()) for x in first_visible_period(
        pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"])), "monthly", 0)],
    "visible_lag7": [str(x.date()) for x in first_visible_period(
        pd.Series(pd.to_datetime(["2024-01-01"])), "monthly", 7)],
}
print(json.dumps(out, default=str))
"""


def test_monthly_grid_profiles_and_helpers():
    out = json.loads(_run(MONTHLY_CHECK, PERIOD_FREQ="M", GATE_PROFILE="production"))
    assert out["freq"] == "M" and out["active"]["name"] == "production"
    assert out["horizon"] == 3 and out["growth"] == [1, 3, 6] and out["z"] == [3, 6]
    assert out["vol"] == 6 and out["min_history"] == 6 and out["support"] == 6 and out["smooth"] == 3
    assert out["retrain"] == 1 and out["abstain_min_obs"] == 12 and out["audit_min"] == 6
    assert out["production"] == {"name": "production", "n_folds": 6, "fold_step": 3, "min_train_periods": 12,
                                 "min_folds": 3, "mae_margin": 0.03, "min_fold_win_rate": 0.6,
                                 "coverage_range": [0.65, 0.95], "note": out["production"]["note"]}
    assert out["demo"]["min_train_periods"] == 9 and out["demo"]["mae_margin"] == 0.0
    assert out["demo"]["min_fold_win_rate"] == 0.5 and out["demo"]["coverage_range"] == [0.55, 0.98]
    assert out["demo"]["fold_step"] == 3 and out["demo"]["n_folds"] == 6
    # period helpers work on month starts
    assert out["to_start"] == "2024-02-01"
    assert out["shift_back3"] == "2023-11-01"
    assert out["range"] == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert out["between"] == 3 and out["span"] == "2024-02-01" and out["label"] == "2024-02-01"
    assert out["ffill_31"] == 1
    # a monthly figure is usable at the first monthly origin whose end is at or after its own month end
    assert out["visible_exact_lag0"] == ["2024-01-01", "2024-02-01"]
    assert out["visible_lag7"] == ["2024-02-01"]


def test_weekly_default_is_unchanged():
    out = json.loads(_run(MONTHLY_CHECK))  # no PERIOD_FREQ, no GATE_PROFILE: the original weekly configuration
    assert out["freq"] == "W" and out["active"]["name"] == "demo"
    assert out["horizon"] == 12 and out["growth"] == [4, 13, 26] and out["z"] == [13, 26]
    assert out["vol"] == 13 and out["min_history"] == 13 and out["support"] == 13 and out["smooth"] == 4
    assert out["retrain"] == 4 and out["abstain_min_obs"] == 26 and out["audit_min"] == 13
    assert out["production"]["min_train_periods"] == 52 and out["production"]["fold_step"] == 13
    assert out["production"]["mae_margin"] == 0.03 and out["demo"]["min_train_periods"] == 26
    assert out["to_start"] == "2024-02-12" and out["shift_back3"] == "2024-01-22"
    assert out["ffill_31"] == 5  # ceil(31 / 7) weeks, as before
    assert out["visible_exact_lag0"] == ["2024-01-29", "2024-02-26"]  # Feb 1 is a Thursday: week of Jan 29


PATHS_CHECK = """
import json
from src.training import config as C
print(json.dumps({
    "data": str(C.DATA_DIR), "raw": str(C.RAW_DIR), "processed": str(C.PROCESSED_DIR),
    "models": str(C.MODELS_DIR), "registry": str(C.REGISTRY_PATH),
    "observations": str(C.OBSERVATIONS_PATH), "state": str(C.STATE_PATH), "forecast_log": str(C.FORECAST_LOG_PATH),
    "outcome_log": str(C.OUTCOME_LOG_PATH), "run_log": str(C.RUN_LOG_PATH), "version_log": str(C.VERSION_LOG_PATH),
    "report": str(C.REPORT_PATH),
    "registry_sources": [s["source_id"] for s in C.load_registry()["sources"]],
    "registry_default_arg": [s["source_id"] for s in C.load_registry(None)["sources"]],
}))
"""


def test_env_overrides_redirect_every_derived_path(tmp_path):
    data, models = tmp_path / "isolated_data", tmp_path / "isolated_models"
    reg = tmp_path / "isolated_registry.json"
    reg.write_text(json.dumps({"version": 5, "disclaimer": "t", "sources": [
        {"source_id": "only_here", "role": "target", "granularity": "monthly", "reliability": 0.9}]}))
    out = json.loads(_run(PATHS_CHECK, SDC_DATA_DIR=data, SDC_MODELS_DIR=models, SDC_REGISTRY=reg, PERIOD_FREQ="M"))
    d, m = str(data), str(models)
    assert out["data"] == d and out["raw"] == f"{d}/raw" and out["processed"] == f"{d}/processed"
    assert out["observations"] == f"{d}/processed/observations.csv" and out["state"] == f"{d}/agent_state.json"
    assert out["forecast_log"] == f"{d}/forecast_log.jsonl" and out["outcome_log"] == f"{d}/forecast_outcomes.jsonl"
    assert out["run_log"] == f"{d}/agent_runs.jsonl" and out["report"] == f"{d}/model_report.json"
    assert out["models"] == m and out["version_log"] == f"{m}/version_log.json"
    assert out["registry"] == str(reg)
    assert out["registry_sources"] == ["only_here"] and out["registry_default_arg"] == ["only_here"]


def test_default_paths_point_at_the_repo_data_without_overrides():
    out = json.loads(_run(PATHS_CHECK))
    assert out["data"] == str(ROOT / "data") and out["models"] == str(ROOT / "src" / "models")
    assert out["registry"] == str(ROOT / "src" / "sources" / "sources_registry.json")


def test_monthly_pipeline_runs_end_to_end_in_isolation(tmp_path):
    """A toy monthly target through the whole agent loop, with every file written under tmp_path."""
    data, models = tmp_path / "data", tmp_path / "models"
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"version": 5, "disclaimer": "toy", "sources": [
        {"source_id": "toy", "role": "target", "enabled": True, "reliability": 0.9, "granularity": "monthly",
         "release_lag_days": 0, "min_period_postings": 1, "min_entity_total": 1, "dense": True,
         "loader": {"type": "csv"}, "scope": "entity"}]}))
    (data / "raw").mkdir(parents=True)
    months = pd.period_range("2021-01", periods=30, freq="M").to_timestamp()
    rows = []
    for k, ent in enumerate(["alpha", "beta", "gamma"]):
        for i, p in enumerate(months):
            rows.append({"entity_id": ent, "kind": "skill", "period": p.strftime("%Y-%m-%d"), "source_id": "toy",
                         "value": 0.05 + 0.002 * i + 0.01 * k * (i % 3) / 3, "support": 300 + 10 * i})
    pd.DataFrame(rows).to_csv(data / "raw" / "toy.csv", index=False)

    code = """
import json
from src.pipeline.agent_loop import AgentLoop
from src.training import config as C
loop = AgentLoop()
runs = loop.replay("2023-03-01", "2023-06-01", 1)
print(json.dumps({"as_of": [r.get("as_of") for r in runs], "status": [r.get("status") for r in runs],
                  "freq": C.PERIOD_FREQ, "logs_intact": loop.status()["logs_intact"]}))
"""
    out = json.loads(_run(code, SDC_DATA_DIR=data, SDC_MODELS_DIR=models, SDC_REGISTRY=reg, PERIOD_FREQ="M",
                          GATE_PROFILE="demo"))
    assert out["freq"] == "M" and out["as_of"] == ["2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"]
    assert out["logs_intact"] is True
    assert (data / "processed" / "observations.csv").exists()
    assert (data / "agent_runs.jsonl").exists()
    assert (models / "version_log.json").exists()
