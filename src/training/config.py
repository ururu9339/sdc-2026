from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SERVING_DIR = ROOT / "serving"
BUNDLE_MANIFEST = "manifest.json"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def resolve_layout(environ: Mapping[str, str] | None = None, root: Path | None = None) -> dict:
    env = os.environ if environ is None else environ
    root = ROOT if root is None else Path(root)
    repo_data, serving = root / "data", root / "serving"
    env_data = env.get("SDC_DATA_DIR")
    env_models = env.get("SDC_MODELS_DIR")
    has_repo_logs = (repo_data / "forecast_log.jsonl").exists() or (repo_data / "preview_log.jsonl").exists()
    if env_data:
        data, mode = Path(env_data), "env"
    elif not has_repo_logs and (serving / BUNDLE_MANIFEST).exists():
        data, mode = serving, "bundle"
    else:
        data, mode = repo_data, "repo"
    if env_models:
        models = Path(env_models)
    elif mode == "bundle":
        models = serving / "models"
    else:
        models = root / "src" / "models"
    return {"data_dir": data, "models_dir": models, "mode": mode, "serving_dir": serving}


_LAYOUT = resolve_layout()
DATA_DIR = _LAYOUT["data_dir"]
DATA_LAYOUT = _LAYOUT["mode"]
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = _LAYOUT["models_dir"]
REGISTRY_PATH = _env_path("SDC_REGISTRY", ROOT / "src" / "sources" / "sources_registry.json")
TAXONOMY_DIR = _env_path("SDC_TAXONOMY_DIR", ROOT / "embeddings" / "taxonomy")

OBSERVATIONS_PATH = PROCESSED_DIR / "observations.csv"
STATE_PATH = DATA_DIR / "agent_state.json"
FORECAST_LOG_PATH = DATA_DIR / "forecast_log.jsonl"
OUTCOME_LOG_PATH = DATA_DIR / "forecast_outcomes.jsonl"
PREVIEW_LOG_PATH = DATA_DIR / "preview_log.jsonl"
PREVIEW_OUTCOME_LOG_PATH = DATA_DIR / "preview_outcomes.jsonl"
RUN_LOG_PATH = DATA_DIR / "agent_runs.jsonl"
VERSION_LOG_PATH = MODELS_DIR / "version_log.json"
REPORT_PATH = DATA_DIR / "model_report.json"


PERIOD_FREQ = os.environ.get("PERIOD_FREQ", "W")
GRANULARITY_FREQ = {"weekly": "W", "monthly": "M", "quarterly": "Q", "annual": "Y"}
GRANULARITY_SPAN_DAYS = {"weekly": 7, "monthly": 31, "quarterly": 92, "annual": 366}
PERIODS_PER_YEAR = {"weekly": 52, "monthly": 12, "quarterly": 4, "annual": 1}

_GATE_NOTE = {"production": "strict constraints for promotion to live use",
              "demo": "relaxed constraints; sources are weak proxies"}


def _gate_profiles(**by_name: dict) -> dict:
    return {name: {"name": name, **params, "note": _GATE_NOTE[name]} for name, params in by_name.items()}


PERIOD_PROFILES = {
    "W": {
        "horizon": 12,
        "smooth_window": 4,
        "growth_lags": (4, 13, 26),
        "z_lags": (13, 26),
        "vol_window": 13,
        "min_history": 13,
        "support_window": 13,
        "retrain_every": 4,
        "abstain_min_obs": 26,
        "audit_min_periods": 13,
        "gate_profiles": _gate_profiles(
            production={"n_folds": 6, "fold_step": 13, "min_train_periods": 52, "min_folds": 3,
                        "mae_margin": 0.03, "min_fold_win_rate": 0.6, "coverage_range": (0.65, 0.95)},
            demo={"n_folds": 6, "fold_step": 13, "min_train_periods": 26, "min_folds": 3,
                  "mae_margin": 0.0, "min_fold_win_rate": 0.5, "coverage_range": (0.55, 0.98)},
        ),
    },
    "M": {
        "horizon": 3,
        "smooth_window": 3,
        "growth_lags": (1, 3, 6),
        "z_lags": (3, 6),
        "vol_window": 6,
        "min_history": 6,
        "support_window": 6,
        "retrain_every": 1,
        "abstain_min_obs": 12,
        "audit_min_periods": 6,
        "gate_profiles": _gate_profiles(
            production={"n_folds": 6, "fold_step": 3, "min_train_periods": 12, "min_folds": 3,
                        "mae_margin": 0.03, "min_fold_win_rate": 0.6, "coverage_range": (0.65, 0.95)},
            demo={"n_folds": 6, "fold_step": 3, "min_train_periods": 9, "min_folds": 3,
                  "mae_margin": 0.0, "min_fold_win_rate": 0.5, "coverage_range": (0.55, 0.98)},
        ),
    },
}
if PERIOD_FREQ not in PERIOD_PROFILES:
    raise ValueError(f"PERIOD_FREQ must be one of {sorted(PERIOD_PROFILES)}, got {PERIOD_FREQ!r}")
_PROFILE = PERIOD_PROFILES[PERIOD_FREQ]


def _freq(freq: str | None) -> str:
    return PERIOD_FREQ if freq is None else freq


def to_period_start(ts, freq: str | None = None):
    freq = _freq(freq)
    if isinstance(ts, pd.Series):
        return pd.to_datetime(ts).dt.to_period(freq).dt.start_time
    if isinstance(ts, pd.Index):
        return pd.DatetimeIndex(pd.to_datetime(ts)).to_period(freq).start_time
    return pd.Timestamp(ts).to_period(freq).start_time


def shift_periods(ts, n: int, freq: str | None = None) -> pd.Timestamp:
    freq = _freq(freq)
    return (pd.Period(to_period_start(ts, freq), freq=freq) + n).start_time


def periods_between(a, b, freq: str | None = None) -> int:
    freq = _freq(freq)
    return pd.Period(pd.Timestamp(b), freq=freq).ordinal - pd.Period(pd.Timestamp(a), freq=freq).ordinal


def period_range(start, end, freq: str | None = None) -> pd.DatetimeIndex:
    freq = _freq(freq)
    return pd.period_range(to_period_start(start, freq), to_period_start(end, freq), freq=freq).start_time


def span_end(start, granularity: str = "weekly"):
    freq = GRANULARITY_FREQ[granularity]
    if isinstance(start, (pd.Series, pd.Index)):
        return (pd.PeriodIndex(pd.DatetimeIndex(pd.to_datetime(start)), freq=freq) + 1).start_time
    return (pd.Period(pd.Timestamp(start), freq=freq) + 1).start_time


PERIOD_DAYS = {"W": 7.0, "M": 30.44}


def ffill_periods(days: float) -> int:
    if PERIOD_FREQ == "W":
        return math.ceil(days / 7)
    return max(1, round(days / 30.44))


def period_label(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def parse_period_label(label, freq: str | None = None) -> pd.Timestamp:
    return to_period_start(pd.Timestamp(label), freq)


PREVIEW_ENABLED = os.environ.get("PREVIEW_FORECASTS", "true").strip().lower() not in ("0", "false", "no", "off")


HORIZON = _PROFILE["horizon"]
SMOOTH_WINDOW = _PROFILE["smooth_window"]
EPS = 1e-4
MIN_HISTORY = _PROFILE["min_history"]
GROWTH_LAGS = _PROFILE["growth_lags"]
Z_LAGS = _PROFILE["z_lags"]
VOL_WINDOW = _PROFILE["vol_window"]
SUPPORT_WINDOW = _PROFILE["support_window"]

LGB_POINT_PARAMS = {
    "objective": "huber",
    "alpha": 0.5,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "seed": 42,
    "verbose": -1,
    "num_threads": 2,
}
LGB_ROUNDS = 250
QUANTILES = (0.1, 0.9)


GATE_KEYS = ("n_folds", "fold_step", "min_train_periods", "min_folds", "mae_margin", "min_fold_win_rate",
             "coverage_range")
GATE_PROFILES = _PROFILE["gate_profiles"]
ACTIVE_GATE_PROFILE = os.environ.get("GATE_PROFILE", "demo")
if ACTIVE_GATE_PROFILE not in GATE_PROFILES:
    raise ValueError(f"GATE_PROFILE must be one of {sorted(GATE_PROFILES)}, got {ACTIVE_GATE_PROFILE!r}")


def active_gate_profile() -> dict:
    return GATE_PROFILES[ACTIVE_GATE_PROFILE]


ABSTAIN = {
    "min_obs_periods": _PROFILE["abstain_min_obs"],
    "min_mean_support": 15.0,
    "max_interval_width": 1.2,
    "abstain_on_noise": False,
}


AUDIT = {"noise_ratio": 1.5, "min_periods": _PROFILE["audit_min_periods"]}
DIRECTION_THRESHOLD = 0.10


RETRAIN_EVERY_PERIODS = _PROFILE["retrain_every"]
DRIFT = {"min_matured": 40, "tolerance": 1.0}


DEDUP = {"rel_tol": 1e-6, "min_periods": 13}
DUPLICATE_CAVEAT = "identical series to {canon} in our data; this is one signal, not several"


STALE_DAYS = 60


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def load_registry(path: Path | None = None) -> dict:
    with open(REGISTRY_PATH if path is None else path) as f:
        return json.load(f)


def target_source(registry: dict) -> str:
    return next(s["source_id"] for s in registry["sources"] if s["role"] == "target")


def leading_sources(registry: dict) -> list[dict]:
    return [s for s in registry["sources"] if s["role"] == "leading" and s.get("enabled", True)]


def enabled_sources(registry: dict) -> list[dict]:
    return [s for s in registry["sources"] if s.get("enabled", True)]


SNAPSHOT_MIN_BLOCK_PERIODS = 4
