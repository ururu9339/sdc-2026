"""Central settings for training, the gate and the agent loop."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "src" / "models"
REGISTRY_PATH = ROOT / "src" / "sources" / "sources_registry.json"

OBSERVATIONS_PATH = PROCESSED_DIR / "observations.csv"
STATE_PATH = DATA_DIR / "agent_state.json"
FORECAST_LOG_PATH = DATA_DIR / "forecast_log.jsonl"
OUTCOME_LOG_PATH = DATA_DIR / "forecast_outcomes.jsonl"
RUN_LOG_PATH = DATA_DIR / "agent_runs.jsonl"
VERSION_LOG_PATH = MODELS_DIR / "version_log.json"

# Forecast target: change in log (3-month-smoothed) share of postings, HORIZON months ahead.
HORIZON = 3
SMOOTH_WINDOW = 3
EPS = 1e-4
MIN_HISTORY = 6  # postings months needed before a row is usable for training

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

# Gate: fixed and deliberately simple. The model never grades itself.
GATE = {
    "n_folds": 6,
    "fold_step": 3,  # months of forecast origins per test fold
    "min_train_months": 9,
    "min_folds": 3,
    "mae_margin": 0.03,  # model MAE must be >= 3% below the best baseline
    "min_fold_win_rate": 0.6,
    "coverage_range": (0.65, 0.95),  # nominal 10-90% interval = 0.80
}

# Abstention rules applied at forecast time.
ABSTAIN = {
    "min_obs_months": 12,
    "min_mean_support": 15.0,  # mean raw mention count over the last 6 months
    "max_interval_width": 1.2,  # log-growth units
}
DIRECTION_THRESHOLD = 0.10  # |median log growth| over the horizon to call emerging/declining

# Agent loop
RETRAIN_EVERY_PERIODS = 1
DRIFT = {"min_matured": 40, "tolerance": 1.0}  # retrain early if live MAE > tolerance * baseline MAE


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def target_source(registry: dict) -> str:
    return next(s["source_id"] for s in registry["sources"] if s["role"] == "target")


def leading_sources(registry: dict) -> list[dict]:
    return [s for s in registry["sources"] if s["role"] == "leading" and s.get("enabled", True)]


def enabled_sources(registry: dict) -> list[dict]:
    return [s for s in registry["sources"] if s.get("enabled", True)]
