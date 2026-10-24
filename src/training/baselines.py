"""Naive forecasters the model must beat. Predictions are in the model's target units
(log change of smoothed postings share over HORIZON periods)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.config import GROWTH_LAGS, HORIZON

MID, LONG = GROWTH_LAGS[1], GROWTH_LAGS[-1]


def predict_baselines(df: pd.DataFrame, horizon: int = HORIZON) -> dict[str, np.ndarray]:
    slope = (df[f"g{LONG}"] / LONG).fillna(df[f"g{MID}"] / MID).fillna(0.0)
    trend = (horizon * slope).to_numpy()
    # Mean reversion: the further above its own long-run mean, the more the series is expected to fall back.
    # NaN (too little history for the expanding mean) is treated as no deviation.
    dev = df["lvl_dev"].fillna(0.0).to_numpy()
    return {
        "persistence": np.zeros(len(df)),
        "trend": trend,
        "damped_trend": 0.5 * trend,
        "mean_reversion": -dev,
        "half_reversion": -0.5 * dev,
    }
