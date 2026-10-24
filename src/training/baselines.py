"""Naive forecasters the model must beat. Predictions are in the model's target units
(log change of smoothed postings share over HORIZON months)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.config import HORIZON


def predict_baselines(df: pd.DataFrame, horizon: int = HORIZON) -> dict[str, np.ndarray]:
    slope = (df["g6"] / 6).fillna(df["g3"] / 3).fillna(0.0)
    trend = (horizon * slope).to_numpy()
    return {
        "persistence": np.zeros(len(df)),
        "trend": trend,
        "damped_trend": 0.5 * trend,
    }
