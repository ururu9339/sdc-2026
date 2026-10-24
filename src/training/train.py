"""One global LightGBM model: a point model plus 10% / 90% quantile models for the interval."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.training.config import LGB_POINT_PARAMS, LGB_ROUNDS, QUANTILES


def fit_models(train: pd.DataFrame, cols: list[str]) -> dict[str, lgb.Booster]:
    X, y = train[cols], train["y"]
    models = {"point": lgb.train(LGB_POINT_PARAMS, lgb.Dataset(X, y), LGB_ROUNDS)}
    for q in QUANTILES:
        params = {**LGB_POINT_PARAMS, "objective": "quantile", "alpha": q}
        models[f"q{int(q * 100)}"] = lgb.train(params, lgb.Dataset(X, y), LGB_ROUNDS)
    return models


def predict(models: dict[str, lgb.Booster], X: pd.DataFrame) -> pd.DataFrame:
    p = models["point"].predict(X)
    lo, hi = models["q10"].predict(X), models["q90"].predict(X)
    return pd.DataFrame({"p50": p, "p10": np.minimum(lo, p), "p90": np.maximum(hi, p)}, index=X.index)


def source_contributions(models: dict[str, lgb.Booster], X: pd.DataFrame, col_source: dict[str, str]) -> pd.DataFrame:
    """SHAP contribution of each source's columns to the point forecast (log-growth units)."""
    contrib = models["point"].predict(X, pred_contrib=True)[:, :-1]
    df = pd.DataFrame(contrib, index=X.index, columns=list(X.columns))
    return df.T.groupby(col_source).sum().T


def save_models(models: dict[str, lgb.Booster], directory: Path, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, m in models.items():
        m.save_model(str(directory / f"{name}.txt"))
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load_models(directory: Path) -> tuple[dict[str, lgb.Booster], dict]:
    models = {p.stem: lgb.Booster(model_file=str(p)) for p in directory.glob("*.txt")}
    return models, json.loads((directory / "meta.json").read_text())
