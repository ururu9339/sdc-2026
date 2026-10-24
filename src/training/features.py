"""Feature table: one row per (entity, month), built only from data known at that month.

Each source is normalised within its own history (expanding z-score, growth rates) before
being combined. Leading sources get lagged columns so the model can learn the lead time.
Postings (the target series) and leading sources are measured independently per entity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sources.loaders import GLOBAL_ID
from src.training.config import EPS, HORIZON, MIN_HISTORY, SMOOTH_WINDOW, leading_sources, target_source

META = ["entity_id", "kind", "period", "y", "support"]
OWN_COLS = ["lvl", "g1", "g3", "g6", "g3_l3", "accel", "vol6", "z", "n_obs", "support_mean6"]
COMPOSITE_COLS = ["lead_composite", "trust_mass"]


def _zscore(s: pd.Series) -> pd.Series:
    z = (s - s.expanding(MIN_HISTORY).mean()) / s.expanding(MIN_HISTORY).std()
    return z.replace([np.inf, -np.inf], np.nan)


def _entity_frame(g: pd.DataFrame, sup: pd.Series, months, tgt: str, leads: list[dict], horizon: int) -> pd.DataFrame:
    g = g.reindex(months)
    post = g[tgt]
    lvl = np.log(post.rolling(SMOOTH_WINDOW, min_periods=2).mean() + EPS)
    f = pd.DataFrame(index=months)
    f["lvl"] = lvl
    for k in (1, 3, 6):
        f[f"g{k}"] = lvl - lvl.shift(k)
    f["g3_l3"] = f["g3"].shift(3)
    f["accel"] = f["g3"] - f["g3_l3"]
    f["vol6"] = lvl.diff().rolling(6, min_periods=4).std()
    f["z"] = _zscore(lvl)
    f["n_obs"] = post.notna().cumsum()
    f["support_mean6"] = sup.reindex(months).rolling(6, min_periods=1).mean()
    f["y"] = lvl.shift(-horizon) - lvl

    num = pd.Series(0.0, index=months)
    den = pd.Series(0.0, index=months)
    for s in leads:
        sid = s["source_id"]
        x = g[sid] if sid in g else pd.Series(np.nan, index=months)
        if s.get("ffill_limit"):
            x = x.ffill(limit=s["ffill_limit"])
        sm = np.log1p(x).rolling(SMOOTH_WINDOW, min_periods=1).mean()
        f[f"{sid}__z"] = _zscore(sm)
        f[f"{sid}__g3"] = sm - sm.shift(3)
        f[f"{sid}__g6"] = sm - sm.shift(6)
        f[f"{sid}__z_l3"] = f[f"{sid}__z"].shift(3)
        f[f"{sid}__z_l6"] = f[f"{sid}__z"].shift(6)
        yield_w = s["_w"]
        avail = f[f"{sid}__g3"].notna()
        num += (f[f"{sid}__g3"].fillna(0) * yield_w).where(avail, 0)
        den += yield_w * avail
    total_w = sum(s["_w"] for s in leads)
    f["lead_composite"] = (num / den).where(den > 0)
    f["trust_mass"] = den / total_w if total_w else 0.0
    f["month"] = months.month
    return f


def build_feature_table(obs: pd.DataFrame, registry: dict, as_of=None, horizon: int = HORIZON,
                        weights: dict | None = None, active: set | None = None) -> pd.DataFrame:
    obs = obs.copy()
    obs["period"] = pd.to_datetime(obs["period"])
    # A figure for period P that is published `release_lag_months` later is only usable from P+lag.
    lag = obs["source_id"].map({s["source_id"]: s.get("release_lag_months", 0) for s in registry["sources"]}).fillna(0)
    obs["period"] = [p + pd.DateOffset(months=int(m)) if m else p for p, m in zip(obs["period"], lag)]
    as_of = pd.Timestamp(as_of) if as_of is not None else obs["period"].max()
    obs = obs[obs["period"] <= as_of]
    tgt = target_source(registry)
    present = set(obs["source_id"])
    leads = []
    for s in leading_sources(registry):
        if s["source_id"] in present and (active is None or s["source_id"] in active):
            leads.append({**s, "_w": (weights or {}).get(s["source_id"], s["reliability"])})

    months = pd.date_range(obs["period"].min(), as_of, freq="MS")
    wide = obs.pivot_table(index=["entity_id", "period"], columns="source_id", values="value", aggfunc="last")
    sup = obs[obs["source_id"] == tgt].pivot_table(index=["entity_id", "period"], values="support", aggfunc="last")["support"]
    kinds = obs[obs["entity_id"] != GLOBAL_ID].drop_duplicates("entity_id").set_index("entity_id")["kind"]

    globals_ = wide.xs(GLOBAL_ID, level=0) if GLOBAL_ID in wide.index.get_level_values(0) else None
    frames = []
    for ent, g in wide.groupby(level=0):
        if ent == GLOBAL_ID:
            continue
        if tgt not in g or g[tgt].notna().sum() == 0:
            continue
        g = g.droplevel(0)
        if globals_ is not None:
            g = g.join(globals_[[c for c in globals_.columns if c not in g.columns]], how="outer")
        f = _entity_frame(g, sup.xs(ent, level=0) if ent in sup.index.get_level_values(0) else pd.Series(dtype=float),
                          months, tgt, leads, horizon)
        f.insert(0, "entity_id", ent)
        f.insert(1, "kind", kinds[ent])
        frames.append(f.rename_axis("period").reset_index())
    df = pd.concat(frames, ignore_index=True)
    df["kind_code"] = (df["kind"] == "job").astype(int)
    df["support"] = df["support_mean6"]
    return df


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with enough own history to be scored/trained on."""
    return df[(df["n_obs"] >= MIN_HISTORY) & df["lvl"].notna() & df["g3"].notna()]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META]


def column_source_map(cols: list[str], registry: dict) -> dict[str, str]:
    """Group feature columns by the source they come from (for contribution reporting)."""
    tgt = target_source(registry)
    lead_ids = [s["source_id"] for s in registry["sources"] if s["role"] == "leading"]
    out = {}
    for c in cols:
        if "__" in c and c.split("__")[0] in lead_ids:
            out[c] = c.split("__")[0]
        elif c in OWN_COLS:
            out[c] = tgt
        elif c in COMPOSITE_COLS:
            out[c] = "composite"
        else:
            out[c] = "calendar_meta"
    return out
