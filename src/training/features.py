from __future__ import annotations

import numpy as np
import pandas as pd

from src.sources.loaders import GLOBAL_ID
from src.training.config import (PERIOD_FREQ, EPS, GRANULARITY_SPAN_DAYS, GROWTH_LAGS, HORIZON, MIN_HISTORY, SMOOTH_WINDOW,
                                 SUPPORT_WINDOW, VOL_WINDOW, Z_LAGS, ffill_periods, leading_sources, period_range,
                                 span_end, target_source, to_period_start)

META = ["entity_id", "kind", "period", "y", "support"]
MID, LONG = GROWTH_LAGS[1], GROWTH_LAGS[-1]
OWN_COLS = (["lvl"] + [f"g{k}" for k in GROWTH_LAGS]
            + [f"g{MID}_l{MID}", "accel", f"vol{VOL_WINDOW}", "z", "lvl_dev", "n_obs", f"support_mean{SUPPORT_WINDOW}"])
COMPOSITE_COLS = ["lead_composite", "trust_mass"]


def _zscore(s: pd.Series) -> pd.Series:
    z = (s - s.expanding(MIN_HISTORY).mean()) / s.expanding(MIN_HISTORY).std()
    return z.replace([np.inf, -np.inf], np.nan)


def _entity_frame(g: pd.DataFrame, sup: pd.Series, periods: pd.DatetimeIndex, tgt: str, leads: list[dict],
                  horizon: int) -> pd.DataFrame:
    g = g.reindex(periods)
    post = g[tgt]
    lvl = np.log(post.rolling(SMOOTH_WINDOW, min_periods=2).mean() + EPS)
    f = pd.DataFrame(index=periods)
    f["lvl"] = lvl
    for k in GROWTH_LAGS:
        f[f"g{k}"] = lvl - lvl.shift(k)
    f[f"g{MID}_l{MID}"] = f[f"g{MID}"].shift(MID)
    f["accel"] = f[f"g{MID}"] - f[f"g{MID}_l{MID}"]
    f[f"vol{VOL_WINDOW}"] = lvl.diff().rolling(VOL_WINDOW, min_periods=2 * VOL_WINDOW // 3).std()
    f["z"] = _zscore(lvl)
    f["lvl_dev"] = lvl - lvl.expanding(MIN_HISTORY).mean()
    f["n_obs"] = post.notna().cumsum()
    f[f"support_mean{SUPPORT_WINDOW}"] = sup.reindex(periods).rolling(SUPPORT_WINDOW, min_periods=1).mean()
    f["y"] = lvl.shift(-horizon) - lvl

    num = pd.Series(0.0, index=periods)
    den = pd.Series(0.0, index=periods)
    for s in leads:
        sid = s["source_id"]
        x = g[sid] if sid in g else pd.Series(np.nan, index=periods)
        if s["ffill_periods"]:
            x = x.ffill(limit=s["ffill_periods"])
        sm = np.log1p(x).rolling(SMOOTH_WINDOW, min_periods=1).mean()
        f[f"{sid}__z"] = _zscore(sm)
        for k in GROWTH_LAGS:
            f[f"{sid}__g{k}"] = sm - sm.shift(k)
        for k in Z_LAGS:
            f[f"{sid}__z_l{k}"] = f[f"{sid}__z"].shift(k)
        avail = f[f"{sid}__g{MID}"].notna()
        num += (f[f"{sid}__g{MID}"].fillna(0) * s["weight"]).where(avail, 0)
        den += s["weight"] * avail
    total_w = sum(s["weight"] for s in leads)
    f["lead_composite"] = (num / den).where(den > 0)
    f["trust_mass"] = den / total_w if total_w else 0.0
    f["month"] = periods.month
    return f


def first_visible_period(starts: pd.Series, granularity: str, lag_days: int = 0) -> pd.Series:
    known_at = pd.Series(span_end(starts, granularity), index=starts.index) + pd.Timedelta(days=lag_days)
    grid = pd.PeriodIndex(pd.DatetimeIndex(known_at), freq=PERIOD_FREQ)
    exact = grid.start_time == pd.DatetimeIndex(known_at)
    first = np.where(exact, (grid - 1).start_time, grid.start_time)
    return pd.Series(pd.DatetimeIndex(first), index=starts.index)


def shift_to_publication(obs: pd.DataFrame, registry: dict) -> pd.DataFrame:
    obs = obs.copy()
    obs["source_period"] = pd.to_datetime(obs["period"])
    if obs.empty:
        return obs
    cfg = {s["source_id"]: s for s in registry["sources"]}
    parts = [first_visible_period(g["source_period"], cfg[sid]["granularity"], cfg[sid].get("release_lag_days", 0))
             for sid, g in obs.groupby("source_id")]
    obs["period"] = pd.concat(parts).reindex(obs.index)
    return obs


def build_feature_table(obs: pd.DataFrame, registry: dict, as_of=None, horizon: int = HORIZON,
                        weights: dict | None = None, active: set | None = None, dedup: bool = True) -> pd.DataFrame:
    raw_obs = obs
    obs = shift_to_publication(obs, registry)
    as_of = to_period_start(pd.Timestamp(as_of)) if as_of is not None else obs["period"].max()
    obs = obs[obs["period"] <= as_of]
    exclude: set[str] = set()
    if dedup:
        from src.training.audit import duplicate_groups, non_canonical_ids
        exclude = non_canonical_ids(duplicate_groups(raw_obs, registry, as_of))
    tgt = target_source(registry)
    present = set(obs["source_id"])
    leads = []
    for s in leading_sources(registry):
        if s["source_id"] in present and (active is None or s["source_id"] in active):
            span = GRANULARITY_SPAN_DAYS[s["granularity"]]
            leads.append({**s, "weight": (weights or {}).get(s["source_id"], s["reliability"]),
                          "ffill_periods": ffill_periods(s.get("ffill_limit_days", span))})

    periods = period_range(obs["period"].min(), as_of)
    wide = obs.pivot_table(index=["entity_id", "period"], columns="source_id", values="value", aggfunc="last")
    sup = obs[obs["source_id"] == tgt].pivot_table(index=["entity_id", "period"], values="support",
                                                   aggfunc="last")["support"]
    kinds = obs[obs["entity_id"] != GLOBAL_ID].drop_duplicates("entity_id").set_index("entity_id")["kind"]

    globals_ = wide.xs(GLOBAL_ID, level=0) if GLOBAL_ID in wide.index.get_level_values(0) else None
    frames = []
    for ent, g in wide.groupby(level=0):
        if ent == GLOBAL_ID or ent in exclude:
            continue
        if tgt not in g or g[tgt].notna().sum() == 0:
            continue
        g = g.droplevel(0)
        if globals_ is not None:
            g = g.join(globals_[[c for c in globals_.columns if c not in g.columns]], how="outer")
        f = _entity_frame(g, sup.xs(ent, level=0) if ent in sup.index.get_level_values(0) else pd.Series(dtype=float),
                          periods, tgt, leads, horizon)
        f.insert(0, "entity_id", ent)
        f.insert(1, "kind", kinds[ent])
        frames.append(f.rename_axis("period").reset_index())
    df = pd.concat(frames, ignore_index=True)
    df["kind_code"] = (df["kind"] == "job").astype(int)
    df["support"] = df[f"support_mean{SUPPORT_WINDOW}"]
    return df


def usable(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["n_obs"] >= MIN_HISTORY) & df["lvl"].notna() & df[f"g{MID}"].notna()]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META]


def column_source_map(cols: list[str], registry: dict) -> dict[str, str]:
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
