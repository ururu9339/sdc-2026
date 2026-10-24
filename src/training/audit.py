"""Signal audit and source coverage.

Each series is a share of postings per period, so if the period-to-period movement were pure sampling
noise, its variance would be about p(1-p)/n_t (binomial). The dispersion ratio compares the observed
variance to that expectation: about 1 means "indistinguishable from noise", well above 1 means the
series moves more than sampling alone explains. Only data known at as_of is used."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sources.loaders import GLOBAL_ID
from src.training.config import AUDIT, GRANULARITY_FREQ, period_label, target_source
from src.training.features import shift_to_publication

AUDIT_COLS = ["source_id", "entity_id", "kind", "n_periods", "first_period", "last_period", "mean_share",
              "dispersion_ratio", "abs_corr_with_time", "noise_like"]


def _audit_row(g: pd.DataFrame) -> dict:
    v = g["value"].astype(float).to_numpy()
    sup = g["support"].astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        n_t = np.round(sup / v)  # postings behind each period's share = count / share
    ok = (v > 0) & np.isfinite(n_t) & (n_t > 0)
    dr = np.nan
    if ok.any() and len(v) >= 2:
        p_bar = sup[ok].sum() / n_t[ok].sum()  # pooled share
        exp_var = float(np.mean(p_bar * (1 - p_bar) / n_t[ok]))
        if exp_var > 0:
            dr = float(np.var(v, ddof=1)) / exp_var
    t = g["period"].astype("int64").to_numpy(dtype=float)  # time index (only its ordering matters)
    corr = np.nan
    if len(v) >= 3 and np.std(v) > 0 and np.std(t) > 0:
        corr = abs(float(np.corrcoef(t, v)[0, 1]))
    n_periods = int(g["period"].nunique())
    noise = bool(dr < AUDIT["noise_ratio"] and n_periods >= AUDIT["min_periods"]) if not np.isnan(dr) else False
    return {"source_id": g["source_id"].iloc[0], "entity_id": g["entity_id"].iloc[0], "kind": g["kind"].iloc[0],
            "n_periods": n_periods, "first_period": period_label(g["source_period"].min()),
            "last_period": period_label(g["source_period"].max()), "mean_share": float(v.mean()),
            "dispersion_ratio": dr, "abs_corr_with_time": corr, "noise_like": noise}


def signal_audit(obs: pd.DataFrame, registry: dict, as_of=None) -> pd.DataFrame:
    """One row per (source_id, entity_id), using only figures known at as_of."""
    obs = shift_to_publication(obs, registry)
    if as_of is not None:
        obs = obs[obs["period"] <= pd.Timestamp(as_of)]
    obs = obs[(obs["entity_id"] != GLOBAL_ID) & obs["value"].notna()].sort_values("period")
    rows = [_audit_row(g) for _, g in obs.groupby(["source_id", "entity_id"], sort=True)]
    return pd.DataFrame(rows, columns=AUDIT_COLS)


def summarise_audit(audit: pd.DataFrame, registry: dict) -> pd.DataFrame:
    """Per source: entities, median dispersion ratio, share of noise-like entities, span in its own periods."""
    gran = {s["source_id"]: s["granularity"] for s in registry["sources"]}
    rows = []
    for sid, g in audit.groupby("source_id", sort=True):
        first, last = g["first_period"].min(), g["last_period"].max()
        span = len(pd.period_range(first, last, freq=GRANULARITY_FREQ[gran[sid]]))
        rows.append({"source_id": sid, "entities": int(len(g)),
                     "median_dispersion_ratio": float(g["dispersion_ratio"].median()),
                     "share_noise_like": float(g["noise_like"].mean()),
                     "first_period": first, "last_period": last, "periods_span": span})
    return pd.DataFrame(rows, columns=["source_id", "entities", "median_dispersion_ratio", "share_noise_like",
                                       "first_period", "last_period", "periods_span"])


def source_coverage(obs: pd.DataFrame, registry: dict, as_of=None) -> pd.DataFrame:
    """Periods each source has data (native periods) and how many of its visible periods overlap the
    target's. With as_of=None the whole table is used; otherwise only figures known at as_of."""
    obs = shift_to_publication(obs, registry)
    if as_of is not None:
        obs = obs[obs["period"] <= pd.Timestamp(as_of)]
    obs = obs[obs["value"].notna()]
    tgt = target_source(registry)
    tgt_periods = set(obs.loc[obs["source_id"] == tgt, "period"])
    rows = []
    for s in registry["sources"]:
        sid = s["source_id"]
        part = obs[obs["source_id"] == sid]
        p = part["source_period"]
        rows.append({"source_id": sid, "role": s["role"], "granularity": s["granularity"],
                     "enabled": bool(s.get("enabled", True)), "periods": int(p.nunique()),
                     "overlap_periods": len(set(part["period"]) & tgt_periods),
                     "first_period": period_label(p.min()) if len(p) else None,
                     "last_period": period_label(p.max()) if len(p) else None})
    return pd.DataFrame(rows)


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame as plain JSON-safe records (NaN -> null, timestamps -> ISO strings)."""
    import json
    return json.loads(df.to_json(orient="records", date_format="iso"))
