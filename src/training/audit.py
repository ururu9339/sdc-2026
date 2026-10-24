from __future__ import annotations

import numpy as np
import pandas as pd

from src.sources.loaders import GLOBAL_ID
from src.training.config import AUDIT, DEDUP, GRANULARITY_FREQ, period_label, target_source
from src.training.features import shift_to_publication

AUDIT_COLS = ["source_id", "entity_id", "kind", "n_periods", "first_period", "last_period", "mean_share",
              "dispersion_ratio", "abs_corr_with_time", "noise_like"]


def _audit_row(g: pd.DataFrame) -> dict:
    v = g["value"].astype(float).to_numpy()
    sup = g["support"].astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        n_t = np.round(sup / v)
    ok = (v > 0) & np.isfinite(n_t) & (n_t > 0)
    dr = np.nan
    if ok.any() and len(v) >= 2:
        p_bar = sup[ok].sum() / n_t[ok].sum()
        exp_var = float(np.mean(p_bar * (1 - p_bar) / n_t[ok]))
        if exp_var > 0:
            dr = float(np.var(v, ddof=1)) / exp_var
    t = g["period"].astype("int64").to_numpy(dtype=float)
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
    obs = shift_to_publication(obs, registry)
    if as_of is not None:
        obs = obs[obs["period"] <= pd.Timestamp(as_of)]
    obs = obs[(obs["entity_id"] != GLOBAL_ID) & obs["value"].notna()].sort_values("period")
    rows = [_audit_row(g) for _, g in obs.groupby(["source_id", "entity_id"], sort=True)]
    return pd.DataFrame(rows, columns=AUDIT_COLS)


def summarise_audit(audit: pd.DataFrame, registry: dict) -> pd.DataFrame:
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


def duplicate_groups(obs: pd.DataFrame, registry: dict, as_of=None) -> list[dict]:
    tgt = target_source(registry)
    obs = shift_to_publication(obs, registry)
    if as_of is not None:
        obs = obs[obs["period"] <= pd.Timestamp(as_of)]
    obs = obs[(obs["source_id"] == tgt) & (obs["entity_id"] != GLOBAL_ID) & obs["value"].notna()]
    if obs.empty:
        return []
    wide = obs.pivot_table(index="period", columns="entity_id", values="value", aggfunc="last").sort_index(axis=1)
    kinds = obs.drop_duplicates("entity_id").set_index("entity_id")["kind"]
    ids = list(wide.columns)
    X = wide.to_numpy(dtype=float)
    seen = ~np.isnan(X)
    parent = list(range(len(ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            common = seen[:, i] & seen[:, j]
            if int(common.sum()) < DEDUP["min_periods"]:
                continue
            a, b = X[common, i], X[common, j]
            if int((a > 0).sum()) < DEDUP["min_periods"]:
                continue
            if np.allclose(a, b, rtol=DEDUP["rel_tol"], atol=0.0):
                parent[find(j)] = find(i)
    comps: dict[int, list[int]] = {}
    for i in range(len(ids)):
        comps.setdefault(find(i), []).append(i)
    groups = []
    for idx in comps.values():
        if len(idx) < 2:
            continue
        members = sorted(ids[i] for i in idx)
        shared = int(seen[:, idx].all(axis=1).sum())
        groups.append({"canonical": members[0], "members": members,
                       "kinds": sorted({str(kinds[m]) for m in members}),
                       "member_kinds": {m: str(kinds[m]) for m in members}, "shared_periods": shared})
    return sorted(groups, key=lambda g: g["canonical"])


def non_canonical_ids(groups: list[dict]) -> set[str]:
    return {m for g in groups for m in g["members"] if m != g["canonical"]}


def duplicate_summary(groups: list[dict], n_entities: int | None = None) -> dict:
    in_groups = sum(len(g["members"]) for g in groups)
    collapsed = sum(len(g["members"]) - 1 for g in groups)
    out = {"groups": len(groups), "entities_in_groups": in_groups, "entities_collapsed": collapsed}
    if n_entities is not None:
        out.update(series_before=int(n_entities), series_after=int(n_entities - collapsed))
    return out


def records(df: pd.DataFrame) -> list[dict]:
    import json
    return json.loads(df.to_json(orient="records", date_format="iso"))
