from __future__ import annotations

import pandas as pd

from src.training.baselines import predict_baselines
from src.training.config import (ABSTAIN, AUDIT, DIRECTION_THRESHOLD, DUPLICATE_CAVEAT, GROWTH_LAGS, HORIZON,
                                 SUPPORT_WINDOW, period_label, shift_periods, to_period_start)
from src.training.features import column_source_map
from src.training.train import predict, source_contributions

MID = GROWTH_LAGS[1]


def _direction(p50: float, p10: float, p90: float) -> tuple[str, str]:
    if p50 > DIRECTION_THRESHOLD:
        return "emerging", "high" if p10 > 0 else "moderate"
    if p50 < -DIRECTION_THRESHOLD:
        return "declining", "high" if p90 < 0 else "moderate"
    return "stable", "high" if (p10 > -DIRECTION_THRESHOLD and p90 < DIRECTION_THRESHOLD) else "moderate"


def _signal_quality(audit: pd.DataFrame | None, ent) -> tuple[str, float | None]:
    if audit is None or ent not in audit.index:
        return "unknown", None
    r = audit.loc[ent]
    dr = r["dispersion_ratio"]
    if pd.isna(dr) or r["n_periods"] < AUDIT["min_periods"]:
        return "unknown", None
    return ("noise_like" if r["noise_like"] else "trend_detected"), float(dr)


COPIED_FROM_CANONICAL = ("status", "reason", "p50", "p10", "p90", "direction", "baseline_persistence", "baseline_trend")


def _apply_duplicates(out: list[dict], duplicates: list[dict], base: dict) -> None:
    by_ent = {r["entity_id"]: r for r in out}
    for g in duplicates:
        canon_id = g["canonical"]
        canon = by_ent.get(canon_id)
        if canon is None:
            continue
        group = list(g["members"])
        canon.update(duplicate_group=group, shares_signal_with=None)
        for m in group:
            if m == canon_id:
                continue
            rec = {**base, "entity_id": m, "kind": g.get("member_kinds", {}).get(m, canon.get("kind")),
                   "forecast_id": f"{base['model_version']}:{m}:{base['as_of']}",
                   "signal_quality": canon.get("signal_quality"), "dispersion_ratio": canon.get("dispersion_ratio"),
                   **{k: canon[k] for k in COPIED_FROM_CANONICAL if k in canon}}
            if rec.get("status") == "ok":
                rec["confidence"] = "low"
            rec.update(shares_signal_with=canon_id, duplicate_group=group, source_contributions=None,
                       caveat=DUPLICATE_CAVEAT.format(canon=canon_id))
            out.append(rec)


def make_forecasts(models, meta: dict, df: pd.DataFrame, as_of, registry: dict, horizon: int = HORIZON,
                   audit: pd.DataFrame | None = None, duplicates: list[dict] | None = None) -> list[dict]:
    as_of = to_period_start(pd.Timestamp(as_of))
    cols = meta["feature_columns"]
    cur = df[df["period"] == as_of].set_index("entity_id")
    base = {"model_version": meta["version"], "gate_profile": meta["gate_profile"],
            "as_of": period_label(as_of), "target_period": period_label(shift_periods(as_of, horizon)),
            "horizon_periods": horizon, "disclaimer": registry.get("disclaimer")}
    out, ok_idx = [], []
    for ent in sorted(df["entity_id"].unique()):
        sq, dr = _signal_quality(audit, ent)
        rec = {**base, "entity_id": ent, "forecast_id": f"{meta['version']}:{ent}:{base['as_of']}",
               "signal_quality": sq, "dispersion_ratio": dr}
        if ent not in cur.index:
            out.append({**rec, "kind": None, "status": "insufficient_data", "reason": "no_current_observation"})
            continue
        r = cur.loc[ent]
        rec["kind"] = r["kind"]
        reason = None
        if r["n_obs"] < ABSTAIN["min_obs_periods"]:
            reason = f"history_too_short ({int(r['n_obs'])} < {ABSTAIN['min_obs_periods']} periods)"
        elif not (support := r[f"support_mean{SUPPORT_WINDOW}"]) >= ABSTAIN["min_mean_support"]:
            reason = f"support_too_thin (mean {support:.1f} < {ABSTAIN['min_mean_support']})"
        elif pd.isna(r[f"g{MID}"]) or pd.isna(r["lvl"]):
            reason = "missing_features"
        elif sq == "noise_like" and ABSTAIN["abstain_on_noise"]:
            reason = "no_detectable_trend"
        if reason:
            out.append({**rec, "status": "insufficient_data", "reason": reason})
        else:
            ok_idx.append(ent)
            out.append(rec)
    if ok_idx:
        X = cur.loc[ok_idx, cols].astype(float)
        pr = predict(models, X)
        contrib = source_contributions(models, X, column_source_map(cols, registry))
        bl = predict_baselines(cur.loc[ok_idx], horizon)
        by_ent = {r["entity_id"]: r for r in out}
        for i, ent in enumerate(ok_idx):
            p50, p10, p90 = (float(pr.loc[ent, c]) for c in ("p50", "p10", "p90"))
            rec = by_ent[ent]
            if p90 - p10 > ABSTAIN["max_interval_width"]:
                rec.update(status="insufficient_data", reason=f"interval_too_wide ({p90 - p10:.2f})")
                continue
            direction, conf = _direction(p50, p10, p90)
            if rec["signal_quality"] == "noise_like":
                conf = "low"
            rec.update(status="ok", p50=p50, p10=p10, p90=p90, direction=direction, confidence=conf,
                       baseline_persistence=float(bl["persistence"][i]), baseline_trend=float(bl["trend"][i]),
                       source_contributions={k: float(v) for k, v in contrib.loc[ent].items()})
    if duplicates:
        _apply_duplicates(out, duplicates, base)
    return out
