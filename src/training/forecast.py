"""Turn a feature table + a trained model into forecast records, with abstention.

Qwen never sets a number: explanations are generated downstream from these structured fields."""
from __future__ import annotations

import pandas as pd

from src.training.baselines import predict_baselines
from src.training.config import ABSTAIN, DIRECTION_THRESHOLD, HORIZON
from src.training.features import column_source_map
from src.training.train import predict, source_contributions


def _direction(p50: float, p10: float, p90: float) -> tuple[str, str]:
    if p50 > DIRECTION_THRESHOLD:
        return "emerging", "high" if p10 > 0 else "moderate"
    if p50 < -DIRECTION_THRESHOLD:
        return "declining", "high" if p90 < 0 else "moderate"
    return "stable", "high" if (p10 > -DIRECTION_THRESHOLD and p90 < DIRECTION_THRESHOLD) else "moderate"


def make_forecasts(models, meta: dict, df: pd.DataFrame, as_of, registry: dict, horizon: int = HORIZON) -> list[dict]:
    as_of = pd.Timestamp(as_of)
    cols = meta["feature_columns"]
    cur = df[df["period"] == as_of].set_index("entity_id")
    target_period = as_of + pd.DateOffset(months=horizon)
    base = {"model_version": meta["version"], "as_of": as_of.strftime("%Y-%m"),
            "target_period": target_period.strftime("%Y-%m"), "horizon_months": horizon,
            "disclaimer": registry.get("disclaimer")}
    out, ok_idx = [], []
    for ent in sorted(df["entity_id"].unique()):
        rec = {**base, "entity_id": ent, "forecast_id": f"{meta['version']}:{ent}:{base['as_of']}"}
        if ent not in cur.index:
            out.append({**rec, "kind": None, "status": "insufficient_data", "reason": "no_current_observation"})
            continue
        r = cur.loc[ent]
        rec["kind"] = r["kind"]
        reason = None
        if r["n_obs"] < ABSTAIN["min_obs_months"]:
            reason = f"history_too_short ({int(r['n_obs'])} < {ABSTAIN['min_obs_months']} months)"
        elif not (r["support_mean6"] >= ABSTAIN["min_mean_support"]):
            reason = f"support_too_thin (mean {r['support_mean6']:.1f} < {ABSTAIN['min_mean_support']})"
        elif pd.isna(r["g3"]) or pd.isna(r["lvl"]):
            reason = "missing_features"
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
            rec.update(status="ok", p50=p50, p10=p10, p90=p90, direction=direction, confidence=conf,
                       baseline_persistence=float(bl["persistence"][i]), baseline_trend=float(bl["trend"][i]),
                       source_contributions={k: float(v) for k, v in contrib.loc[ent].items()})
    return out
