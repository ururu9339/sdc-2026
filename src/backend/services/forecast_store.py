"""Read-only access to what the pipeline has written: the forecast log, the version log, observations,
the job-skill graph and the registry. No forecast is computed here and no language model is used;
explanations are template strings filled only from stored fields.

Paths are read from src.training.config at call time so tests can point them at fixtures.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.store import AppendOnlyLog, ObservationStore, load_json
from src.sources.loaders import GLOBAL_ID
from src.training import config as C
from src.training.audit import signal_audit, source_coverage

GRAPH_FILENAME = "job_skill_graph.csv"
NO_ATTEMPT_REASON = "no retraining attempt has been recorded yet"
PERIOD_WORD = {"W": "weeks", "M": "months", "D": "days"}  # the model grid is config.PERIOD_FREQ (weekly)
DIRECTION_PHRASE = {
    "emerging": "emerging (share of postings rising)",
    "declining": "declining (share of postings falling)",
    "stable": "stable (no clear move)",
}


# ---- helpers ----------------------------------------------------------------------------
def _clean(x: Any) -> Any:
    """Recursively convert numpy/pandas values to JSON-safe Python values (NaN and inf become None)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if x is pd.NaT:
        return None
    return x


def _version_log() -> dict:
    return load_json(C.VERSION_LOG_PATH, {"live": None, "history": []})


def _registry() -> dict:
    return C.load_registry(C.REGISTRY_PATH)


def disclaimer() -> str:
    return _registry()["disclaimer"]


def _observations() -> pd.DataFrame:
    return ObservationStore(C.OBSERVATIONS_PATH).load()


def _graph() -> pd.DataFrame:
    path = C.PROCESSED_DIR / GRAPH_FILENAME
    if not path.exists():
        return pd.DataFrame(columns=["job", "skill", "n_cooccur", "n_job_postings", "weight"])
    return pd.read_csv(path)


def _pct(log_change: float) -> float:
    return 100.0 * (math.exp(log_change) - 1.0)


def _gate_summary(entry: dict | None) -> dict | None:
    if entry is None:
        return None
    g = entry.get("gate") or {}
    return _clean({
        "created_at": entry.get("created_at"), "as_of": entry.get("as_of"), "version": entry.get("version"),
        "status": entry.get("status"), "passed": g.get("passed"), "reason": g.get("reason"),
        "profile": g.get("profile"), "checks": g.get("checks"), "n_folds": g.get("n_folds"),
        "model_mae": g.get("model_mae"), "best_baseline": g.get("best_baseline"),
        "best_baseline_mae": g.get("best_baseline_mae"), "improvement_pct": g.get("improvement_pct"),
        "fold_win_rate": g.get("fold_win_rate"), "coverage": g.get("coverage"),
    })


def latest_attempt() -> dict | None:
    history = _version_log().get("history") or []
    return history[-1] if history else None


def no_live_reason() -> str:
    last = latest_attempt()
    if last is None:
        return NO_ATTEMPT_REASON
    return (last.get("gate") or {}).get("reason") or "gate not passed"


# ---- forecasts ----------------------------------------------------------------------------
def live_version() -> str | None:
    return _version_log().get("live")


def _forecast_records() -> list[dict]:
    return AppendOnlyLog(C.FORECAST_LOG_PATH).read()


def latest_forecasts() -> list[dict]:
    """Records of the newest as_of. Empty when no model is live (forecasts are only served from a live model)."""
    if live_version() is None:
        return []
    recs = _forecast_records()
    if not recs:
        return []
    newest = max(r["as_of"] for r in recs)  # ISO labels sort chronologically
    return [r for r in recs if r["as_of"] == newest]


def forecast_for(entity_id: str) -> dict | None:
    """Newest logged record for the entity, or None (also None when no model is live)."""
    if live_version() is None:
        return None
    mine = [r for r in _forecast_records() if r["entity_id"] == entity_id]
    return max(mine, key=lambda r: r["as_of"]) if mine else None


def select_forecasts(kind: str | None = None, direction: str | None = None, status: str | None = None,
                     limit: int = 100) -> list[dict]:
    out = []
    for r in latest_forecasts():
        if kind and r.get("kind") != kind:
            continue
        if direction and r.get("direction") != direction:
            continue
        if status and r.get("status") != status:
            continue
        out.append(r)
    out.sort(key=lambda r: (r.get("status") != "ok", r["entity_id"]))
    return [_clean(r) for r in out[:limit]]


def forecast_envelope() -> dict:
    """Shared header for a live forecast listing: which model, which gate, which horizon and origin."""
    vlog = _version_log()
    live = vlog.get("live")
    meta = load_json(C.MODELS_DIR / f"forecaster_{live}" / "meta.json", {}) if live else {}
    recs = latest_forecasts()
    first = recs[0] if recs else {}
    return _clean({
        "model_version": live,
        "gate_profile": meta.get("gate_profile") or first.get("gate_profile"),
        "as_of": first.get("as_of"),
        "target_period": first.get("target_period"),
        "horizon_periods": first.get("horizon_periods", meta.get("horizon_periods")),
        "disclaimer": first.get("disclaimer") or _registry().get("disclaimer"),
    })


def no_live_envelope() -> dict:
    return _clean({
        "reason": no_live_reason(),
        "gate": _gate_summary(latest_attempt()),
        "disclaimer": disclaimer(),
    })


# ---- entities -----------------------------------------------------------------------------
def entity_known(entity_id: str) -> bool:
    if entity_id == GLOBAL_ID:
        return False
    obs = _observations()
    return bool((obs["entity_id"] == entity_id).any())


def entity_kind(entity_id: str) -> str | None:
    obs = _observations()
    rows = obs[obs["entity_id"] == entity_id]
    return str(rows["kind"].iloc[0]) if not rows.empty else None


def history(entity_id: str, n_periods: int = 52) -> dict | None:
    """Target-source share series for the entity (native periods, latest n), with the entity's audit row."""
    obs = _observations()
    rows = obs[obs["entity_id"] == entity_id]
    if rows.empty:
        return None
    reg = _registry()
    tgt = C.target_source(reg)
    tgt_rows = rows[rows["source_id"] == tgt].sort_values("period").tail(n_periods)
    audit = signal_audit(obs, reg)
    arow = audit[(audit["source_id"] == tgt) & (audit["entity_id"] == entity_id)]
    granularity = next(s["granularity"] for s in reg["sources"] if s["source_id"] == tgt)
    series = [{"period": C.period_label(p), "value": float(v), "support": _clean(s)}
              for p, v, s in zip(tgt_rows["period"], tgt_rows["value"], tgt_rows["support"])]
    return _clean({
        "entity_id": entity_id, "kind": str(rows["kind"].iloc[0]), "source_id": tgt, "granularity": granularity,
        "n_periods_returned": len(series), "series": series,
        "audit": arow.iloc[0].to_dict() if not arow.empty else None,
    })


# ---- graph ----------------------------------------------------------------------------------
def graph_neighbours(entity_id: str, top_k: int = 10) -> list[dict]:
    """For a job: its skills, highest weight first. For a skill: the jobs it appears in, highest weight first.
    The weight is the share of the job's postings that mention the skill (explanation only)."""
    g = _graph()
    if (g["job"] == entity_id).any():
        rows = g[g["job"] == entity_id].assign(neighbour=lambda d: d["skill"], neighbour_kind="skill")
    elif (g["skill"] == entity_id).any():
        rows = g[g["skill"] == entity_id].assign(neighbour=lambda d: d["job"], neighbour_kind="job")
    else:
        return []
    rows = rows.sort_values(["weight", "n_cooccur", "neighbour"], ascending=[False, False, True]).head(top_k)
    return [_clean({"entity_id": r.neighbour, "kind": r.neighbour_kind, "weight": float(r.weight),
                    "n_cooccur": int(r.n_cooccur)}) for r in rows.itertuples()]


# ---- sources and model -----------------------------------------------------------------------
def sources_info() -> list[dict]:
    reg = _registry()
    obs = _observations()
    cov = source_coverage(obs, reg) if not obs.empty else None
    cov_by = {r["source_id"]: r for r in cov.to_dict("records")} if cov is not None else {}
    counts = obs["source_id"].value_counts().to_dict() if not obs.empty else {}
    out = []
    for s in reg["sources"]:
        sid = s["source_id"]
        c = cov_by.get(sid, {})
        out.append(_clean({
            "source_id": sid, "name": s.get("name"), "role": s.get("role"), "tier": s.get("tier"),
            "reliability": s.get("reliability"), "granularity": s.get("granularity"),
            "enabled": bool(s.get("enabled", True)), "license": s.get("license"),
            "attribution": s.get("attribution"), "source_url": s.get("source_url"),
            "has_data": int(counts.get(sid, 0)) > 0, "rows": int(counts.get(sid, 0)),
            "periods": c.get("periods"), "overlap_periods": c.get("overlap_periods"),
            "first_period": c.get("first_period"), "last_period": c.get("last_period"),
        }))
    return out


def model_info() -> dict:
    vlog = _version_log()
    live = vlog.get("live")
    last = latest_attempt()
    meta = load_json(C.MODELS_DIR / f"forecaster_{live}" / "meta.json", None) if live else None
    live_entry = next((h for h in reversed(vlog.get("history") or []) if h.get("version") == live), None)
    report = load_json(C.REPORT_PATH, None)
    live_block = None
    if live:
        live_block = {
            "version": live,
            "gate_profile": (meta or {}).get("gate_profile") or (live_entry or {}).get("gate_profile"),
            "trained_as_of": (meta or {}).get("trained_as_of") or (live_entry or {}).get("as_of"),
            "horizon_periods": (meta or {}).get("horizon_periods"),
            "n_feature_columns": len((meta or {}).get("feature_columns", [])) or None,
            "sources_used": (live_entry or {}).get("sources_used"),
            "source_importance": (live_entry or {}).get("source_importance"),
            "gate": _gate_summary(live_entry),
        }
    return _clean({
        "live": live_block,
        "live_version": live,
        "active_gate_profile": C.ACTIVE_GATE_PROFILE,
        "attempts": len(vlog.get("history") or []),
        "latest_attempt": _gate_summary(last),
        "audit_summary": (last or {}).get("audit") if last else (report or {}).get("audit_summary"),
        "timeline": (report or {}).get("timeline"),
        "report_generated_at": (report or {}).get("generated_at"),
        "disclaimer": disclaimer(),
    })


# ---- explanation -------------------------------------------------------------------------------
def explain(entity_id: str) -> dict:
    """Template explanation. Every number in the text comes from the stored forecast record."""
    kind = entity_kind(entity_id)
    name = entity_id.replace("_", " ")
    reg = _registry()
    caveats = [reg["disclaimer"]]  # the registry disclaimer is always present
    related = graph_neighbours(entity_id)
    rec = forecast_for(entity_id)
    drivers = {"source_contributions": None, "source_contributions_units": "log change over the horizon",
               "related": related}
    if live_version() is None:
        status = "no_live_model"
        summary = (f"No forecast is issued for {name}: no model has passed the gate yet "
                   f"({no_live_reason()}). The system abstains rather than guess.")
        caveats.append("No live model: no forecasts are served for any entity, so no numbers are shown.")
    elif rec is None:
        status = "insufficient_data"
        summary = f"No forecast is issued for {name}: no forecast has been logged for this entity yet."
    elif rec["status"] != "ok":
        status = "insufficient_data"
        summary = f"No forecast is issued for {name}: {rec.get('reason')}."
    else:
        status = "ok"
        weeks = rec["horizon_periods"]  # periods on the model grid (weekly)
        lo, mid, hi = _pct(rec["p10"]), _pct(rec["p50"]), _pct(rec["p90"])
        sq = rec.get("signal_quality") or "unknown"
        sq_text = {"noise_like": "noise-like", "trend_detected": "movement above sampling noise",
                   "unknown": "not assessable"}.get(sq, sq)
        unit = PERIOD_WORD.get(C.PERIOD_FREQ, "periods")
        summary = (f"{name.capitalize()} ({rec.get('kind') or kind}) is {DIRECTION_PHRASE[rec['direction']]} over "
                   f"the next {weeks} {unit} (to {rec['target_period']}). Median change in its share of postings: "
                   f"{mid:+.1f}%; 10-90% range {lo:+.1f}% to {hi:+.1f}%. Signal quality: {sq_text}; "
                   f"confidence: {rec.get('confidence')}.")
        drivers["source_contributions"] = rec.get("source_contributions")
        if rec.get("gate_profile") == "demo":
            caveats.append("Gate profile 'demo': relaxed constraints (sources are weak proxies), not the strict "
                           "production gate.")
        if sq == "noise_like":
            summary += " Its movement is indistinguishable from random variation, so treat the direction as unconfirmed."
            dr = rec.get("dispersion_ratio")
            caveats.append("Movement is indistinguishable from random variation"
                           + (f" (dispersion ratio {dr:.2f})." if dr is not None else "."))
    return _clean({"entity_id": entity_id, "kind": kind, "status": status, "summary": summary,
                   "drivers": drivers, "caveats": caveats})
