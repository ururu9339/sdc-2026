"""Perceive -> decide -> act -> log.

    python -m src.pipeline.agent_loop run                         # one pass at the latest data
    python -m src.pipeline.agent_loop replay --start 2024-06 --end 2024-12   # step month by month
    python -m src.pipeline.agent_loop status

perceive: pull new rows from every enabled source into the accumulating table, check source health.
decide:   retrain if there is new data, no live model, or live accuracy has drifted below baseline.
act:      refit on the FULL table, walk-forward backtest, fixed gate; promote only if it passes.
          Then forecast with the live model (abstaining on thin series) and reconcile matured forecasts.
log:      append-only forecast log, outcome log and run log.
`replay` re-runs the loop as if each month were 'today' (loaders are sliced to as_of), which
demonstrates continuous retraining on real data without waiting for new months to arrive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.pipeline.store import AppendOnlyLog, ObservationStore, load_json, save_json
from src.sources.loaders import get_loader
from src.training import config as C
from src.training.backtest import evaluate_gate, walk_forward
from src.training.features import build_feature_table, column_source_map, feature_columns, usable
from src.training.forecast import make_forecasts
from src.training.train import fit_models, load_models, save_models, source_contributions


def _months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


class AgentLoop:
    def __init__(self):
        self.registry = C.load_registry()
        self.store = ObservationStore()
        self.forecast_log = AppendOnlyLog(C.FORECAST_LOG_PATH)
        self.outcome_log = AppendOnlyLog(C.OUTCOME_LOG_PATH)
        self.run_log = AppendOnlyLog(C.RUN_LOG_PATH)

    # ---- perceive ----------------------------------------------------------------------
    def perceive(self, as_of: pd.Timestamp | None) -> dict:
        added = {}
        for src in C.enabled_sources(self.registry):
            df = get_loader(src).fetch(as_of)
            added[src["source_id"]] = self.store.upsert(df)
        obs = self.store.load()
        tgt = C.target_source(self.registry)
        tgt_obs = obs[obs["source_id"] == tgt]
        if tgt_obs.empty:
            return {"ok": False, "reason": f"no '{tgt}' data yet; run src.sources.jobstreet build", "added": added}
        as_of = as_of or tgt_obs["period"].max()
        as_of = min(as_of, tgt_obs["period"].max())
        health, active, weights = {}, set(), {}
        for src in C.leading_sources(self.registry):
            sid = src["source_id"]
            s = obs[(obs["source_id"] == sid) & (obs["period"] <= as_of)]
            if s.empty:
                health[sid] = "no_data"
                continue
            stale = _months_between(s["period"].max() + pd.DateOffset(months=src.get("release_lag_months", 0)), as_of)
            if stale > src["max_staleness_months"]:
                health[sid] = f"stale ({stale} months)"
                continue
            health[sid] = "ok"
            active.add(sid)
            weights[sid] = src["reliability"]
        return {"ok": True, "as_of": as_of, "added": added, "health": health, "active": active, "weights": weights}

    # ---- decide ------------------------------------------------------------------------
    def decide(self, as_of: pd.Timestamp, state: dict, vlog: dict, drift: dict) -> tuple[bool, str]:
        if not vlog.get("live"):
            return True, "no live model"
        last = state.get("last_trained_as_of")
        if last is None or _months_between(pd.Timestamp(last), as_of) >= C.RETRAIN_EVERY_PERIODS:
            return True, "new data since last training"
        if drift.get("drifted"):
            return True, f"drift: {drift['reason']}"
        return False, "model is current"

    # ---- act ---------------------------------------------------------------------------
    def retrain(self, feats: pd.DataFrame, as_of: pd.Timestamp, vlog: dict, active: set) -> dict:
        cols = feature_columns(feats)
        folds, _ = walk_forward(feats, cols)
        gate = evaluate_gate(folds)
        n_prev = len([h for h in vlog["history"] if h.get("version")])
        entry = {"created_at": datetime.now(timezone.utc).isoformat(), "as_of": as_of.strftime("%Y-%m"),
                 "gate": gate, "folds": folds.to_dict("records") if not folds.empty else [],
                 "sources_used": sorted(active), "version": None, "status": "rejected"}
        if gate["passed"]:
            train = usable(feats)
            train = train[train["y"].notna()]
            models = fit_models(train, cols)
            version = f"v{n_prev + 1}"
            X = train[cols].astype(float)
            imp = source_contributions(models, X, column_source_map(cols, self.registry)).abs().mean()
            entry.update(version=version, status="promoted", n_train_rows=len(train),
                         source_importance={k: float(v) for k, v in imp.items()})
            meta = {"version": version, "feature_columns": cols, "horizon": C.HORIZON,
                    "trained_as_of": as_of.strftime("%Y-%m"), "gate": gate, "disclaimer": self.registry["disclaimer"]}
            save_models(models, C.MODELS_DIR / f"forecaster_{version}", meta)
            vlog["live"] = version
        vlog["history"].append(entry)
        save_json(C.VERSION_LOG_PATH, vlog)
        return entry

    def reconcile(self, feats: pd.DataFrame, as_of: pd.Timestamp) -> list[dict]:
        """Score matured forecasts against what actually happened. Appends; never edits the forecast log."""
        done = {r["forecast_id"] for r in self.outcome_log.read()}
        actual = feats.set_index(["entity_id", "period"])["lvl"]
        out = []
        for rec in self.forecast_log.read():
            if rec.get("status") != "ok" or rec["forecast_id"] in done:
                continue
            target = pd.Timestamp(rec["target_period"] + "-01")
            if target > as_of:
                continue
            key0, key1 = (rec["entity_id"], pd.Timestamp(rec["as_of"] + "-01")), (rec["entity_id"], target)
            if key0 not in actual.index or key1 not in actual.index or pd.isna(actual[key0]) or pd.isna(actual[key1]):
                continue
            y = float(actual[key1] - actual[key0])
            out.append({"forecast_id": rec["forecast_id"], "model_version": rec["model_version"],
                        "entity_id": rec["entity_id"], "target_period": rec["target_period"], "actual": y,
                        "abs_err": abs(y - rec["p50"]), "abs_err_persistence": abs(y),
                        "abs_err_trend": abs(y - rec["baseline_trend"]),
                        "in_interval": bool(rec["p10"] <= y <= rec["p90"])})
        if out:
            self.outcome_log.append_many(out)
        return out

    def drift_check(self) -> dict:
        res = self.outcome_log.read()
        if len(res) < C.DRIFT["min_matured"]:
            return {"drifted": False, "reason": f"only {len(res)} matured forecasts"}
        recent = res[-C.DRIFT["min_matured"]:]
        m, b = np.mean([r["abs_err"] for r in recent]), min(
            np.mean([r["abs_err_persistence"] for r in recent]), np.mean([r["abs_err_trend"] for r in recent]))
        return {"drifted": bool(m > C.DRIFT["tolerance"] * b), "reason": f"live MAE {m:.3f} vs baseline {b:.3f}"}

    # ---- one pass ----------------------------------------------------------------------
    def run_once(self, as_of: str | pd.Timestamp | None = None) -> dict:
        as_of = pd.Timestamp(as_of) if as_of is not None else None
        p = self.perceive(as_of)
        run = {"run_at": datetime.now(timezone.utc).isoformat(), "added_rows": p["added"]}
        if not p["ok"]:
            run.update(status="no_data", reason=p["reason"])
            self.run_log.append_many([run])
            return run
        as_of = p["as_of"]
        run.update(as_of=as_of.strftime("%Y-%m"), source_health=p["health"])
        state = load_json(C.STATE_PATH, {})
        vlog = load_json(C.VERSION_LOG_PATH, {"live": None, "history": []})
        feats = build_feature_table(self.store.load(), self.registry, as_of, weights=p["weights"], active=p["active"])

        matured = self.reconcile(feats, as_of)
        run["outcomes_reconciled"] = len(matured)
        retrain, why = self.decide(as_of, state, vlog, self.drift_check())
        run["decision"] = {"retrain": retrain, "why": why}
        if retrain:
            entry = self.retrain(feats, as_of, vlog, p["active"])
            run["retrain"] = {"status": entry["status"], "version": entry["version"], "gate": entry["gate"]}
            state["last_trained_as_of"] = as_of.strftime("%Y-%m")
            save_json(C.STATE_PATH, state)
        if not vlog.get("live"):
            run.update(status="no_live_model", reason="gate not passed yet; no forecasts issued")
            self.run_log.append_many([run])
            return run

        models, meta = load_models(C.MODELS_DIR / f"forecaster_{vlog['live']}")
        missing = set(meta["feature_columns"]) - set(feats.columns)
        if missing:
            run.update(status="feature_mismatch", reason=f"live model needs {sorted(missing)}")
            self.run_log.append_many([run])
            return run
        existing = {r["forecast_id"] for r in self.forecast_log.read()}
        fc = [f for f in make_forecasts(models, meta, feats, as_of, self.registry) if f["forecast_id"] not in existing]
        self.forecast_log.append_many(fc)
        run.update(status="ok", live_version=vlog["live"], n_forecasts=len(fc),
                   n_abstained=sum(f["status"] != "ok" for f in fc))
        self.run_log.append_many([run])
        return run

    def replay(self, start: str, end: str) -> list[dict]:
        return [self.run_once(m) for m in pd.date_range(start, end, freq="MS")]

    def status(self) -> dict:
        vlog = load_json(C.VERSION_LOG_PATH, {"live": None, "history": []})
        out = self.outcome_log.read()
        s = {"live": vlog["live"], "attempts": len(vlog["history"]),
             "logs_intact": all(l.verify() for l in (self.forecast_log, self.outcome_log, self.run_log)),
             "matured_forecasts": len(out)}
        if out:
            s.update(live_mae=float(np.mean([r["abs_err"] for r in out])),
                     persistence_mae=float(np.mean([r["abs_err_persistence"] for r in out])),
                     interval_coverage=float(np.mean([r["in_interval"] for r in out])))
        return s


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--as-of")
    rp = sub.add_parser("replay"); rp.add_argument("--start", required=True); rp.add_argument("--end", required=True)
    sub.add_parser("status")
    a = ap.parse_args()
    loop = AgentLoop()
    if a.cmd == "run":
        print(save_summary(loop.run_once(a.as_of)))
    elif a.cmd == "replay":
        for res in loop.replay(a.start, a.end):
            print(save_summary(res))
    else:
        print(loop.status())


def save_summary(run: dict) -> str:
    keys = ("as_of", "status", "reason", "decision", "live_version", "n_forecasts", "n_abstained")
    return " | ".join(f"{k}={run[k]}" for k in keys if k in run)


if __name__ == "__main__":
    main()
