"""Perceive -> decide -> act -> log.

    python3 -m src.pipeline.agent_loop run [--as-of YYYY-MM-DD]
    python3 -m src.pipeline.agent_loop replay --start YYYY-MM-DD --end YYYY-MM-DD [--step N]
    python3 -m src.pipeline.agent_loop status

perceive: pull new rows from every enabled source into the accumulating table, check source health.
decide:   retrain if there is new data, no live model, or live accuracy has drifted below baseline.
act:      refit on the FULL table, walk-forward backtest, fixed gate (active profile); promote only if it
          passes. Then forecast with the live model (abstaining on thin series) and reconcile matured
          forecasts.
log:      append-only forecast log, outcome log and run log.
preview:  while NO validated live model exists, a gate-rejected candidate is still fitted on the full usable
          table and its forecasts go to a SEPARATE preview log (validated: false, feed: "preview"), scored
          in a separate preview outcome log. The live pointer stays null; live logs are never written here.
`replay` re-runs the loop as if each period were 'today' (loaders are sliced to as_of), so continuous
retraining can be demonstrated on the real data. Dates may be any day; they are snapped to the period.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.pipeline.store import AppendOnlyLog, ObservationStore, load_json, save_json
from src.sources.loaders import get_loader
from src.training import config as C
from src.training.audit import records, signal_audit, source_coverage, summarise_audit
from src.training.backtest import evaluate_gate, walk_forward
from src.training.features import build_feature_table, column_source_map, feature_columns, usable
from src.training.forecast import make_forecasts
from src.training.train import fit_models, load_models, save_models, source_contributions


class AgentLoop:
    def __init__(self):
        self.registry = C.load_registry()
        self.store = ObservationStore()
        self.forecast_log = AppendOnlyLog(C.FORECAST_LOG_PATH)
        self.outcome_log = AppendOnlyLog(C.OUTCOME_LOG_PATH)
        self.preview_log = AppendOnlyLog(C.PREVIEW_LOG_PATH)
        self.preview_outcome_log = AppendOnlyLog(C.PREVIEW_OUTCOME_LOG_PATH)
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
            return {"ok": False, "reason": f"no '{tgt}' data yet; run src.sources.build_all", "added": added}
        as_of = as_of or tgt_obs["period"].max()
        as_of = min(as_of, tgt_obs["period"].max())
        health, active, weights = {}, set(), {}
        for src in C.leading_sources(self.registry):
            sid = src["source_id"]
            s = obs[(obs["source_id"] == sid) & (obs["period"] <= as_of)]
            if s.empty:
                health[sid] = "no_data"
                continue
            available = C.span_end(s["period"].max(), src["granularity"]) + pd.Timedelta(days=src.get("release_lag_days", 0))
            stale_days = (as_of - available).days
            if stale_days > src["max_staleness_days"]:
                health[sid] = f"stale ({stale_days} days)"
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
        if last is None or C.periods_between(C.parse_period_label(last), as_of) >= C.RETRAIN_EVERY_PERIODS:
            return True, "new data since last training"
        if drift.get("drifted"):
            return True, f"drift: {drift['reason']}"
        return False, "model is current"

    # ---- act ---------------------------------------------------------------------------
    def retrain(self, feats: pd.DataFrame, as_of: pd.Timestamp, vlog: dict, active: set, audit: list[dict]) -> dict:
        cols = feature_columns(feats)
        profile = C.active_gate_profile()
        folds, _ = walk_forward(feats, cols, profile=profile)
        gate = evaluate_gate(folds, profile)
        n_prev = len([h for h in vlog["history"] if h.get("version")])
        entry = {"created_at": datetime.now(timezone.utc).isoformat(), "as_of": C.period_label(as_of),
                 "gate_profile": gate["profile"], "gate": gate,
                 "folds": folds.to_dict("records") if not folds.empty else [],
                 "sources_used": sorted(active), "audit": audit, "version": None, "status": "rejected"}
        train = usable(feats)
        train = train[train["y"].notna()]
        if gate["passed"]:
            models = fit_models(train, cols)
            version = f"v{n_prev + 1}"
            entry.update(version=version, status="promoted", n_train_rows=len(train),
                         source_importance=self._importance(models, train, cols))
            meta = {"version": version, "feature_columns": cols, "horizon_periods": C.HORIZON,
                    "trained_as_of": C.period_label(as_of), "gate_profile": gate["profile"], "gate": gate,
                    "disclaimer": self.registry["disclaimer"]}
            save_models(models, C.MODELS_DIR / f"forecaster_{version}", meta)
            vlog["live"] = version
        elif C.PREVIEW_ENABLED and not vlog.get("live"):
            entry.update(self._fit_preview(train, cols, as_of, gate, vlog))
        vlog["history"].append(entry)
        save_json(C.VERSION_LOG_PATH, vlog)
        return entry

    def _importance(self, models, train: pd.DataFrame, cols: list[str]) -> dict:
        X = train[cols].astype(float)
        imp = source_contributions(models, X, column_source_map(cols, self.registry)).abs().mean()
        return {k: float(v) for k, v in imp.items()}

    def _fit_preview(self, train: pd.DataFrame, cols: list[str], as_of: pd.Timestamp, gate: dict, vlog: dict) -> dict:
        """Fit the rejected candidate for the preview feed. Stored apart from the live models; never promoted."""
        if train.empty:
            return {"status": "rejected_preview", "preview_version": None, "preview_reason": "no usable training rows"}
        n_prev = sum(1 for h in vlog["history"] if h.get("preview_version"))
        pv = f"v{n_prev + 1}"
        models = fit_models(train, cols)
        dir_name = f"preview_{pv}"
        meta = {"version": f"preview:{pv}", "feed": "preview", "validated": False, "preview_reason": gate["reason"],
                "feature_columns": cols, "horizon_periods": C.HORIZON, "trained_as_of": C.period_label(as_of),
                "gate_profile": gate["profile"], "gate": gate, "disclaimer": self.registry["disclaimer"]}
        save_models(models, C.MODELS_DIR / dir_name, meta)
        vlog["preview"] = dir_name
        return {"status": "rejected_preview", "preview_version": f"preview:{pv}", "model_dir": dir_name,
                "preview_reason": gate["reason"], "n_train_rows": len(train),
                "source_importance": self._importance(models, train, cols)}

    def reconcile(self, feats: pd.DataFrame, as_of: pd.Timestamp, feed: str = "live") -> list[dict]:
        """Score matured forecasts against what actually happened. Appends; never edits the forecast log.
        Preview forecasts are scored into their own outcome log, so live statistics never mix with them."""
        preview = feed == "preview"
        fc_log = self.preview_log if preview else self.forecast_log
        oc_log = self.preview_outcome_log if preview else self.outcome_log
        done = {r["forecast_id"] for r in oc_log.read()}
        actual = feats.set_index(["entity_id", "period"])["lvl"]
        out = []
        for rec in fc_log.read():
            if rec.get("status") != "ok" or rec["forecast_id"] in done:
                continue
            target = C.parse_period_label(rec["target_period"])
            if target > as_of:
                continue
            key0, key1 = (rec["entity_id"], C.parse_period_label(rec["as_of"])), (rec["entity_id"], target)
            if key0 not in actual.index or key1 not in actual.index or pd.isna(actual[key0]) or pd.isna(actual[key1]):
                continue
            y = float(actual[key1] - actual[key0])
            out.append({"forecast_id": rec["forecast_id"], "model_version": rec["model_version"],
                        "gate_profile": rec["gate_profile"], "entity_id": rec["entity_id"],
                        "target_period": rec["target_period"], "actual": y,
                        "abs_err": abs(y - rec["p50"]), "abs_err_persistence": abs(y),
                        "abs_err_trend": abs(y - rec["baseline_trend"]),
                        "in_interval": bool(rec["p10"] <= y <= rec["p90"]),
                        **({"feed": "preview", "validated": False} if preview else {})})
        if out:
            oc_log.append_many(out)
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
        # The origin is a period start. Snapping before loading keeps a partly-elapsed week out of the table.
        as_of = C.to_period_start(pd.Timestamp(as_of)) if as_of is not None else None
        p = self.perceive(as_of)
        run = {"run_at": datetime.now(timezone.utc).isoformat(), "added_rows": p["added"]}
        if not p["ok"]:
            run.update(status="no_data", reason=p["reason"])
            self.run_log.append_many([run])
            return run
        as_of = p["as_of"]
        run.update(as_of=C.period_label(as_of), source_health=p["health"])
        state = load_json(C.STATE_PATH, {})
        vlog = load_json(C.VERSION_LOG_PATH, {"live": None, "history": []})
        obs = self.store.load()
        tgt = C.target_source(self.registry)
        audit = signal_audit(obs, self.registry, as_of)  # causal: only figures known at as_of
        target_audit = audit[audit["source_id"] == tgt].set_index("entity_id")
        run["source_coverage"] = records(source_coverage(obs, self.registry, as_of))
        feats = build_feature_table(obs, self.registry, as_of, weights=p["weights"], active=p["active"])

        matured = self.reconcile(feats, as_of)
        run["outcomes_reconciled"] = len(matured)
        run["preview_outcomes_reconciled"] = len(self.reconcile(feats, as_of, feed="preview"))
        retrain, why = self.decide(as_of, state, vlog, self.drift_check())
        run["decision"] = {"retrain": retrain, "why": why}
        if retrain:
            entry = self.retrain(feats, as_of, vlog, p["active"], records(summarise_audit(audit, self.registry)))
            run["retrain"] = {"status": entry["status"], "version": entry["version"],
                              "preview_version": entry.get("preview_version"), "gate": entry["gate"],
                              "gate_profile": entry["gate_profile"]}
            state["last_trained_as_of"] = C.period_label(as_of)
            save_json(C.STATE_PATH, state)
        if vlog.get("live"):
            return self._issue(run, "live", f"forecaster_{vlog['live']}", as_of, feats, target_audit)
        if C.PREVIEW_ENABLED and vlog.get("preview"):
            return self._issue(run, "preview", vlog["preview"], as_of, feats, target_audit)
        run.update(status="no_live_model", feed="none", reason="gate not passed yet; no forecasts issued")
        self.run_log.append_many([run])
        return run

    def _issue(self, run: dict, feed: str, model_dir: str, as_of: pd.Timestamp, feats: pd.DataFrame,
               target_audit: pd.DataFrame) -> dict:
        """Forecast with a stored model into the log of its feed. Abstention and signal rules are identical."""
        models, meta = load_models(C.MODELS_DIR / model_dir)
        missing = set(meta["feature_columns"]) - set(feats.columns)
        if missing:
            run.update(status="feature_mismatch", feed=feed, reason=f"{feed} model needs {sorted(missing)}")
            self.run_log.append_many([run])
            return run
        preview = feed == "preview"
        log = self.preview_log if preview else self.forecast_log
        existing = {r["forecast_id"] for r in log.read()}
        fc = [f for f in make_forecasts(models, meta, feats, as_of, self.registry, audit=target_audit)
              if f["forecast_id"] not in existing]
        if preview:
            for f in fc:
                f.update(validated=False, feed="preview", preview_reason=meta["preview_reason"])
        log.append_many(fc)
        quality = {q: sum(f["signal_quality"] == q for f in fc) for q in ("noise_like", "trend_detected", "unknown")}
        run.update(status="ok", feed=feed, gate_profile=meta["gate_profile"], n_forecasts=len(fc),
                   n_abstained=sum(f["status"] != "ok" for f in fc), signal_quality=quality)
        if preview:
            run.update(preview_version=meta["version"], preview_reason=meta["preview_reason"])
        else:
            run["live_version"] = meta["version"]
        self.run_log.append_many([run])
        return run

    def replay(self, start: str, end: str, step: int = 1) -> list[dict]:
        starts = C.period_range(pd.Timestamp(start), pd.Timestamp(end))[::step]
        return [self.run_once(m) for m in starts]

    def status(self) -> dict:
        vlog = load_json(C.VERSION_LOG_PATH, {"live": None, "history": []})
        out = self.outcome_log.read()
        s = {"live": vlog["live"], "gate_profile": C.ACTIVE_GATE_PROFILE, "attempts": len(vlog["history"]),
             "logs_intact": all(log.verify() for log in (self.forecast_log, self.outcome_log, self.run_log)),
             "matured_forecasts": len(out),
             "preview_model": vlog.get("preview"), "preview_forecasts": len(self.preview_log.read()),
             "preview_matured": len(self.preview_outcome_log.read()),
             "preview_logs_intact": all(log.verify() for log in (self.preview_log, self.preview_outcome_log))}
        if out:
            s.update(live_mae=float(np.mean([r["abs_err"] for r in out])),
                     persistence_mae=float(np.mean([r["abs_err_persistence"] for r in out])),
                     interval_coverage=float(np.mean([r["in_interval"] for r in out])))
        return s


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--as-of")
    rp = sub.add_parser("replay")
    rp.add_argument("--start", required=True)
    rp.add_argument("--end", required=True)
    rp.add_argument("--step", type=int, default=1, help="run every N periods (default 1)")
    sub.add_parser("status")
    a = ap.parse_args()
    loop = AgentLoop()
    if a.cmd == "run":
        print(save_summary(loop.run_once(a.as_of)))
    elif a.cmd == "replay":
        for res in loop.replay(a.start, a.end, a.step):
            print(save_summary(res))
    else:
        print(loop.status())


def save_summary(run: dict) -> str:
    keys = ("as_of", "status", "feed", "reason", "decision", "live_version", "preview_version", "gate_profile",
            "n_forecasts", "n_abstained")
    return " | ".join(f"{k}={run[k]}" for k in keys if k in run)


if __name__ == "__main__":
    main()
