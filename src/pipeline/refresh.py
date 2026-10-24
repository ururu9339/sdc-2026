from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.pipeline import export_bundle
from src.pipeline import export_forecasts as export
from src.pipeline.agent_loop import AgentLoop
from src.pipeline.store import AppendOnlyLog, load_json, save_json
from src.training import config as C

STALE_HOURS = 6
FAILURE_STATUSES = {"no_data", "feature_mismatch"}
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


class LockHeld(RuntimeError):
    pass


class RefreshRefused(RuntimeError):
    pass


def lock_path() -> Path:
    return C.DATA_DIR / "refresh.lock"


def last_refresh_path() -> Path:
    return C.DATA_DIR / "last_refresh.json"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_stale(path: Path, now: datetime) -> bool:
    try:
        info = json.loads(path.read_text())
        started = datetime.fromisoformat(info["started_at"])
        pid = int(info["pid"])
    except (OSError, ValueError, KeyError, TypeError):

        age = now.timestamp() - path.stat().st_mtime
        return age > STALE_HOURS * 3600
    if now - started > timedelta(hours=STALE_HOURS):
        return True
    return not _pid_alive(pid)


class RefreshLock:

    def __init__(self, path: Path, now: datetime | None = None):
        self.path = path
        self.now = now or datetime.now(timezone.utc)
        self.held = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if not _is_stale(self.path, self.now):
                    raise LockHeld(f"another refresh is running (lock {self.path.name}); exiting")
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps({"pid": os.getpid(), "started_at": self.now.isoformat()}))
            self.held = True
            return self
        raise LockHeld("could not acquire the refresh lock")

    def __exit__(self, *exc):
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False
        return False


def _log_counts() -> dict:
    return {"forecasts": len(AppendOnlyLog(C.FORECAST_LOG_PATH).read()),
            "outcomes": len(AppendOnlyLog(C.OUTCOME_LOG_PATH).read()),
            "preview_forecasts": len(AppendOnlyLog(C.PREVIEW_LOG_PATH).read()),
            "preview_outcomes": len(AppendOnlyLog(C.PREVIEW_OUTCOME_LOG_PATH).read())}


def _gate_verdict() -> dict | None:
    vlog = load_json(C.VERSION_LOG_PATH, {"live": None, "history": []})
    last = (vlog.get("history") or [None])[-1]
    if last is None:
        return None
    gate = last.get("gate") or {}
    return {"version": last.get("version") or last.get("preview_version"), "status": last.get("status"),
            "passed": gate.get("passed"), "reason": gate.get("reason"), "profile": gate.get("profile"),
            "model_mae": gate.get("model_mae"), "best_baseline": gate.get("best_baseline"),
            "best_baseline_mae": gate.get("best_baseline_mae")}


def _check_layout() -> None:
    if C.DATA_LAYOUT == "bundle" or C.DATA_DIR.resolve() == C.SERVING_DIR.resolve():
        raise RefreshRefused("refresh writes to data/ and must run where the local data/ folder is "
                             "(not from the serving bundle). Run it on your own machine.")


def run_refresh(as_of: str | None = None, bundle_dir: Path | None = None, no_bundle: bool = False,
                now: datetime | None = None) -> dict:
    _check_layout()
    now = now or datetime.now(timezone.utc)
    t_all = time.perf_counter()
    record: dict = {"started_at": now.isoformat(), "requested_as_of": as_of, "data_layout": C.DATA_LAYOUT,
                    "data_dir": str(C.DATA_DIR), "status": "failed", "steps": {}}
    with RefreshLock(lock_path(), now=now):
        before = _log_counts()
        try:
            t = time.perf_counter()
            run = AgentLoop().run_once(as_of)
            record["steps"]["loop_s"] = round(time.perf_counter() - t, 2)
            record["run"] = {k: run.get(k) for k in ("as_of", "status", "feed", "reason", "decision",
                                                     "added_rows", "source_health", "live_version",
                                                     "preview_version", "n_forecasts", "n_abstained",
                                                     "outcomes_reconciled", "preview_outcomes_reconciled")}
            record["as_of"] = run.get("as_of")
            if run.get("status") in FAILURE_STATUSES or run.get("status") is None:
                raise RuntimeError(f"agent loop stopped: {run.get('status')} ({run.get('reason')})")
            record["gate_verdict"] = _gate_verdict()
            if run.get("retrain"):
                record["retrain"] = {k: run["retrain"].get(k) for k in ("status", "version", "preview_version",
                                                                        "gate_profile")}

            t = time.perf_counter()
            feed = export_bundle.active_feed()["feed"]
            record["feed"] = feed
            record["export"] = {"feed": feed, "rows": 0, "path": None}
            if feed in ("live", "preview"):
                rows, counts = export.export(feed, now=now)
                path = C.DATA_DIR / "handoff" / f"entity_forecast_{feed}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                export.write_csv(rows, str(path))
                record["export"] = {"feed": feed, "rows": len(rows), "path": str(path), **{
                    k: counts[k] for k in ("records", "not_ok", "not_in_taxonomy", "from_older_run")}}
            record["steps"]["export_s"] = round(time.perf_counter() - t, 2)

            if no_bundle:
                record["bundle"] = {"skipped": True}
            else:
                t = time.perf_counter()
                target = Path(bundle_dir) if bundle_dir is not None else C.SERVING_DIR
                manifest = export_bundle.build_bundle(target)
                record["bundle"] = {"dir": str(target), "manifest_sha256": export_bundle.manifest_sha256(target),
                                    "files": len(manifest["files"]), "total_bytes": manifest["total_bytes"],
                                    "feed": manifest["feed"], "validated": manifest["validated"],
                                    "as_of": manifest["as_of"]}
                record["steps"]["bundle_s"] = round(time.perf_counter() - t, 2)
            record["status"] = "ok"
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
            raise
        finally:
            after = _log_counts()
            record["log_records"] = {"before": before, "after": after,
                                     "added": {k: after[k] - before[k] for k in after}}
            record["steps"]["total_s"] = round(time.perf_counter() - t_all, 2)
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            save_json(last_refresh_path(), record)
    return record


def plan(as_of: str | None = None, bundle_dir: Path | None = None, no_bundle: bool = False) -> dict:
    _check_layout()
    registry = C.load_registry()
    target = C.target_source(registry)
    raw = C.RAW_DIR / f"{target}.csv"
    out = {"as_of_requested": as_of, "data_layout": C.DATA_LAYOUT, "data_dir": str(C.DATA_DIR),
           "target_source": target, "target_raw_present": raw.exists(),
           "active_feed_now": export_bundle.active_feed()["feed"],
           "lock_file": str(lock_path()), "lock_present": lock_path().exists(),
           "would_write": [str(last_refresh_path())],
           "bundle": None if no_bundle else str(Path(bundle_dir) if bundle_dir else C.SERVING_DIR)}
    if not no_bundle:
        out["would_write"].append(out["bundle"] + "/ (replaced atomically)")
    out["would_write"].append(str(C.DATA_DIR / "handoff") + "/entity_forecast_<feed>.csv")
    return out


def _summary_lines(record: dict) -> list[str]:
    run = record.get("run", {})
    gate = record.get("gate_verdict") or {}
    added = record.get("log_records", {}).get("added", {})
    lines = [
        f"refresh: status={record['status']} as_of={record.get('as_of')} feed={record.get('feed')} "
        f"layout={record.get('data_layout')}",
        f"loop: status={run.get('status')} decision={(run.get('decision') or {}).get('why')}",
        f"gate: {'passed' if gate.get('passed') else 'not passed'} ({gate.get('reason')}) "
        f"profile={gate.get('profile')} version={gate.get('version')}",
        f"outcomes reconciled: live={run.get('outcomes_reconciled')} preview={run.get('preview_outcomes_reconciled')}",
        f"records added: {added}",
    ]
    exp = record.get("export", {})
    lines.append(f"export: feed={exp.get('feed')} rows={exp.get('rows')} path={exp.get('path')}")
    bundle = record.get("bundle", {})
    if bundle.get("skipped"):
        lines.append("bundle: skipped (--no-bundle)")
    elif bundle:
        lines.append(f"bundle: dir={bundle.get('dir')} files={bundle.get('files')} total_bytes={bundle.get('total_bytes')}"
                     f" manifest_sha256={str(bundle.get('manifest_sha256'))[:16]}...")
    lines.append(f"timings: {record.get('steps')}")
    if record.get("error"):
        lines.append(f"error: {record['error']['type']}: {record['error']['message']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Refresh forecasts and write the serving bundle")
    ap.add_argument("--as-of", help="origin date (YYYY-MM-DD); default: the newest target period")
    ap.add_argument("--bundle", help="bundle directory (default: <repo>/serving)")
    ap.add_argument("--no-bundle", action="store_true", help="do not write the serving bundle")
    ap.add_argument("--dry-run", action="store_true", help="check inputs and print the plan; write nothing")
    a = ap.parse_args(argv)
    bundle_dir = Path(a.bundle) if a.bundle else None
    try:
        if a.dry_run:
            print(json.dumps(plan(a.as_of, bundle_dir, a.no_bundle), indent=2))
            return EXIT_OK
        record = run_refresh(a.as_of, bundle_dir, a.no_bundle)
    except (LockHeld, RefreshRefused) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED
    for line in _summary_lines(record):
        print(line)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
