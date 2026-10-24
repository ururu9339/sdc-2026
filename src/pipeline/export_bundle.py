"""Write the serving bundle: the committed snapshot the API serves on a deployment with no local data/ folder.

    python3 -m src.pipeline.export_bundle [--out DIR]

The bundle holds ONLY what the API reads, laid out so that SDC_DATA_DIR=<bundle> and SDC_MODELS_DIR=<bundle>/models
work (and so that resolve_layout picks it up when no data/ exists):

    manifest.json                     what is in the bundle, sha256 and sizes, row counts, feed, validated flag
    forecast_log.jsonl                LIVE feed only: the newest as_of records, re-chained (hash chain verifies)
    preview_log.jsonl                 PREVIEW feed only: the newest as_of records, re-chained
    processed/job_skill_graph.csv     the measured job-skill graph (explanations, offline matcher)
    processed/observations.csv        target-source rows, plus rows of every enabled leading source that has data
    raw/<source>.csv                  the best-tier leading source's raw file (for /snapshot), nothing else
    model_report.json                 model report (timeline and audit summary)
    models/version_log.json           the last KEEP_VERSIONS attempts, plus the live and preview entries
    models/forecaster_<live>/meta.json  only when a live model exists

It never contains src/data, CVs, the candidates database, models' weights or secrets. The bundle is a SNAPSHOT:
it is written from data/ by src.pipeline.refresh, and the API serves it without recomputing anything. If the total
exceeds MAX_BYTES, the export fails and the previous bundle is left in place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.pipeline.store import AppendOnlyLog, load_json, save_json, write_chain
from src.training import config as C

MAX_BYTES = 5 * 1024 * 1024  # hard limit for everything in serving/, manifest included
KEEP_VERSIONS = 5
NOTE = ("snapshot for serving: the files the API reads, written by src.pipeline.refresh from the local data/ folder. "
        "Not a full copy of the data and not a re-computation; regenerate it to refresh.")


class BundleError(RuntimeError):
    pass


def _line_count(path: Path) -> int:
    with open(path) as fh:
        return sum(1 for _ in fh)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_feed() -> dict:
    """Which feed the API would serve from the local state: live, preview or none."""
    vlog = load_json(C.VERSION_LOG_PATH, {"live": None, "history": []})
    if vlog.get("live"):
        return {"feed": "live", "log": C.FORECAST_LOG_PATH, "outcomes": C.OUTCOME_LOG_PATH,
                "model_version": vlog["live"], "name": "forecast_log.jsonl"}
    if C.PREVIEW_ENABLED and AppendOnlyLog(C.PREVIEW_LOG_PATH).read():
        # the preview model directory is named preview_v<N>; the record label is preview:v<N>
        model_dir = vlog.get("preview") or ""
        label = f"preview:{model_dir.removeprefix('preview_')}" if model_dir else None
        return {"feed": "preview", "log": C.PREVIEW_LOG_PATH, "outcomes": C.PREVIEW_OUTCOME_LOG_PATH,
                "model_version": label, "name": "preview_log.jsonl"}
    return {"feed": "none", "log": None, "outcomes": None, "model_version": None, "name": None}


def newest_records(records: list[dict]) -> list[dict]:
    if not records:
        return []
    newest = max(r["as_of"] for r in records)
    return [r for r in records if r["as_of"] == newest]


def _outcome_summary(path: Path | None, feed: str) -> dict:
    rows = AppendOnlyLog(path).read() if path is not None else []
    if not rows:
        return {"feed": feed, "validated": feed == "live", "matured": 0}
    return {"feed": feed, "validated": feed == "live", "matured": len(rows),
            "live_mae": float(sum(r["abs_err"] for r in rows) / len(rows)),
            "persistence_mae": float(sum(r["abs_err_persistence"] for r in rows) / len(rows)),
            "interval_coverage": float(sum(bool(r["in_interval"]) for r in rows) / len(rows))}


def leading_sources_with_raw(registry: dict) -> list[dict]:
    """The leading sources the snapshot can use: enabled, with a non-empty raw file, best tier only (the same
    rule as forecast_store.snapshot)."""
    present = []
    for s in C.leading_sources(registry):
        raw = C.RAW_DIR / f"{s['source_id']}.csv"
        if raw.exists() and raw.stat().st_size > 0 and _line_count(raw) > 1:
            present.append(s)
    if not present:
        return []
    best = min(s.get("tier", 99) for s in present)
    return [s for s in present if s.get("tier", 99) == best]


def _version_log_trimmed(live: str | None, preview_dir: str | None) -> dict:
    vlog = load_json(C.VERSION_LOG_PATH, {"live": None, "history": []})
    history = vlog.get("history") or []
    keep = set(range(max(0, len(history) - KEEP_VERSIONS), len(history)))
    for i, h in enumerate(history):
        if live and h.get("version") == live:
            keep.add(i)
        if preview_dir and h.get("model_dir") == preview_dir:
            keep.add(i)
    trimmed = {k: v for k, v in vlog.items() if k != "history"}
    trimmed["history"] = [history[i] for i in sorted(keep)]
    trimmed["history_total"] = len(history)
    trimmed["history_kept"] = len(trimmed["history"])
    return trimmed


def _write_tree(stage: Path, now: datetime) -> dict:
    """Write every bundle file into `stage` (an empty directory). Returns the facts for the manifest."""
    registry = C.load_registry()
    active = active_feed()
    facts: dict = {"feed": active["feed"], "model_version": active["model_version"], "row_counts": {}}
    vlog = load_json(C.VERSION_LOG_PATH, {"live": None, "history": []})
    live = vlog.get("live")
    preview_dir = vlog.get("preview") if not live else None

    if active["feed"] != "none":
        records = newest_records(AppendOnlyLog(active["log"]).read())
        if not AppendOnlyLog(active["log"]).verify():
            raise BundleError(f"{active['name']} fails its hash chain; refusing to export it")
        write_chain(stage / active["name"], records)
        facts["row_counts"]["forecasts"] = len(records)
        facts["as_of"] = records[0]["as_of"] if records else None
        facts["gate_profile"] = records[0].get("gate_profile") if records else None
        facts["n_outcomes"] = _outcome_summary(active["outcomes"], active["feed"])
    else:
        facts["as_of"] = None
        facts["gate_profile"] = None
        facts["n_outcomes"] = {"feed": "none", "validated": False, "matured": 0}

    graph = C.PROCESSED_DIR / "job_skill_graph.csv"
    if graph.exists():
        (stage / "processed").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(graph, stage / "processed" / "job_skill_graph.csv")
        facts["row_counts"]["graph_edges"] = max(0, _line_count(graph) - 1)

    obs_path = C.OBSERVATIONS_PATH
    if obs_path.exists():
        obs = pd.read_csv(obs_path)
        keep_sources = {C.target_source(registry)} | {s["source_id"] for s in C.leading_sources(registry)
                                                     if (obs["source_id"] == s["source_id"]).any()}
        kept = obs[obs["source_id"].isin(keep_sources)]
        (stage / "processed").mkdir(parents=True, exist_ok=True)
        kept.to_csv(stage / "processed" / "observations.csv", index=False)
        facts["row_counts"]["observations"] = int(len(kept))
        facts["observation_sources"] = sorted(keep_sources)

    raw_rows = {}
    for s in leading_sources_with_raw(registry):
        sid = s["source_id"]
        (stage / "raw").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.RAW_DIR / f"{sid}.csv", stage / "raw" / f"{sid}.csv")
        raw_rows[sid] = max(0, _line_count(C.RAW_DIR / f"{sid}.csv") - 1)
    facts["row_counts"]["raw_rows"] = raw_rows

    if C.REPORT_PATH.exists():
        shutil.copyfile(C.REPORT_PATH, stage / "model_report.json")

    (stage / "models").mkdir(parents=True, exist_ok=True)
    save_json(stage / "models" / "version_log.json", _version_log_trimmed(live, preview_dir))
    if live:
        meta = C.MODELS_DIR / f"forecaster_{live}" / "meta.json"
        if meta.exists():
            (stage / "models" / f"forecaster_{live}").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(meta, stage / "models" / f"forecaster_{live}" / "meta.json")
    last = (vlog.get("history") or [None])[-1]
    facts["gate"] = {"passed": (last or {}).get("gate", {}).get("passed") if last else None,
                     "reason": (last or {}).get("gate", {}).get("reason") if last else None}
    facts["validated"] = active["feed"] == "live"
    return facts


def _files(stage: Path) -> list[Path]:
    return sorted(p for p in stage.rglob("*") if p.is_file() and p.name != "manifest.json")


def build_bundle(out_dir: Path | None = None, *, now: datetime | None = None, max_bytes: int = MAX_BYTES) -> dict:
    """Write the bundle to `out_dir` (default <repo>/serving) and return its manifest. Fails, leaving the previous
    bundle untouched, when the result is larger than max_bytes or a chain does not verify."""
    now = now or datetime.now(timezone.utc)
    out_dir = Path(out_dir) if out_dir is not None else C.SERVING_DIR
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".serving-", dir=out_dir.parent))
    try:
        facts = _write_tree(stage, now)
        files = _files(stage)
        entries = [{"path": p.relative_to(stage).as_posix(), "bytes": p.stat().st_size, "sha256": _sha256(p)}
                   for p in files]
        total = sum(e["bytes"] for e in entries)
        manifest = {
            "generated_at": now.isoformat(),
            "as_of": facts.get("as_of"),
            "feed": facts["feed"],
            "model_version": facts["model_version"],
            "validated": facts["validated"],
            "gate_profile": facts.get("gate_profile"),
            "gate": facts["gate"],
            "row_counts": facts["row_counts"],
            "outcomes": facts["n_outcomes"],
            "observation_sources": facts.get("observation_sources", []),
            "files": entries,
            "total_bytes": total,
            "max_total_bytes": max_bytes,
            "note": NOTE,
        }
        manifest["total_bytes"] = total
        for _ in range(10):  # the manifest counts its own bytes; iterate until the figure is stable
            text = json.dumps(manifest, indent=2, default=str) + "\n"
            if manifest["total_bytes"] == total + len(text.encode()):
                break
            manifest["total_bytes"] = total + len(text.encode())
        text = json.dumps(manifest, indent=2, default=str) + "\n"
        (stage / "manifest.json").write_text(text)
        stage.chmod(0o755)  # mkdtemp creates 0700; the committed directory should be ordinary
        actual = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
        if actual > max_bytes:
            raise BundleError(f"serving bundle is {actual} bytes, over the {max_bytes}-byte limit; "
                              "nothing was written")
        if out_dir.exists():
            old = out_dir.with_name(out_dir.name + ".previous")
            if old.exists():
                shutil.rmtree(old)
            out_dir.rename(old)
            stage.rename(out_dir)
            shutil.rmtree(old)
        else:
            stage.rename(out_dir)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def manifest_sha256(out_dir: Path | None = None) -> str | None:
    path = (Path(out_dir) if out_dir is not None else C.SERVING_DIR) / "manifest.json"
    return _sha256(path) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="directory to write (default: <repo>/serving)")
    a = ap.parse_args(argv)
    try:
        m = build_bundle(Path(a.out) if a.out else None)
    except BundleError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    print(f"wrote bundle: feed={m['feed']} validated={m['validated']} files={len(m['files'])} "
          f"total_bytes={m['total_bytes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
