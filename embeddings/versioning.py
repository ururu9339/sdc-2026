"""Step 10: models/version_log.json, shared with the forecaster.

The file is a list of entries, each tagged with a component ("embeddings", "forecaster", ...).
At most one entry per component has status "active"; recording a new one retires the old.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from embeddings import embedder

LOG = Path(__file__).resolve().parent.parent / "models" / "version_log.json"
COMPONENT = "embeddings"


def load():
    return json.loads(LOG.read_text()) if LOG.exists() else []


def active(component=COMPONENT):
    return next((e for e in reversed(load()) if e["component"] == component and e["status"] == "active"), None)


def record(entry, component=COMPONENT):
    log = load()
    for e in log:
        if e["component"] == component and e["status"] == "active":
            e["status"] = "retired"
    log.append({"component": component, "status": "active",
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry})
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text(json.dumps(log, indent=2) + "\n")


def gate(metrics, current=None, keys=("top1", "mrr")):
    """Failures if `metrics` scores below the active version on the frozen gold set; [] means it may replace it."""
    current = current or active()
    if not current:
        return []
    return [f"{k} {metrics[k]:.3f} < active {current['metrics'][k]:.3f} ({current['version_tag']})"
            for k in keys if metrics[k] < current["metrics"][k]]


def active_thresholds():
    """Thresholds picked by evaluate.py, only if they were picked for the settings in use now."""
    e = active()
    return e["thresholds"] if e and e["version_tag"] == embedder.version_tag() else {}
