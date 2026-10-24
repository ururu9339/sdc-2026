"""Accumulating observation table + tamper-evident append-only logs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.sources.loaders import COLUMNS
from src.training import config as C

KEY = ["entity_id", "period", "source_id"]


class ObservationStore:
    def __init__(self, path: Path | None = None):
        self.path = path or C.OBSERVATIONS_PATH

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        return pd.read_csv(self.path, parse_dates=["period"])

    def upsert(self, new: pd.DataFrame) -> int:
        """Append rows, latest value wins per (entity, period, source). Returns rows added."""
        old = self.load()
        if new.empty:
            return 0
        merged = pd.concat([old, new[COLUMNS]], ignore_index=True)
        merged["period"] = pd.to_datetime(merged["period"])
        merged = merged.drop_duplicates(KEY, keep="last").sort_values(KEY)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(self.path, index=False)
        return len(merged) - len(old)


class AppendOnlyLog:
    """JSONL log; each record carries the hash of the previous one, so edits are detectable."""

    def __init__(self, path: Path):
        self.path = path

    def _last_hash(self) -> str:
        if not self.path.exists():
            return "genesis"
        last = None
        with open(self.path) as f:
            for last in f:
                pass
        return json.loads(last)["hash"] if last else "genesis"

    def append_many(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prev = self._last_hash()
        with open(self.path, "a") as f:
            for rec in records:
                rec = {**rec, "logged_at": datetime.now(timezone.utc).isoformat(), "prev_hash": prev}
                rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
                f.write(json.dumps(rec, default=str) + "\n")
                prev = rec["hash"]

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def verify(self) -> bool:
        prev = "genesis"
        for rec in self.read():
            body = {k: v for k, v in rec.items() if k != "hash"}
            if body["prev_hash"] != prev or hashlib.sha256(
                    json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != rec["hash"]:
                return False
            prev = rec["hash"]
        return True


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
