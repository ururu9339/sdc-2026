"""Source loaders. Every loader returns the long format the pipeline stores:

    entity_id, kind ('skill'|'job'), period (month start), source_id, value, support

`support` is the raw sample size behind a value (e.g. mention count); NaN if not applicable.
Real sources plug in by dropping a long-format CSV in data/raw/<source_id>.csv, or by adding a
loader type to LOADERS and pointing the registry entry at it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.training.config import RAW_DIR

MAPPING_DIR = Path(__file__).with_name("mappings")
GLOBAL_ID = "__GLOBAL__"

COLUMNS = ["entity_id", "kind", "period", "source_id", "value", "support"]


class CsvLoader:
    """Reads data/raw/<source_id>.csv in the long format.

    scope='entity' (default): rows already per skill/job.
    scope='mapped': rows are per source key (e.g. a sector or occupation code, in `entity_id`);
        src/sources/mappings/<source_id>.csv (entity_id, kind, key) fans each key out to its entities.
    scope='global': rows are economy-wide (any `entity_id`); stored under GLOBAL_ID and broadcast
        to every entity when features are built.
    """

    def __init__(self, source_id: str, path: str | None = None, scope: str = "entity", **_):
        self.source_id, self.scope = source_id, scope
        self.path = RAW_DIR / (path or f"{source_id}.csv")

    def fetch(self, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        df = pd.read_csv(self.path, parse_dates=["period"])
        if "support" not in df:
            df["support"] = np.nan
        df = df[COLUMNS]
        if self.scope == "global":
            df = df.assign(entity_id=GLOBAL_ID, kind="global")
        elif self.scope == "mapped":
            mapping = pd.read_csv(MAPPING_DIR / f"{self.source_id}.csv")  # entity_id, kind, key
            df = df.drop(columns=["kind"]).rename(columns={"entity_id": "key"}).merge(mapping, on="key")[COLUMNS]
        return df if as_of is None else df[df["period"] <= as_of].copy()


LOADERS = {"csv": CsvLoader}


def get_loader(source: dict):
    cfg = dict(source["loader"])
    return LOADERS[cfg.pop("type")](source["source_id"], scope=source.get("scope", "entity"), **cfg)
