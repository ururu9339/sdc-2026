from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.training import config as C

RAW_DIR = None

MAPPING_DIR = Path(__file__).with_name("mappings")
GLOBAL_ID = "__GLOBAL__"

COLUMNS = ["entity_id", "kind", "period", "source_id", "value", "support"]


class CsvLoader:

    def __init__(self, source_id: str, path: str | None = None, scope: str = "entity", **_):
        self.source_id, self.scope = source_id, scope
        self.path = (RAW_DIR or C.RAW_DIR) / (path or f"{source_id}.csv")

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
            mapping = pd.read_csv(MAPPING_DIR / f"{self.source_id}.csv")
            df = df.drop(columns=["kind"]).rename(columns={"entity_id": "key"}).merge(mapping, on="key")[COLUMNS]
        return df if as_of is None else df[df["period"] <= as_of].copy()


LOADERS = {"csv": CsvLoader}


def get_loader(source: dict):
    cfg = dict(source["loader"])
    return LOADERS[cfg.pop("type")](source["source_id"], scope=source.get("scope", "entity"), **cfg)
