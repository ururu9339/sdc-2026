from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.sources.loaders import COLUMNS
from src.training import config as C


def source_cfg(source_id: str, registry: dict | None = None) -> dict:
    registry = registry or C.load_registry()
    return next(s for s in registry["sources"] if s["source_id"] == source_id)


def to_period(dates: pd.Series, granularity: str) -> pd.Series:
    day = pd.to_datetime(dates.astype("string").str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    return C.to_period_start(day, C.GRANULARITY_FREQ[granularity])


def select_periods(totals: pd.Series, min_postings: int, granularity: str, dense: bool) -> tuple[pd.DatetimeIndex, list[dict]]:
    totals = totals.sort_index()
    kept = totals[totals >= min_postings]
    dropped = [{"period": C.period_label(p), "total_postings": int(n), "reason": f"fewer than {min_postings} postings"}
               for p, n in totals[totals < min_postings].items()]
    if kept.empty:
        return pd.DatetimeIndex([]), dropped
    if dense:
        grid = pd.period_range(kept.index.min(), kept.index.max(), freq=C.GRANULARITY_FREQ[granularity]).start_time
        missing = grid.difference(kept.index)
        if len(missing):
            raise ValueError(f"{len(missing)} {granularity} periods inside the dense range have no kept postings "
                             f"(first: {C.period_label(missing[0])}); zero-fill is not defined for them")
        return grid, dropped
    return pd.DatetimeIndex(kept.index), dropped


def gap_periods(kept_index: pd.DatetimeIndex, granularity: str) -> list[str]:
    span = pd.period_range(kept_index.min(), kept_index.max(), freq=C.GRANULARITY_FREQ[granularity]).start_time
    return [C.period_label(p) for p in span.difference(kept_index)]


def long_frame(counts: pd.DataFrame, totals: pd.Series, kind: str, source_id: str) -> pd.DataFrame:
    if counts.shape[1] == 0:
        return pd.DataFrame(columns=COLUMNS)
    counts = counts.reindex(totals.index).fillna(0)
    long = counts.rename_axis(index="period", columns="entity_id").stack().rename("support").reset_index()
    long["kind"] = kind
    long["value"] = long["support"].to_numpy(dtype=float) / totals.reindex(long["period"]).to_numpy(dtype=float)
    long["source_id"] = source_id
    return long[COLUMNS]


def period_rows(totals: pd.Series) -> list[dict]:
    return [{"period": C.period_label(p), "total_postings": int(n)} for p, n in totals.items()]


def write_outputs(obs: pd.DataFrame, meta: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    obs.to_csv(out, index=False)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=_jsonable) + "\n")
    return out


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)
