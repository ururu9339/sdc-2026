from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.sources.common import long_frame, period_rows, select_periods, source_cfg, write_outputs
from src.sources.taxonomy import job_of, load_vocabulary
from src.training import config as C
from src.training.config import GRANULARITY_FREQ, ROOT, to_period_start

SOURCE_ID = "market_signal"
DEFAULT_RAW = ROOT / "src" / "data" / "job_market_dataset.csv"
USECOLS = ["year", "month", "occupation"]


def build_observations(raw: pd.DataFrame, cfg: dict | None = None, vocab=None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or source_cfg(SOURCE_ID)
    vocab = vocab if vocab is not None else load_vocabulary()
    gran, source_id, min_entity = cfg["granularity"], cfg["source_id"], cfg["min_entity_total"]
    first_of_month = pd.to_datetime(pd.DataFrame({"year": raw["year"], "month": raw["month"], "day": 1}))
    period = to_period_start(first_of_month, GRANULARITY_FREQ[gran])
    family = job_of(raw["occupation"], vocab.job_aliases)
    totals_all = period.value_counts()
    grid, dropped = select_periods(totals_all, cfg["min_period_postings"], gran, cfg["dense"])
    totals = totals_all.reindex(grid)
    fam_cnt = (pd.DataFrame({"period": period, "fam": family}).dropna()
               .groupby(["period", "fam"]).size().unstack(fill_value=0).reindex(grid).fillna(0).astype(int))
    fam_cnt = fam_cnt.loc[:, fam_cnt.sum() >= min_entity]
    obs = long_frame(fam_cnt, totals, "job", source_id)
    meta = {
        "granularity": gran,
        "rows_used": int(len(raw)),
        "first_period": str(grid[0].date()),
        "last_period": str(grid[-1].date()),
        "periods": period_rows(totals),
        "dropped_periods": dropped,
        "min_period_postings": cfg["min_period_postings"],
        "min_entity_total": min_entity,
        "kept_entities": {"job": int(fam_cnt.shape[1])},
        "unmatched_occupations": sorted(raw.loc[family.isna(), "occupation"].unique().tolist()),
    }
    return obs, meta


def build(path: str | Path | None = None, out: Path | None = None, cfg: dict | None = None) -> Path:
    path = Path(path or DEFAULT_RAW)
    cfg = cfg or source_cfg(SOURCE_ID)
    out = Path(out or C.RAW_DIR / f"{cfg['source_id']}.csv")
    raw = pd.read_csv(path, usecols=USECOLS)
    obs, meta = build_observations(raw, cfg)
    meta = {"source_id": cfg["source_id"], "raw_file": str(path), "rows_read": len(raw), **meta}
    write_outputs(obs, meta, out)
    print(f"wrote {out}: {len(obs):,} rows, {meta['kept_entities']} entities, {len(meta['periods'])} periods")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
