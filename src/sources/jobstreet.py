from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.sources.common import gap_periods, long_frame, period_rows, select_periods, source_cfg, to_period, write_outputs
from src.sources.taxonomy import job_of, load_vocabulary, skill_flags
from src.training import config as C
from src.training.config import ROOT, period_label

SOURCE_ID = "jobstreet_my"
DEFAULT_RAW = ROOT / "src" / "data" / "jobstreet_all_job_dataset.csv"
USECOLS = ["job_id", "job_title", "descriptions", "role", "listingDate"]
DATE_COL = "listingDate"


def prepare(raw: pd.DataFrame, granularity: str) -> pd.DataFrame:
    df = raw.drop_duplicates("job_id").copy()
    df["period"] = to_period(df[DATE_COL], granularity)
    return df[df["period"].notna()].copy()


def build_observations(df: pd.DataFrame, cfg: dict | None = None, vocab=None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or source_cfg(SOURCE_ID)
    gran, source_id = cfg["granularity"], cfg["source_id"]
    vocab = vocab if vocab is not None else load_vocabulary()
    totals_all = df.groupby("period").size()
    grid, dropped = select_periods(totals_all, cfg["min_period_postings"], gran, cfg["dense"])
    df = df[df["period"].isin(grid)]
    totals = totals_all.reindex(grid).astype(int)
    min_entity = cfg["min_entity_total"]

    fam = job_of(df["role"], vocab.job_aliases).fillna(job_of(df["job_title"], vocab.job_aliases))
    text = df["job_title"].fillna("") + " " + df["descriptions"].fillna("")
    skill_cnt = skill_flags(text, vocab.skill_aliases).groupby(df["period"]).sum().reindex(grid).fillna(0).astype(int)
    fam_cnt = (pd.DataFrame({"period": df["period"], "fam": fam}).dropna()
               .groupby(["period", "fam"]).size().unstack(fill_value=0).reindex(grid).fillna(0).astype(int))
    skill_cnt = skill_cnt.loc[:, skill_cnt.sum() >= min_entity]
    fam_cnt = fam_cnt.loc[:, fam_cnt.sum() >= min_entity]

    obs = pd.concat([long_frame(skill_cnt, totals, "skill", source_id),
                     long_frame(fam_cnt, totals, "job", source_id)], ignore_index=True)
    meta = {
        "granularity": gran,
        "rows_dated_deduped": int(totals_all.sum()),
        "rows_used": int(totals.sum()),
        "first_period": period_label(grid[0]) if len(grid) else None,
        "last_period": period_label(grid[-1]) if len(grid) else None,
        "periods": period_rows(totals),
        "dropped_periods": dropped,
        "gap_periods": gap_periods(grid, gran) if len(grid) else [],
        "min_period_postings": cfg["min_period_postings"],
        "min_entity_total": min_entity,
        "kept_entities": {"skill": int(skill_cnt.shape[1]), "job": int(fam_cnt.shape[1])},
    }
    return obs, meta


def build(path: str | Path | None = None, out: Path | None = None, cfg: dict | None = None) -> Path:
    path = Path(path or DEFAULT_RAW)
    cfg = cfg or source_cfg(SOURCE_ID)
    out = Path(out or C.RAW_DIR / f"{cfg['source_id']}.csv")
    raw = pd.read_csv(path, usecols=USECOLS, dtype=str)
    obs, meta = build_observations(prepare(raw, cfg["granularity"]), cfg)
    meta = {"source_id": cfg["source_id"], "raw_file": str(path), "rows_read": len(raw), **meta}
    write_outputs(obs, meta, out)
    print(f"wrote {out}: {len(obs):,} rows, {meta['kept_entities']} entities, {len(meta['periods'])} periods, "
          f"{len(meta['dropped_periods'])} dropped")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
