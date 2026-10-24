"""Adapter: job_descriptions.csv (team-provided) -> the target series (registry entry with adapter = this module).

    python3 -m src.sources.job_descriptions [path/to/job_descriptions.csv]

Streams the ~1.7 GB file in chunks, reading only the four columns it needs, and aggregates to the
source's granularity (weekly, Monday start). Writes:
  data/raw/<source_id>.csv            long format, one row per (entity, period), share of that period's postings
  data/raw/<source_id>.meta.json      coverage, thresholds, per-period totals, dropped periods
  data/processed/job_skill_graph.csv  family x skill co-occurrence, for explanation only (never for trends)

Provenance: provided by the team, apparently synthetic (period shares are indistinguishable from
sampling noise). License unverified. Demo/training use only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.sources.common import (gap_periods, long_frame, period_rows, select_periods, source_cfg, to_period,
                                write_outputs)
from src.sources.taxonomy import family_of, load_lexicon, skill_flags
from src.training import config as C
from src.training.config import ROOT, period_label

SOURCE_ID = "job_postings"
DEFAULT_RAW = ROOT / "src" / "data" / "job_descriptions.csv"
DATE_COL, TITLE_COL, ROLE_COL, SKILLS_COL = "Job Posting Date", "Job Title", "Role", "skills"
USECOLS = [DATE_COL, TITLE_COL, ROLE_COL, SKILLS_COL]
CHUNK_ROWS = 200_000


def _tally(ch: pd.DataFrame, lexicon: dict, granularity: str) -> dict:
    """Counts for one chunk. Every quantity is a sum, so chunks add up exactly."""
    period = to_period(ch[DATE_COL], granularity)
    keep = period.notna()
    ch, period = ch[keep], period[keep]
    fam = family_of(ch[ROLE_COL]).fillna(family_of(ch[TITLE_COL]))  # Role first, then Job Title
    skills = skill_flags(ch[SKILLS_COL], lexicon)
    has_fam = fam.notna()
    return {
        "rows_used": len(ch),
        "totals": period.value_counts(),  # every dated row counts, matched or not
        "skills": skills.groupby(period).sum(),
        "fams": pd.DataFrame({"period": period, "fam": fam}).dropna()
                  .groupby(["period", "fam"]).size().unstack(fill_value=0),
        "co": skills[has_fam].groupby(fam[has_fam]).sum(),  # family x skill: postings with both
        "fam_tot": fam[has_fam].value_counts(),
    }


def _add(a, b):
    return b if a is None else a.add(b, fill_value=0)


def build(path: str | Path | None = None, out: Path | None = None, graph_out: Path | None = None,
          cfg: dict | None = None, chunk_rows: int = CHUNK_ROWS) -> Path:
    path = Path(path or DEFAULT_RAW)
    cfg = cfg or source_cfg(SOURCE_ID)
    source_id, gran = cfg["source_id"], cfg["granularity"]
    out = Path(out or C.RAW_DIR / f"{source_id}.csv")
    graph_out = Path(graph_out or C.PROCESSED_DIR / "job_skill_graph.csv")
    lexicon = load_lexicon()

    acc: dict = {k: None for k in ("totals", "skills", "fams", "co", "fam_tot")}
    rows_read = rows_used = 0
    for ch in pd.read_csv(path, usecols=USECOLS, dtype=str, chunksize=chunk_rows):
        rows_read += len(ch)
        part = _tally(ch, lexicon, gran)
        rows_used += part.pop("rows_used")
        for k, v in part.items():
            acc[k] = _add(acc[k], v)

    totals = acc["totals"].astype(int).sort_index()
    grid, dropped = select_periods(totals, cfg["min_period_postings"], gran, cfg["dense"])
    totals = totals.reindex(grid)
    min_entity = cfg["min_entity_total"]
    skill_cnt = acc["skills"].reindex(grid).fillna(0).astype(int)
    fam_cnt = acc["fams"].reindex(grid).fillna(0).astype(int)
    skill_cnt = skill_cnt.loc[:, skill_cnt.sum() >= min_entity]
    fam_cnt = fam_cnt.loc[:, fam_cnt.sum() >= min_entity]

    obs = pd.concat([long_frame(skill_cnt, totals, "skill", source_id),
                     long_frame(fam_cnt, totals, "job", source_id)], ignore_index=True)
    write_outputs(obs, {
        "source_id": source_id,
        "raw_file": str(path),
        "granularity": gran,
        "rows_read": rows_read,
        "rows_used": rows_used,
        "first_period": period_label(grid[0]),
        "last_period": period_label(grid[-1]),
        "periods": period_rows(totals),
        "dropped_periods": dropped,
        "gap_periods": gap_periods(grid, gran),
        "min_period_postings": cfg["min_period_postings"],
        "min_entity_total": min_entity,
        "kept_entities": {"skill": skill_cnt.shape[1], "job": fam_cnt.shape[1]},
    }, out)

    co = acc["co"].fillna(0).astype(int)
    edges = co.rename_axis(index="job", columns="skill").stack().rename("n_cooccur").reset_index()
    edges = edges[edges["n_cooccur"] > 0].copy()
    edges["n_job_postings"] = edges["job"].map(acc["fam_tot"]).astype(int)
    edges["weight"] = edges["n_cooccur"] / edges["n_job_postings"]
    graph_out.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(graph_out, index=False)

    print(f"wrote {out}: {len(obs):,} rows, {skill_cnt.shape[1]} skills, {fam_cnt.shape[1]} job families, "
          f"{len(grid)} {gran} periods ({grid[0].date()}..{grid[-1].date()}), {len(dropped)} dropped")
    print(f"wrote {graph_out}: {len(edges):,} job-skill edges")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
