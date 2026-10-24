from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.sources.common import (gap_periods, long_frame, period_rows, select_periods, source_cfg, to_period,
                                write_outputs)
from src.sources.taxonomy import job_of, load_vocabulary, skill_flags
from src.training import config as C
from src.training.config import ROOT, period_label

SOURCE_ID = "job_postings"
DEFAULT_RAW = ROOT / "src" / "data" / "job_descriptions.csv"
DATE_COL, TITLE_COL, ROLE_COL, SKILLS_COL = "Job Posting Date", "Job Title", "Role", "skills"
USECOLS = [DATE_COL, TITLE_COL, ROLE_COL, SKILLS_COL]
CHUNK_ROWS = 200_000
JOB_ID_COL = "Job Id"
DUP_KEY_COLS = [TITLE_COL, ROLE_COL, SKILLS_COL, "Company", DATE_COL]


def _tally(ch: pd.DataFrame, vocab, granularity: str) -> dict:
    period = to_period(ch[DATE_COL], granularity)
    keep = period.notna()
    ch, period = ch[keep], period[keep]
    fam = job_of(ch[ROLE_COL], vocab.job_aliases).fillna(job_of(ch[TITLE_COL], vocab.job_aliases))
    skills = skill_flags(ch[SKILLS_COL], vocab.skill_aliases)
    has_fam = fam.notna()
    return {
        "rows_used": len(ch),
        "totals": period.value_counts(),
        "skills": skills.groupby(period).sum(),
        "fams": pd.DataFrame({"period": period, "fam": fam}).dropna()
                  .groupby(["period", "fam"]).size().unstack(fill_value=0),
        "co": skills[has_fam].groupby(fam[has_fam]).sum(),
        "fam_tot": fam[has_fam].value_counts(),
    }


def _add(a, b):
    return b if a is None else a.add(b, fill_value=0)


def measure_duplicates(path: str | Path, chunk_rows: int = CHUNK_ROWS) -> dict:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    key_cols = [c for c in DUP_KEY_COLS if c in header]
    all_but_id = [c for c in header if c != JOB_ID_COL]
    h_key, h_all, n = [], [], 0
    for ch in pd.read_csv(path, dtype=str, chunksize=chunk_rows):
        n += len(ch)
        h_key.append(pd.util.hash_pandas_object(ch[key_cols], index=False).to_numpy(dtype=np.uint64))
        h_all.append(pd.util.hash_pandas_object(ch[all_but_id], index=False).to_numpy(dtype=np.uint64))
    if n == 0:
        return {"rows": 0, "key_columns": key_cols, "repeat_rows_key": 0, "share_repeat_rows_key": None,
                "repeat_rows_except_job_id": 0, "share_repeat_rows_except_job_id": None, "method": None}
    keys, alls = np.concatenate(h_key), np.concatenate(h_all)
    rep_key = int(n - len(np.unique(keys)))
    rep_all = int(n - len(np.unique(alls)))
    return {
        "rows": int(n),
        "key_columns": key_cols,
        "key_columns_absent": [c for c in DUP_KEY_COLS if c not in header],
        "repeat_rows_key": rep_key,
        "share_repeat_rows_key": rep_key / n if n else None,
        "repeat_rows_except_job_id": rep_all,
        "share_repeat_rows_except_job_id": rep_all / n if n else None,
        "method": "a row counts as a repeat when an earlier row has the same key (64-bit hash); rows are not removed",
    }


def build(path: str | Path | None = None, out: Path | None = None, graph_out: Path | None = None,
          cfg: dict | None = None, chunk_rows: int = CHUNK_ROWS) -> Path:
    path = Path(path or DEFAULT_RAW)
    cfg = cfg or source_cfg(SOURCE_ID)
    source_id, gran = cfg["source_id"], cfg["granularity"]
    out = Path(out or C.RAW_DIR / f"{source_id}.csv")
    graph_out = Path(graph_out or C.PROCESSED_DIR / "job_skill_graph.csv")
    vocab = load_vocabulary()

    acc: dict = {k: None for k in ("totals", "skills", "fams", "co", "fam_tot")}
    rows_read = rows_used = 0
    for ch in pd.read_csv(path, usecols=USECOLS, dtype=str, chunksize=chunk_rows):
        rows_read += len(ch)
        part = _tally(ch, vocab, gran)
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

    dups = measure_duplicates(path)
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["posting_duplicates"] = dups
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"posting duplicates: key {dups['repeat_rows_key']:,} rows ({dups['share_repeat_rows_key']:.2%}), "
          f"all columns but Job Id {dups['repeat_rows_except_job_id']:,} rows "
          f"({dups['share_repeat_rows_except_job_id']:.2%}) of {dups['rows']:,}; totals unchanged")

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
