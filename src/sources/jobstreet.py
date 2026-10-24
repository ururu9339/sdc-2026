"""Adapter: JobStreet listings (azrai99/job-dataset, CC BY 4.0) -> long-format job_postings.csv.

    python -m src.sources.jobstreet inspect  path/to/jobs.csv
    python -m src.sources.jobstreet build    path/to/jobs.csv

`build` writes data/raw/job_postings.csv with one row per (entity, month): the share of that
month's postings mentioning the skill (keyword lexicon) or holding the role (job title/role).
`support` is the raw mention count. Entity extraction is a keyword fallback; swap `extract_skills`
for the Qwen extractor when it exists.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.config import RAW_DIR

LEXICON_PATH = Path(__file__).with_name("skills_lexicon.json")
DATE_COL = "listingDate"
MIN_ROLE_POSTINGS = 30  # a role needs this many postings overall to become a 'job' entity
MAX_ROLES = 150
MIN_SKILL_POSTINGS = 30


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"job_id", "job_title", DATE_COL} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    df = df.drop_duplicates("job_id").copy()
    df["listed"] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["listed"])
    df["period"] = df["listed"].dt.to_period("M").dt.to_timestamp()
    return df


def inspect(path: str | Path) -> None:
    df = load_raw(path)
    print(f"rows (deduped, dated): {len(df):,}")
    print(f"date range: {df['listed'].min()} -> {df['listed'].max()}")
    per = df.groupby("period").size()
    print(f"distinct months: {len(per)}  (plan wants 20+)")
    print(per.to_string())
    print("columns:", list(df.columns))
    for col in ("category", "role", "location"):
        if col in df:
            print(f"top {col}:", df[col].value_counts().head(8).to_dict())


def normalise_role(s: pd.Series) -> pd.Series:
    return (s.fillna("").str.lower().str.replace(r"[^a-z0-9 /+#.-]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def extract_skills(text: pd.Series, lexicon: dict[str, list[str]]) -> pd.DataFrame:
    """Boolean frame, one column per canonical skill. Keyword fallback for the Qwen extractor."""
    low = text.fillna("").str.lower()
    out = {}
    for skill, aliases in lexicon.items():
        if skill.startswith("_"):
            continue
        pat = "|".join(f"(?<![a-z0-9]){a}(?![a-z0-9])" for a in aliases)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # alias groups are intentional
            out[skill] = low.str.contains(re.compile(pat), regex=True)
    return pd.DataFrame(out, index=text.index)


def build_observations(df: pd.DataFrame, lexicon: dict | None = None) -> pd.DataFrame:
    lexicon = lexicon or json.loads(LEXICON_PATH.read_text())
    total = df.groupby("period").size()
    months = total.index

    text = df["job_title"].fillna("") + " " + df.get("descriptions", pd.Series("", index=df.index)).fillna("")
    skills = extract_skills(text, lexicon)
    skills["period"] = df["period"]
    skill_counts = skills.groupby("period").sum().reindex(months, fill_value=0)
    skill_counts = skill_counts.loc[:, skill_counts.sum() >= MIN_SKILL_POSTINGS]

    role_col = normalise_role(df["role"] if "role" in df else df["job_title"])
    top_roles = role_col[role_col != ""].value_counts()
    top_roles = top_roles[top_roles >= MIN_ROLE_POSTINGS].head(MAX_ROLES).index
    jr = pd.DataFrame({"period": df["period"], "role": role_col})
    jr = jr[jr["role"].isin(top_roles)]
    role_counts = jr.groupby(["period", "role"]).size().unstack(fill_value=0).reindex(months, fill_value=0)

    frames = []
    for counts, kind in ((skill_counts, "skill"), (role_counts, "job")):
        long = counts.rename_axis(index="period", columns="entity_id").stack().rename("support").reset_index()
        long["kind"] = kind
        frames.append(long)
    obs = pd.concat(frames, ignore_index=True)
    obs["value"] = obs["support"].to_numpy() / total.reindex(obs["period"]).to_numpy()
    obs["source_id"] = "job_postings"
    return obs[["entity_id", "kind", "period", "source_id", "value", "support"]]


def build(path: str | Path, out: Path | None = None) -> Path:
    out = out or RAW_DIR / "job_postings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    obs = build_observations(load_raw(path))
    obs.to_csv(out, index=False)
    n_months = obs["period"].nunique()
    print(f"wrote {out}: {len(obs):,} rows, {obs['entity_id'].nunique()} entities, {n_months} months")
    if n_months < 20:
        print(f"WARNING: only {n_months} months of history; the plan wants 20+. Backtests will be thin.")
    return out


if __name__ == "__main__":
    cmd, path = (sys.argv + [None, None])[1:3]
    if cmd == "inspect" and path:
        inspect(path)
    elif cmd == "build" and path:
        build(path)
    else:
        sys.exit("usage: python -m src.sources.jobstreet [inspect|build] <jobs.csv>")
