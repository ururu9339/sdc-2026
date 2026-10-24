from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.sources.common import long_frame, period_rows, select_periods, source_cfg, write_outputs
from src.sources.validate import validate
from src.training import config as C

SOURCE_ID = "job_sdf"
DEFAULT_DATASET = C.ROOT / "src" / "data" / "job_sdf" / "dataset"
TOP_SKILLS = 300
MIN_SKILL_TOTAL = 1000
IDENTITY_TOL = 1e-9
INTEGER_TOL = 1e-6
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
KEYS_R0, KEYS_R1 = ["r0_id", "skill_id"], ["r1_id", "skill_id"]


def _read_pair(dataset: Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    demand = pd.read_parquet(dataset / "demand" / f"{name}.parquet")
    prop = pd.read_parquet(dataset / "proportion" / f"{name}.parquet")
    keys = KEYS_R0 if name == "r0" else KEYS_R1
    if list(demand.columns) != list(prop.columns) or not (demand[keys].to_numpy() == prop[keys].to_numpy()).all():
        raise ValueError(f"{name}: demand and proportion tables do not have the same rows and columns")
    return demand, prop


def _ads(demand: np.ndarray, prop: np.ndarray, label: str) -> np.ndarray:
    p_tot = prop.sum(axis=0)
    if (p_tot <= 0).any():
        raise ValueError(f"{label}: a month has no proportion mass")
    ads = demand.sum(axis=0) / p_tot
    if np.abs(ads - np.round(ads)).max() > INTEGER_TOL:
        raise ValueError(f"{label}: implied job-ad counts are not integers")
    resid = float(np.abs(demand / ads - prop).max())
    if resid > IDENTITY_TOL:
        raise ValueError(f"{label}: proportion is not demand / job-ad count (max residual {resid:.3g})")
    return np.round(ads).astype(np.int64)


def _checked(dataset: Path) -> dict:
    d0, p0 = _read_pair(dataset, "r0")
    d1, p1 = _read_pair(dataset, "r1")
    months = [c for c in d0.columns if MONTH_RE.match(str(c))]
    if not months or [c for c in d1.columns if MONTH_RE.match(str(c))] != months:
        raise ValueError("demand tables do not share the same monthly columns")
    if pd.period_range(months[0], months[-1], freq="M").strftime("%Y-%m").tolist() != months:
        raise ValueError(f"monthly columns are not contiguous: {months[0]}..{months[-1]}")
    if d0["r0_id"].nunique() != 1 or d0.duplicated(KEYS_R0).any():
        raise ValueError("r0 is expected to be one all-occupation group with unique skill rows")
    if d1["r1_id"].nunique() < 1 or d1.duplicated(KEYS_R1).any():
        raise ValueError("r1 is expected to hold L1 occupations with unique (occupation, skill) rows")
    for name, d in (("r0", d0), ("r1", d1)):
        if (d[months].to_numpy() < 0).any() or not np.isfinite(d[months].to_numpy()).all():
            raise ValueError(f"{name}: demand must be finite and non-negative")

    D0, P0 = d0[months].to_numpy(float), p0[months].to_numpy(float)
    ads_all = _ads(D0, P0, "r0")
    skills = d0.set_index("skill_id")[months].astype(float)


    occ_ids = sorted(d1["r1_id"].unique())
    occ = pd.DataFrame(index=months, columns=occ_ids, dtype=np.int64)
    for o in occ_ids:
        m = (d1["r1_id"] == o).to_numpy()
        occ[o] = _ads(d1.loc[m, months].to_numpy(float), p1.loc[m, months].to_numpy(float), f"r1 occupation {o}")
    if not (occ.sum(axis=1).to_numpy() == ads_all).all():
        raise ValueError("occupation job-ad counts do not sum to the all-occupation total")


    by_skill = d1.groupby("skill_id")[months].sum().reindex(skills.index).fillna(0.0)
    if not np.array_equal(by_skill.to_numpy(), skills.to_numpy()):
        raise ValueError("r0 demand is not the sum of r1 demand over occupations")

    return {"months": months, "skills": skills, "ads": ads_all, "occupations": occ, "n_r1_rows": len(d1)}


def _facts(chk: dict, dataset: Path) -> dict:
    ads = pd.Series(chk["ads"], index=chk["months"])
    skill_tot = chk["skills"].sum(axis=1)
    return {
        "dataset": str(dataset),
        "identities_ok": True,
        "months": [chk["months"][0], chk["months"][-1], len(chk["months"])],
        "all_occupation_ads_per_month": ads.astype(int).to_dict(),
        "total_ads": int(ads.sum()),
        "total_mentions": int(chk["skills"].to_numpy().sum()),
        "skills_in_r0": int(len(skill_tot)),
        "skills_with_mentions": int((skill_tot > 0).sum()),
        "occupations": int(chk["occupations"].shape[1]),
        "occupation_ads_total": chk["occupations"].sum(axis=0).astype(int).to_dict(),
        "mentions_per_ad_mean": float(chk["skills"].to_numpy().sum() / ads.sum()),
    }


def inspect(dataset: str | Path = DEFAULT_DATASET) -> dict:
    dataset = Path(dataset)
    return _facts(_checked(dataset), dataset)


def build_observations(dataset: Path, cfg: dict | None = None, top_skills: int = TOP_SKILLS,
                       min_skill_total: int = MIN_SKILL_TOTAL) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or source_cfg(SOURCE_ID)
    chk = _checked(dataset)
    months = chk["months"]
    periods = pd.to_datetime([m + "-01" for m in months])
    ads = pd.Series(chk["ads"], index=periods)

    skills = chk["skills"]
    skill_tot = skills.sum(axis=1).sort_values(ascending=False, kind="stable")
    top = skill_tot.head(top_skills)
    dropped_low = top[top < min_skill_total]
    keep = top[top >= min_skill_total].index
    skill_counts = skills.loc[keep].T
    skill_counts.index, skill_counts.columns = periods, [f"skill_{int(s)}" for s in keep]

    occ = chk["occupations"].copy()
    occ.index, occ.columns = periods, [f"occupation_{int(o)}" for o in occ.columns]

    gran, source_id = cfg["granularity"], cfg["source_id"]
    grid, dropped_periods = select_periods(ads, cfg["min_period_postings"], gran, cfg["dense"])
    totals = ads.reindex(grid)
    obs = pd.concat([
        long_frame(skill_counts.reindex(grid), totals, "skill", source_id),
        long_frame(occ.reindex(grid), totals, "job", source_id),
    ], ignore_index=True)
    facts = _facts(chk, dataset)
    meta = {
        "granularity": gran,
        "first_period": str(grid[0].date()), "last_period": str(grid[-1].date()),
        "periods": period_rows(totals),
        "dropped_periods": dropped_periods,
        "min_period_postings": cfg["min_period_postings"],
        "top_skills_requested": top_skills,
        "min_skill_total": min_skill_total,
        "skills_kept": int(len(keep)),
        "skills_dropped_low_total": [{"skill": f"skill_{int(s)}", "total_mentions": int(t)}
                                     for s, t in dropped_low.items()],
        "kept_entities": {"skill": int(len(keep)), "job": int(occ.shape[1])},
        "value_definition": "share of all job ads in the month: count / ADS(month)",
        "support_definition": "skill: raw mention count in the month (r0 demand); "
                              "job: number of ads in the occupation that month (r1 ADS)",
        "denominator_definition": "ADS(month) = number of job advertisements in the month, all occupations "
                                  "(sum of r0 demand / sum of r0 proportion). Not a mention count.",
        "identities": {k: facts[k] for k in ("identities_ok", "total_ads", "total_mentions", "skills_in_r0",
                                             "skills_with_mentions", "occupations", "mentions_per_ad_mean")},
        "occupation_ids": [int(c.split("_")[1]) for c in occ.columns],
        "note": "Real Chinese job-ad data with anonymised ids. Proof-run only; not Malaysian market data.",
    }
    return obs, meta


def build(path: str | Path | None = None, out: Path | None = None, cfg: dict | None = None,
          top_skills: int = TOP_SKILLS, min_skill_total: int = MIN_SKILL_TOTAL) -> Path:
    dataset = Path(path or DEFAULT_DATASET)
    cfg = cfg or source_cfg(SOURCE_ID)
    out = Path(out or C.RAW_DIR / f"{cfg['source_id']}.csv")
    obs, meta = build_observations(dataset, cfg, top_skills, min_skill_total)
    res = validate(obs, cfg, None)
    if not res["ok"]:
        raise ValueError("validation failed: " + "; ".join(res["errors"]))
    meta = {"source_id": cfg["source_id"], "raw_file": str(dataset), **meta, "validation": res["facts"]}
    write_outputs(obs, meta, out)
    print(f"wrote {out}: {len(obs):,} rows, {meta['kept_entities']} entities, {len(meta['periods'])} periods")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "build"):
        p = sub.add_parser(name)
        p.add_argument("dataset", nargs="?", default=str(DEFAULT_DATASET))
    a = ap.parse_args()
    if a.cmd == "inspect":
        import json
        print(json.dumps(inspect(a.dataset), indent=2, default=str))
    else:
        build(a.dataset)


if __name__ == "__main__":
    main()
