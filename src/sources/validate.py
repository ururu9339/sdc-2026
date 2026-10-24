from __future__ import annotations

import argparse

import pandas as pd

from src.sources.loaders import COLUMNS, GLOBAL_ID
from src.training import config as C

KINDS = {"skill", "job", "global"}


def validate(df: pd.DataFrame, source: dict | None = None, target_entities: set | None = None,
             vocabulary_ids: set | None = None) -> dict:
    errors, warnings = [], []
    missing = [c for c in COLUMNS if c not in df.columns and c != "support"]
    if missing:
        return {"ok": False, "errors": [f"missing columns: {missing}"], "warnings": [], "facts": {}}
    gran = (source or {}).get("granularity", "weekly")
    freq = C.GRANULARITY_FREQ[gran]
    df = df.copy()
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    if df["period"].isna().any():
        errors.append(f"{int(df['period'].isna().sum())} rows have an unparseable period")
    valid = df["period"].dropna()
    if (C.to_period_start(valid, freq) != valid).any():
        warnings.append(f"some periods are not {gran} period starts; aggregate to the period start first")
    value = pd.to_numeric(df["value"], errors="coerce")
    if value.isna().any():
        errors.append(f"{int(value.isna().sum())} rows have a non-numeric value")
    if (value < 0).any():
        errors.append("negative values found (counts, shares and amounts must be >= 0)")
    dups = int(df.duplicated(["entity_id", "period", "source_id"]).sum())
    if dups:
        errors.append(f"{dups} duplicate (entity_id, period, source_id) rows")
    bad_kind = set(df["kind"].dropna()) - KINDS
    if bad_kind:
        errors.append(f"unknown kind values {sorted(bad_kind)}; use skill|job (or 'global' for economy-wide)")
    if vocabulary_ids is not None:
        known = set(vocabulary_ids) | {GLOBAL_ID}
        strays = sorted(set(df.loc[df["kind"].isin(["skill", "job"]), "entity_id"]) - known)
        if strays:
            errors.append(f"{len(strays)} entity ids are not in the shared taxonomy (e.g. {strays[:3]})")
    if source and source["source_id"] not in set(df["source_id"]):
        errors.append(f"no rows with source_id == {source['source_id']!r}")

    periods = int(C.to_period_start(df["period"].dropna(), freq).nunique())
    wanted = C.PERIODS_PER_YEAR[gran]
    if periods < wanted:
        warnings.append(f"only {periods} distinct {gran} periods (wants {wanted}+, i.e. one year)")
    facts = {"rows": len(df), "entities": df["entity_id"].nunique(), "granularity": gran, "periods": periods,
             "first": str(df["period"].min().date()), "last": str(df["period"].max().date())}
    scope = (source or {}).get("scope", "entity")
    if scope == "entity" and target_entities:
        overlap = len(set(df["entity_id"]) & target_entities) / max(len(target_entities), 1)
        facts["target_entity_overlap"] = round(overlap, 2)
        if overlap < 0.3:
            warnings.append(f"covers only {overlap:.0%} of the target's entities; use scope 'mapped' or 'global'")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "facts": facts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--source-id")
    a = ap.parse_args()
    registry = C.load_registry()
    source = next((s for s in registry["sources"] if s["source_id"] == a.source_id), None)
    target = None
    tpath = C.RAW_DIR / f"{C.target_source(registry)}.csv"
    if tpath.exists():
        target = set(pd.read_csv(tpath, usecols=["entity_id"])["entity_id"]) - {GLOBAL_ID}
    res = validate(pd.read_csv(a.path), source, target)
    print("OK" if res["ok"] else "FAILED", res["facts"])
    for e in res["errors"]:
        print("  ERROR  ", e)
    for w in res["warnings"]:
        print("  warning", w)


if __name__ == "__main__":
    main()
