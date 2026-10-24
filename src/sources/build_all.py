"""Build every registered source from its raw_file, validate the outputs and summarise coverage.

    python3 -m src.sources.build_all [--only job_postings,jobstreet_my]

Each registry entry with an `adapter` module is built with `build(raw_file)`, which writes
data/raw/<source_id>.csv plus its meta sidecar. The summary shows rows, entities and periods per
source (in each source's own granularity), and how many periods and entities each pair of sources
shares. Vocabulary collisions between job families and skills are an error.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

import pandas as pd

from src.sources.taxonomy import load_vocabulary
from src.sources.validate import validate
from src.training import config as C


def _overlap(outputs: dict[str, pd.DataFrame], granularity: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Periods of row-source a that overlap in time with any period of column-source b, and shared entity ids.
    Periods are compared as intervals [start, end), because sources have different granularities."""
    ids = list(outputs)
    starts = {k: pd.DatetimeIndex(sorted(set(pd.to_datetime(v["period"])))) for k, v in outputs.items()}
    ends = {k: C.span_end(starts[k], granularity[k]) for k in ids}
    ents = {k: set(v["entity_id"]) for k, v in outputs.items()}
    m = pd.DataFrame(index=ids, columns=ids, dtype=int)
    e = pd.DataFrame(index=ids, columns=ids, dtype=int)
    for a in ids:
        for b in ids:
            # a's period overlaps b's when a.start < b.end and b.start < a.end
            b_start, b_end = starts[b].to_numpy(), ends[b].to_numpy()
            hit = [(a_s < b_end) & (b_start < a_e) for a_s, a_e in zip(starts[a].to_numpy(), ends[a].to_numpy())]
            m.loc[a, b] = int(sum(bool(h.any()) for h in hit))
            e.loc[a, b] = len(ents[a] & ents[b])
    return m, e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated source_ids to build")
    a = ap.parse_args()
    try:
        vocab = load_vocabulary()
    except ValueError as e:
        sys.exit(f"taxonomy: {e}")
    vocab_ids = vocab.ids("skill") | vocab.ids("job")
    registry = C.load_registry()
    wanted = set(a.only.split(",")) if a.only else None
    sources = [s for s in registry["sources"] if s.get("adapter") and (wanted is None or s["source_id"] in wanted)]

    outputs, facts, timings = {}, {}, {}
    t_all = time.time()
    for src in sources:
        sid = src["source_id"]
        t0 = time.time()
        out = importlib.import_module(src["adapter"]).build(C.ROOT / src["raw_file"])
        timings[sid] = time.time() - t0
        outputs[sid] = pd.read_csv(out)

    target = C.target_source(registry)
    tpath = C.RAW_DIR / f"{target}.csv"
    target_entities = set(pd.read_csv(tpath, usecols=["entity_id"])["entity_id"]) if tpath.exists() else None

    print("\nvalidation")
    for src in sources:
        sid = src["source_id"]
        res = validate(outputs[sid], src, target_entities, vocabulary_ids=vocab_ids)
        facts[sid] = res["facts"]
        print(f"  {sid}: {'OK' if res['ok'] else 'FAILED'}  build {timings[sid]:.1f}s")
        for e in res["errors"]:
            print("    ERROR  ", e)
        for w in res["warnings"]:
            print("    warning", w)

    print("\n| source | granularity | rows | entities | periods | first | last |")
    print("|---|---|---|---|---|---|---|")
    for sid, f in facts.items():
        print(f"| {sid} | {f['granularity']} | {f['rows']:,} | {f['entities']} | {f['periods']} | "
              f"{f['first']} | {f['last']} |")
    print(f"\ntotal build time {time.time() - t_all:.1f}s")

    if len(outputs) > 1:
        m, e = _overlap(outputs, {s["source_id"]: s["granularity"] for s in sources})
        print("\nperiods of the row source overlapping in time with the column source")
        print(m.to_string())
        print("\nshared entity_ids (rows = source, cols = source)")
        print(e.to_string())


if __name__ == "__main__":
    main()
