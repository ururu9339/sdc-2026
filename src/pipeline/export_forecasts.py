"""Export forecasts as rows for the teammates' entity_forecast table (embeddings/schema.sql).

    python3 -m src.pipeline.export_forecasts [--feed live|preview] [--dry-run] [--csv PATH] [--show N]

One row per entity: the newest status-ok record of the chosen feed.
  entity_id       taxonomy id ('skill:python', 'job:accountant'). Ids the taxonomy lacks are skipped and counted.
  horizon_months  round(horizon_periods * period_days / 30.44): 12 weekly periods -> 3 months.
  growth          exp(p50) - 1. This is the median forecast relative change in the SMOOTHED SHARE of postings
                  over the horizon (share = postings with the entity / all postings), not a change in absolute
                  posting counts. 0.12 means the share is forecast to be 12% higher.
  model_version   'v<N>' for the live feed, 'preview:v<N>' for the preview feed.
  created_at      export time, UTC.

Writing to Postgres needs DATABASE_URL and psycopg; embeddings.db is imported only then. The SQL is tested
with a fake connection only. It has NOT been run against a real Postgres.
Refuses the preview feed while a validated live model exists: preview rows must never sit beside live rows,
because match.py reads the newest row per entity regardless of model_version.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone

from src.pipeline.store import AppendOnlyLog, load_json
from src.sources.taxonomy import load_vocabulary
from src.training import config as C

COLUMNS = ["entity_id", "horizon_months", "growth", "model_version", "created_at"]
UPSERT_SQL = (
    "INSERT INTO entity_forecast (entity_id, horizon_months, growth, model_version, created_at) "
    "VALUES (%s, %s, %s, %s, %s) "
    "ON CONFLICT (entity_id, horizon_months, model_version) "
    "DO UPDATE SET growth = EXCLUDED.growth, created_at = EXCLUDED.created_at"
)


class ExportRefused(RuntimeError):
    pass


def horizon_months(horizon_periods: int, period_days: float = C.PERIOD_DAYS[C.PERIOD_FREQ]) -> int:
    return round(horizon_periods * period_days / 30.44)


def model_version_for(feed: str, record_version: str) -> str:
    bare = record_version.removeprefix("preview:")
    return f"preview:{bare}" if feed == "preview" else bare


def rows_from_records(records: list[dict], feed: str, valid_ids: set[str], now: datetime,
                      period_days: float = C.PERIOD_DAYS[C.PERIOD_FREQ]) -> tuple[list[dict], dict]:
    """Rows for entity_forecast plus counts of what was skipped. Pure function: no I/O."""
    counts = {"records": len(records), "not_ok": 0, "not_in_taxonomy": 0, "entities_exported": 0,
              "from_older_run": 0}
    newest_ok = {}
    for rec in records:
        if rec.get("status") != "ok":
            counts["not_ok"] += 1
            continue
        best = newest_ok.get(rec["entity_id"])
        if best is None or rec["as_of"] > best["as_of"]:
            newest_ok[rec["entity_id"]] = rec
    exportable = {e: r for e, r in newest_ok.items() if e in valid_ids}
    counts["not_in_taxonomy"] = len(newest_ok) - len(exportable)
    feed_newest = max((r["as_of"] for r in exportable.values()), default=None)
    rows = []
    for ent in sorted(exportable):
        rec = exportable[ent]
        if rec["as_of"] != feed_newest:
            counts["from_older_run"] += 1
        rows.append({
            "entity_id": ent,
            "horizon_months": horizon_months(int(rec["horizon_periods"]), period_days),
            "growth": math.exp(float(rec["p50"])) - 1.0,
            "model_version": model_version_for(feed, rec["model_version"]),
            "created_at": now,
        })
    counts["entities_exported"] = len(rows)
    return rows, counts


def check_feed(feed: str, live_version: str | None) -> None:
    if feed == "preview" and live_version is not None:
        raise ExportRefused(f"refusing to export the preview feed: validated live model {live_version} exists. "
                            "Preview rows must not sit beside live rows in entity_forecast.")


def write_rows(conn, rows: list[dict]) -> int:
    """Upsert into entity_forecast on a DB-API connection (psycopg 3 style). Returns rows written."""
    params = [(r["entity_id"], r["horizon_months"], r["growth"], r["model_version"], r["created_at"]) for r in rows]
    with conn.cursor() as cur:
        cur.executemany(UPSERT_SQL, params)
    conn.commit()
    return len(params)


def write_to_database(rows: list[dict], url: str | None = None) -> int:
    from embeddings import db  # lazy: pulls in psycopg, numpy and the embedder config
    with db.connect(url) as conn:
        return write_rows(conn, rows)


def export(feed: str = "live", now: datetime | None = None, valid_ids: set[str] | None = None,
           records: list[dict] | None = None, live_version: str | None = None) -> tuple[list[dict], dict]:
    now = now or datetime.now(timezone.utc)
    live = live_version if live_version is not None else load_json(C.VERSION_LOG_PATH, {"live": None}).get("live")
    check_feed(feed, live)
    if records is None:
        log = C.PREVIEW_LOG_PATH if feed == "preview" else C.FORECAST_LOG_PATH
        records = AppendOnlyLog(log).read()
    if valid_ids is None:
        vocab = load_vocabulary()
        valid_ids = vocab.ids("skill") | vocab.ids("job")
    return rows_from_records(records, feed, valid_ids, now)


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "created_at": r["created_at"].isoformat()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--feed", choices=["live", "preview"], default="live")
    ap.add_argument("--dry-run", action="store_true", help="print the rows and write nothing")
    ap.add_argument("--csv", metavar="PATH", help="write the rows to a CSV file instead of the database")
    ap.add_argument("--show", type=int, default=10, help="rows to print in a dry run (default 10)")
    a = ap.parse_args(argv)
    try:
        rows, counts = export(a.feed)
    except ExportRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"feed={a.feed} records={counts['records']} exported={counts['entities_exported']} "
          f"not_ok={counts['not_ok']} not_in_taxonomy={counts['not_in_taxonomy']} "
          f"from_older_run={counts['from_older_run']}")
    if a.dry_run or a.csv:
        for r in rows[: a.show]:
            print(f"  {r['entity_id']:28s} h={r['horizon_months']} growth={r['growth']:+.4f} {r['model_version']}")
        if len(rows) > a.show:
            print(f"  ... {len(rows) - a.show} more")
    if a.csv:
        write_csv(rows, a.csv)
        print(f"wrote {len(rows)} rows to {a.csv}")
    elif not a.dry_run:
        n = write_to_database(rows)
        print(f"upserted {n} rows into entity_forecast (NOT verified against a real Postgres)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
