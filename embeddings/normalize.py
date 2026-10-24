import argparse
import csv
from dataclasses import dataclass, field

from embeddings import db, embedder, versioning

# From the active version log entry when it matches these settings (evaluate.py --record), else hand-set fallbacks.
_T = versioning.active_thresholds()
HIGH = _T.get("high") or 0.60
LOW = _T.get("low") or 0.45
TOP_K = 5

@dataclass
class Match:
    mention: str
    entity_type: str
    status: str                      
    entity_id: str | None
    score: float
    candidates: list = field(default_factory=list)


def classify(score):
    return "high" if score >= HIGH else "low" if score >= LOW else "unmapped"


def normalize(conn, mentions, entity_type):
    """Embed mentions as queries and look up the top-k entities of the same type."""
    tag = db.check_version(conn)
    mentions = list(mentions)
    vecs = embedder.embed_queries(mentions)
    out = []
    for m, v in zip(mentions, vecs):
        rows = conn.execute(
            "SELECT entity_id, name, 1 - (embedding <=> %s::vector) AS score FROM entity_embedding "
            "WHERE entity_type = %s AND model_version = %s ORDER BY embedding <=> %s::vector LIMIT %s",
            (db.to_pgvector(v), entity_type, tag, db.to_pgvector(v), TOP_K),
        ).fetchall()
        best_id, _, best = rows[0] if rows else (None, None, 0.0)
        status = classify(best)
        out.append(Match(m, entity_type, status, best_id if status != "unmapped" else None, float(best),
                         [(i, n, round(float(s), 4)) for i, n, s in rows]))
    return out


def _month(s):
    return f"{s.strip()[:7]}-01" 

def map_postings(conn, records):
    """records: iterable of dicts with posting_id, month, mention, entity_type.
    Writes high/low matches to posting_entity and the rest to unmapped_mention."""
    records = [r for r in records if r["mention"].strip()]
    tag = embedder.version_tag()
    counts = {"high": 0, "low": 0, "unmapped": 0}
    for etype in ("skill", "job"):
        group = [r for r in records if r["entity_type"] == etype]
        if not group:
            continue
        unique = list(dict.fromkeys(r["mention"].strip() for r in group))
        by_mention = {m.mention: m for m in normalize(conn, unique, etype)}
        with conn.transaction():
            for r in group:
                m = by_mention[r["mention"].strip()]
                month = _month(r["month"])
                counts[m.status] += 1
                if m.status == "unmapped":
                    conn.execute(
                        "INSERT INTO unmapped_mention (mention, entity_type, best_entity_id, best_score, "
                        "first_seen, last_seen, model_version) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (mention, entity_type, model_version) DO UPDATE SET "
                        "occurrences = unmapped_mention.occurrences + 1, "
                        "first_seen = LEAST(unmapped_mention.first_seen, EXCLUDED.first_seen), "
                        "last_seen = GREATEST(unmapped_mention.last_seen, EXCLUDED.last_seen)",
                        (m.mention, etype, m.candidates[0][0] if m.candidates else None, m.score, month, month, tag),
                    )
                else:
                    conn.execute(
                        "INSERT INTO posting_entity (posting_id, month, entity_id, entity_type, mention, score, "
                        "confidence, model_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (posting_id, entity_id, mention) DO UPDATE SET score = EXCLUDED.score, "
                        "confidence = EXCLUDED.confidence, model_version = EXCLUDED.model_version",
                        (r["posting_id"], month, m.entity_id, etype, m.mention, m.score, m.status, tag),
                    )
    conn.commit()
    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mentions", nargs="*")
    p.add_argument("--type", choices=["skill", "job"], default="skill")
    p.add_argument("--input", help="CSV of posting_id, month, mention, entity_type to map into Postgres")
    args = p.parse_args()
    with db.connect() as conn:
        if args.input:
            with open(args.input, newline="", encoding="utf-8") as fh:
                print(map_postings(conn, csv.DictReader(fh)))
        for m in normalize(conn, args.mentions, args.type):
            top = " | ".join(f"{i} {s:.2f}" for i, _, s in m.candidates[:3])
            print(f"{m.mention:28s} {m.status:9s} -> {top}")
