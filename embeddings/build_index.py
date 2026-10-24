import argparse
import sys

from embeddings import db, embedder, versioning

def build(conn, rebuild=False):
    db.init_schema(conn)
    tag = embedder.version_tag()
    stored = db.get_meta(conn, "model_version")
    if stored and stored != tag and not rebuild:
        raise SystemExit(f"Index was built with {stored!r}, current settings are {tag!r}. Re-run with --rebuild.")
    active = versioning.active()
    if stored and stored != tag and (not active or active["version_tag"] != tag):

        raise SystemExit(f"{tag!r} is not the active version in {versioning.LOG}. "
                         "Run python -m embeddings.eval.evaluate --record first.")
    if not active or active["version_tag"] != tag:
        print(f"warning: {tag!r} has not been evaluated (python -m embeddings.eval.evaluate --record)", file=sys.stderr)

    rows = embedder.load_taxonomy()
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("Duplicate ids in taxonomy CSVs")
    vecs = embedder.embed_documents([embedder.entity_text(r) for r in rows])

    with conn.transaction():
        if stored and stored != tag:
            conn.execute("TRUNCATE job_skill_edge, posting_entity, entity_embedding")
        conn.cursor().executemany(
            "INSERT INTO entity_embedding (entity_id, entity_type, name, embedding, model_version) "
            "VALUES (%s, %s, %s, %s::vector, %s) "
            "ON CONFLICT (entity_id) DO UPDATE SET entity_type = EXCLUDED.entity_type, name = EXCLUDED.name, "
            "embedding = EXCLUDED.embedding, model_version = EXCLUDED.model_version, created_at = now()",
            [(r["id"], r["type"], r["name"], db.to_pgvector(v), tag) for r, v in zip(rows, vecs)],
        )
        conn.execute("DELETE FROM job_skill_edge WHERE NOT (job_id = ANY(%s) AND skill_id = ANY(%s))", (ids, ids))
        conn.execute("DELETE FROM entity_embedding WHERE NOT (entity_id = ANY(%s))", (ids,))
        db.set_meta(conn, "model_version", tag)
        db.set_meta(conn, "model_digest", db.ollama_model_digest())
    conn.commit()
    return len(rows), tag


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true", help="replace an index built with different settings")
    args = p.parse_args()
    with db.connect() as conn:
        n, tag = build(conn, args.rebuild)
    print(f"Indexed {n} entities with {tag}")
