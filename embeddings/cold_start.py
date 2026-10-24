import argparse
from embeddings import db, embedder
from embeddings.graph import job_skill_profile
from embeddings.normalize import HIGH, _T

K = 5
MIN_SIM = _T.get("cold_start_min") or 0.50

def infer_job_profile(conn, title, k=K, limit=15):
    tag = db.check_version(conn)
    v = db.to_pgvector(embedder.embed_queries([title])[0])
    rows = conn.execute(
        "SELECT entity_id, name, 1 - (embedding <=> %s::vector) FROM entity_embedding "
        "WHERE entity_type = 'job' AND model_version = %s ORDER BY embedding <=> %s::vector LIMIT %s",
        (v, tag, v, k),
    ).fetchall()
    neighbours = [{"job_id": j, "name": n, "similarity": round(float(s), 4)} for j, n, s in rows]
    best = neighbours[0]["similarity"] if neighbours else 0.0
    result = {"title": title, "confidence": best, "neighbours": neighbours, "model_version": tag}

    if best < MIN_SIM:
        return {**result, "source": "insufficient_data", "skills": []}

    if best >= HIGH:
        top = neighbours[0]
        return {**result, "source": "taxonomy_match", "job_id": top["job_id"],
                "neighbours": [top], "skills": job_skill_profile(conn, top["job_id"], limit=limit)}

    used = [n for n in neighbours if n["similarity"] >= MIN_SIM]
    total = sum(n["similarity"] for n in used)
    skills = {}
    for n in used:
        share = n["similarity"] / total
        for d in job_skill_profile(conn, n["job_id"], limit=None):
            s = skills.setdefault(d["skill_id"], {"skill_id": d["skill_id"], "name": d["name"],
                                                  "weight": 0.0, "from": []})
            s["weight"] += share * d["weight"]
            s["from"].append(n["job_id"])
    ranked = sorted(skills.values(), key=lambda d: -d["weight"])[:limit]
    for d in ranked:
        d["weight"] = round(d["weight"], 4)
    return {**result, "source": "inferred_from_analogous_role", "neighbours": used, "skills": ranked}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("titles", nargs="+")
    args = p.parse_args()
    with db.connect() as conn:
        for t in args.titles:
            r = infer_job_profile(conn, t, limit=6)
            nb = ", ".join(f"{n['job_id']} {n['similarity']:.2f}" for n in r["neighbours"][:3])
            print(f"\n{t!r}: {r['source']} (confidence {r['confidence']:.2f}) via {nb}")
            for d in r["skills"]:
                print(f"  {d['skill_id']:28s} {d['weight']:.2f}")
