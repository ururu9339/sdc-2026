import argparse
import numpy as np
from embeddings import db

EMBED_TOP_K = 25
MIN_POSTINGS = 20


def _vectors(conn, entity_type, tag):
    rows = conn.execute(
        "SELECT entity_id, embedding::text FROM entity_embedding WHERE entity_type = %s AND model_version = %s "
        "ORDER BY entity_id",
        (entity_type, tag),
    ).fetchall()
    ids = [r[0] for r in rows]
    vecs = np.array([np.array(r[1][1:-1].split(","), dtype=np.float32) for r in rows])
    return ids, vecs


def build_graph(conn):
    tag = db.check_version(conn)
    db.init_schema(conn)

    cooc = conn.execute(
        """
        WITH jobs AS (SELECT DISTINCT posting_id, entity_id AS job_id FROM posting_entity
                      WHERE entity_type = 'job' AND confidence = 'high'),
             skills AS (SELECT DISTINCT posting_id, entity_id AS skill_id FROM posting_entity
                        WHERE entity_type = 'skill' AND confidence = 'high'),
             totals AS (SELECT job_id, count(*) AS n FROM jobs GROUP BY job_id)
        SELECT j.job_id, s.skill_id, count(*) AS c, t.n
        FROM jobs j JOIN skills s USING (posting_id) JOIN totals t USING (job_id)
        GROUP BY j.job_id, s.skill_id, t.n
        """
    ).fetchall()
    totals = dict(conn.execute(
        "SELECT entity_id, count(DISTINCT posting_id) FROM posting_entity "
        "WHERE entity_type = 'job' AND confidence = 'high' GROUP BY entity_id"
    ).fetchall())

    edges = {(j, s): {"c": c, "n": n, "cw": c / n, "sim": None} for j, s, c, n in cooc}

    job_ids, J = _vectors(conn, "job", tag)
    skill_ids, S = _vectors(conn, "skill", tag)
    sims = J @ S.T
    for ji, job in enumerate(job_ids):
        for si in np.argsort(-sims[ji])[:EMBED_TOP_K]:
            e = edges.setdefault((job, skill_ids[si]), {"c": 0, "n": totals.get(job, 0), "cw": None, "sim": None})
            e["sim"] = float(sims[ji, si])
    index = {s: i for i, s in enumerate(skill_ids)}
    jindex = {j: i for i, j in enumerate(job_ids)}
    for (job, skill), e in edges.items():
        if e["sim"] is None:
            e["sim"] = float(sims[jindex[job], index[skill]])

    with conn.transaction():
        conn.execute("TRUNCATE job_skill_edge")
        conn.cursor().executemany(
            "INSERT INTO job_skill_edge (job_id, skill_id, cooc_postings, job_postings, cooc_weight, embed_sim, "
            "model_version) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [(j, s, e["c"], e["n"], e["cw"], e["sim"], tag) for (j, s), e in edges.items()],
        )
    conn.commit()
    return {"edges": len(edges), "cooc_edges": len(cooc), "jobs_with_postings": len(totals)}


def job_skill_profile(conn, job_id, limit=15):
    rows = conn.execute(
        "SELECT e.skill_id, x.name, e.cooc_postings, e.job_postings, e.cooc_weight, e.embed_sim "
        "FROM job_skill_edge e JOIN entity_embedding x ON x.entity_id = e.skill_id WHERE e.job_id = %s",
        (job_id,),
    ).fetchall()
    if not rows:
        return []
    n = rows[0][3]
    alpha = min(n / MIN_POSTINGS, 1.0)
    top_k = sorted((r[5] for r in rows if r[5] is not None), reverse=True)[:EMBED_TOP_K]
    hi, lo = (top_k[0], top_k[-1]) if top_k else (1.0, 0.0)
    out = []
    for skill_id, name, c, _, cw, sim in rows:
        sim_w = min(max(((sim or lo) - lo) / ((hi - lo) or 1.0), 0.0), 1.0)
        weight = alpha * (cw or 0.0) + (1 - alpha) * sim_w
        if weight <= 0:
            continue
        evidence = "co_occurrence" if alpha == 1 else "embedding_similarity" if not cw else "mixed"
        out.append({"skill_id": skill_id, "name": name, "weight": round(weight, 4), "evidence": evidence,
                    "cooc_postings": c, "job_postings": n, "cooc_weight": cw, "embed_sim": sim})
    out.sort(key=lambda d: -d["weight"])
    return out[:limit]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("jobs", nargs="*", help="job ids to show; omit to rebuild the graph")
    args = p.parse_args()
    with db.connect() as conn:
        if not args.jobs:
            print(build_graph(conn))
        for job in args.jobs:
            print(f"\n{job}")
            for d in job_skill_profile(conn, job, limit=8):
                cw = f"{d['cooc_postings']}/{d['job_postings']}"
                print(f"  {d['skill_id']:28s} {d['weight']:.2f}  {d['evidence']:21s} cooc={cw:7s} sim={d['embed_sim']:.2f}")
