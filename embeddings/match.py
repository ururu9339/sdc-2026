"""Step 8: match a user's skills to jobs and rank their skill gaps.

    python -m embeddings.match "python3" "SQL" "excel" "data viz"

Similarity decides overlap; growth numbers come only from the forecaster's entity_forecast table.
An entity without a forecast is marked 'no_forecast' and treated as neutral (growth 0), never estimated here.
"""
import argparse
import math

from embeddings import db
from embeddings.graph import job_skill_profile
from embeddings.normalize import normalize

HORIZON = 12        # months
REACH = 0.5         # weighted share of a job's profile the user already has
DECLINE_LOG = 0.10  # src/training/config.py DIRECTION_THRESHOLD: below -this log growth the forecaster says declining
PROFILE_SIZE = 15   # skills per job considered
TOP_TARGETS = 3     # jobs whose gaps are ranked


def declining(growth):
    """Same rule as the forecaster's direction label: median log growth below -DECLINE_LOG."""
    return growth is not None and math.log1p(growth) < -DECLINE_LOG


def resolve_skills(conn, items):
    """Skill IDs picked from the taxonomy pass through; free text is normalized like posting mentions."""
    known = {r[0] for r in conn.execute("SELECT entity_id FROM entity_embedding WHERE entity_type = 'skill'")}
    ids, report = {}, []
    text = [s for s in items if s not in known]
    for s in items:
        if s in known:
            ids[s] = "high"
            report.append({"input": s, "skill_id": s, "status": "picked"})
    for m in normalize(conn, text, "skill") if text else []:
        report.append({"input": m.mention, "skill_id": m.entity_id, "status": m.status, "score": round(m.score, 3)})
        if m.entity_id and ids.get(m.entity_id) != "high":
            ids[m.entity_id] = m.status
    return ids, report


def forecasts(conn, horizon=HORIZON):
    rows = conn.execute(
        "SELECT DISTINCT ON (entity_id) entity_id, growth, model_version FROM entity_forecast "
        "WHERE horizon_months = %s ORDER BY entity_id, created_at DESC",
        (horizon,),
    ).fetchall()
    return {e: (g, v) for e, g, v in rows}


def match_user(conn, skills, horizon=HORIZON):
    have, resolved = resolve_skills(conn, skills)
    fc = forecasts(conn, horizon)
    jobs = conn.execute(
        "SELECT DISTINCT e.job_id, x.name FROM job_skill_edge e JOIN entity_embedding x ON x.entity_id = e.job_id"
    ).fetchall()

    results = []
    for job_id, name in jobs:
        profile = job_skill_profile(conn, job_id, limit=PROFILE_SIZE)
        total = sum(d["weight"] for d in profile) or 1.0
        matched = [d for d in profile if d["skill_id"] in have]
        coverage = sum(d["weight"] for d in matched) / total
        growth, fv = fc.get(job_id, (None, None))
        results.append({
            "job_id": job_id, "name": name, "coverage": round(coverage, 3),
            "job_growth": growth, "forecast": "ok" if growth is not None else "no_forecast", "forecast_version": fv,
            "matched": [d["skill_id"] for d in matched],
            "missing": [d for d in profile if d["skill_id"] not in have],
        })

    results.sort(key=lambda r: (-r["coverage"], -(r["job_growth"] or 0)))
    # Within reach: enough overlap, and the forecaster does not call the job declining. A small dip it still
    # labels stable (e.g. -3%) is noise, not a reason to hide a job the user nearly qualifies for.
    within_reach = [r for r in results if r["coverage"] >= REACH and not declining(r["job_growth"])]
    targets = (within_reach or results)[:TOP_TARGETS]

    # Gap score = graph weight x (1 + forecast skill growth): weight alone when there is no forecast,
    # boosted for growing skills, discounted for shrinking ones. Summed over target jobs, so a gap shared
    # by several jobs earns credit in proportion to how much each needs it, not just for being shared.
    gaps = {}
    for r in targets:
        for d in r["missing"]:
            growth, _ = fc.get(d["skill_id"], (None, None))
            score = d["weight"] * (1 + (growth or 0.0))
            g = gaps.setdefault(d["skill_id"], {
                "skill_id": d["skill_id"], "name": d["name"], "score": 0.0, "skill_growth": growth,
                "forecast": "ok" if growth is not None else "no_forecast", "for_jobs": [], "evidence": d["evidence"]})
            g["score"] += score
            g["for_jobs"].append(r["job_id"])
    for g in gaps.values():
        g["score"] = round(g["score"], 4)
    ranked_gaps = sorted(gaps.values(), key=lambda g: -g["score"])

    for r in results:
        r["missing"] = [d["skill_id"] for d in r["missing"]]
    return {"resolved_skills": resolved, "within_reach": within_reach, "targets": [r["job_id"] for r in targets],
            "skill_gaps": ranked_gaps, "all_jobs": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("skills", nargs="+", help="skill IDs from the taxonomy or free-text CV skills")
    p.add_argument("--horizon", type=int, default=HORIZON)
    args = p.parse_args()
    with db.connect() as conn:
        out = match_user(conn, args.skills, args.horizon)
    for s in out["resolved_skills"]:
        print(f"  {s['input']!r:22} -> {s['skill_id']} ({s['status']})")
    print("\nWithin reach:" if out["within_reach"] else "\nNothing within reach; closest jobs:")
    for r in (out["within_reach"] or out["all_jobs"])[:5]:
        g = f"{r['job_growth']:+.0%}" if r["job_growth"] is not None else "no forecast"
        print(f"  {r['job_id']:28s} coverage {r['coverage']:.0%}  growth {g}")
    print(f"\nSkill gaps for {', '.join(out['targets'])}:")
    for gp in out["skill_gaps"][:8]:
        g = f"{gp['skill_growth']:+.0%}" if gp["skill_growth"] is not None else "no forecast"
        print(f"  {gp['skill_id']:28s} {gp['score']:.2f}  growth {g:12s} for {len(gp['for_jobs'])} job(s)")
