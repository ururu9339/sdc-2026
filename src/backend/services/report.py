from __future__ import annotations

import math
import os
import re
from typing import Any, Callable

import pandas as pd

from src.backend.models.report import (JobFit, LearningPath, LearningStage, MarketItem, Report, ResolvedSkill,
                                       SkillGap, SkillRef)
from src.backend.services import forecast_store as store
from src.sources import taxonomy as T
from src.training import config as C


PROFILE_SIZE = 15
REACH = 0.5
TOP_TARGETS = 3
MIN_POSTINGS = 20

FORECAST_HORIZON_MONTHS = 3
BEST_FIT_SHOWN = 5
STAGE_TITLES = {1: ["Start here"], 2: ["Start here", "Next"], 3: ["Start here", "Next", "Later"]}
ENGINES = ("auto", "qwen_pgvector", "offline_alias")
MAX_SKILLS = 100
MAX_TEXT_CHARS = 20000
BLOB_MIN_ALIAS_CHARS = 3
GRAPH_FILENAME = store.GRAPH_FILENAME

LEARNING_NOTE = ("The order is a heuristic, not a curriculum: skills are sorted by how many postings in our data "
                 "mention them for your target jobs, adjusted by the forecast outlook where one exists. It does "
                 "not estimate time, cost or difficulty.")
BASE_CAVEATS = [
    "Coverage is an overlap score between your skill list and the most common skills in a job's postings. "
    "It is not a judgement of ability and not a hiring recommendation.",
    "Skill gaps come from job postings in our data. They show what postings ask for, not what any course teaches.",
    "The market snapshot is descriptive: the past share of postings, not a forecast.",
]


def _forecast_info(entity_id: str, horizon_months: int, cache: dict) -> dict:
    if entity_id in cache:
        return cache[entity_id]
    info: dict[str, Any] = {"growth": None, "forecast_status": "no_forecast", "direction": None,
                            "confidence": None, "signal_quality": None, "as_of": None, "target_period": None,
                            "shares_signal_with": None}
    _, rec = store.record_for(entity_id)
    if rec is not None and horizon_months == FORECAST_HORIZON_MONTHS and rec.get("status") == "ok" \
            and rec.get("p50") is not None:
        info.update(growth=math.exp(float(rec["p50"])) - 1.0, forecast_status="ok")
    elif rec is not None:
        info["forecast_status"] = "insufficient_data" if rec.get("status") != "ok" else "no_forecast"
    if rec is not None:
        info.update(direction=rec.get("direction"), confidence=rec.get("confidence"),
                    signal_quality=rec.get("signal_quality"), as_of=rec.get("as_of"),
                    target_period=rec.get("target_period"), shares_signal_with=rec.get("shares_signal_with"))
    cache[entity_id] = info
    return info


def _best_alias(text: str, aliases: dict) -> tuple[int, str, str] | None:
    best = None
    for sid, alist in aliases.items():
        for alias in alist:
            spec = T._specificity(alias)
            m = T._compile(alias).search(text)
            if m and (best is None or spec > best[0]):
                best = (spec, sid, m.group(0))
    return best


def _blob_matches(text: str, aliases: dict) -> dict[str, str]:
    low = text.lower()
    found: dict[str, str] = {}
    for sid, alist in aliases.items():
        best = None
        for alias in alist:
            if T._specificity(alias) < BLOB_MIN_ALIAS_CHARS:
                continue
            m = T._compile(alias).search(low)
            if m and (best is None or len(m.group(0)) > len(best)):
                best = m.group(0)
        if best:
            found[sid] = best
    return found


def _load_graph() -> pd.DataFrame | None:
    path = C.PROCESSED_DIR / GRAPH_FILENAME
    if not path.exists():
        return None
    return pd.read_csv(path)


def _offline_profiles(graph: pd.DataFrame | None) -> dict[str, list[dict]]:
    profiles: dict[str, list[dict]] = {}
    if graph is None or graph.empty:
        return profiles
    for job_id, part in graph.groupby("job", sort=True):
        rows = []
        for r in part.itertuples(index=False):
            n = int(r.n_job_postings)
            if n <= 0:
                continue
            share = float(r.n_cooccur) / n
            weight = min(n / MIN_POSTINGS, 1.0) * share
            if weight <= 0:
                continue
            rows.append({"skill_id": str(r.skill), "weight": round(weight, 4), "share": round(share, 4)})
        rows.sort(key=lambda d: (-d["weight"], d["skill_id"]))
        if rows:
            profiles[str(job_id)] = rows
    return profiles


def _score(profiles: dict[str, list[dict]], have: set[str], names: dict, info_of: Callable[[str], dict]) -> dict:
    jobs = []
    for job_id, full in profiles.items():
        profile = full[:PROFILE_SIZE]
        total = sum(d["weight"] for d in profile) or 1.0
        matched = [d for d in profile if d["skill_id"] in have]
        missing = [d for d in profile if d["skill_id"] not in have]
        coverage = sum(d["weight"] for d in matched) / total
        info = info_of(job_id)
        jobs.append({"job_id": job_id, "name": names.get(job_id, job_id), "coverage": coverage,
                     "matched": [d["skill_id"] for d in matched], "missing": [d["skill_id"] for d in missing],
                     "_missing_detail": missing, **_jobfit_info(info)})
    jobs.sort(key=lambda j: (-round(j["coverage"], 4), -(j["growth"] or 0.0), j["job_id"]))
    within = {j["job_id"] for j in jobs if j["coverage"] >= REACH and (j["growth"] is None or j["growth"] > 0)}
    pool = [j for j in jobs if j["job_id"] in within] or jobs
    targets = [j for j in pool[:TOP_TARGETS]] if jobs and jobs[0]["coverage"] > 0 else []

    gaps: dict[str, dict] = {}
    for job in targets:
        for d in job["_missing_detail"]:
            sid = d["skill_id"]
            info = info_of(sid)
            growth = info["growth"]
            score = d["weight"] * (1 + (growth or 0.0))
            g = gaps.get(sid)
            if g is None:
                g = gaps[sid] = {"skill_id": sid, "name": names.get(sid, sid), "score": -1.0, "weight": 0.0,
                                 "share": None, "best_job": None, "for_jobs": [], **_gapfit_info(info)}
            g["for_jobs"].append(job["job_id"])
            if score > g["score"]:
                g.update(score=score, weight=d["weight"], share=d["share"], best_job=job["job_id"])
    for job in jobs:
        job.pop("_missing_detail", None)
    return {"jobs": jobs, "within_reach": within, "targets": [t["job_id"] for t in targets],
            "gaps": list(gaps.values())}


def _jobfit_info(info: dict) -> dict:
    return {"growth": info["growth"], "forecast_status": info["forecast_status"], "direction": info["direction"],
            "confidence": info["confidence"], "signal_quality": info["signal_quality"],
            "as_of": info.get("as_of"), "target_period": info.get("target_period"),
            "shares_signal_with": info.get("shares_signal_with")}


_gapfit_info = _jobfit_info


def _offline_data(items: list[str], blob: dict[str, str], vocab: T.Vocabulary, horizon: int, warnings: list[str]):
    names = vocab.names
    skill_ids = vocab.ids("skill")
    resolved: list[ResolvedSkill] = []
    have: set[str] = set()
    seen: set[str] = set()
    for raw in items:
        if raw.lower() in seen:
            continue
        seen.add(raw.lower())
        sid, method = None, "none"
        if raw in skill_ids:
            sid, method = raw, "taxonomy_id"
        else:
            hit = _best_alias(raw.lower(), vocab.skill_aliases)
            if hit:
                sid, method = hit[1], "alias"
        if sid:
            have.add(sid)
            resolved.append(ResolvedSkill(input=raw, skill_id=sid, name=names[sid], status="matched", method=method))
        else:
            resolved.append(ResolvedSkill(input=raw, status="unresolved", method="none"))
    for sid, matched_text in blob.items():
        if sid not in have:
            have.add(sid)
            resolved.append(ResolvedSkill(input=matched_text, skill_id=sid, name=names[sid], status="matched",
                                          method="text_alias"))
    graph = _load_graph()
    if graph is None:
        warnings.append(f"The job-skill graph ({GRAPH_FILENAME}) was not found, so no job profiles exist.")
    cache: dict = {}
    scored = _score(_offline_profiles(graph), have, names, lambda e: _forecast_info(e, horizon, cache))
    return {"resolved": resolved, "have": have, **scored}


def _qwen_data(items: list[str], blob: dict[str, str], vocab: T.Vocabulary, horizon: int, warnings: list[str]):
    from embeddings import db as edb
    from embeddings import match as M
    from embeddings.graph import job_skill_profile

    names = vocab.names
    list_set = set(items)
    extra = [sid for sid in blob if sid not in list_set]
    with edb.connect() as conn:
        out = M.match_user(conn, items + extra, horizon)
        resolved: list[ResolvedSkill] = []
        for r in out["resolved_skills"]:
            inp, status = r["input"], r.get("status")
            if inp in extra and inp not in list_set:
                resolved.append(ResolvedSkill(input=blob[inp], skill_id=inp, name=names.get(inp, inp),
                                              status="matched", method="text_alias"))
            elif status in ("picked", "high"):
                resolved.append(ResolvedSkill(input=inp, skill_id=r["skill_id"], name=names.get(r["skill_id"]),
                                              status="matched",
                                              method="taxonomy_id" if status == "picked" else "qwen_embedding"))
            elif status == "low" and r.get("skill_id"):
                resolved.append(ResolvedSkill(input=inp, skill_id=r["skill_id"], name=names.get(r["skill_id"]),
                                              status="matched", method="qwen_embedding_low"))
                warnings.append(f"'{inp}' matched a skill with low similarity; check it in the resolved list.")
            else:
                resolved.append(ResolvedSkill(input=inp, status="unresolved", method="qwen_embedding"))
        have = {r.skill_id for r in resolved if r.status == "matched"}

        cache: dict = {}

        def info_of(entity_id: str) -> dict:
            return _forecast_info(entity_id, horizon, cache)

        def with_db_growth(info: dict, growth: float | None, db_status: str | None) -> dict:
            ok = growth is not None and db_status == "ok"
            return {**info, "growth": growth if ok else None, "forecast_status": "ok" if ok else "no_forecast",
                    "direction": info["direction"] if ok else None}

        within = {r["job_id"] for r in out["within_reach"]}
        jobs = []
        for r in out["all_jobs"]:
            jobs.append({"job_id": r["job_id"], "name": names.get(r["job_id"], r["name"]),
                         "coverage": r["coverage"], "matched": list(r["matched"]), "missing": list(r["missing"]),
                         **_jobfit_info(with_db_growth(info_of(r["job_id"]), r.get("job_growth"), r.get("forecast")))})
        profiles = {}
        for tj in out["targets"]:
            profiles[tj] = {d["skill_id"]: d for d in job_skill_profile(conn, tj, limit=PROFILE_SIZE)}
        gaps = []
        for g in out["skill_gaps"]:
            growth = g.get("skill_growth")
            info = with_db_growth(info_of(g["skill_id"]), growth, g.get("forecast"))
            best = None
            for tj in g["for_jobs"]:
                d = profiles.get(tj, {}).get(g["skill_id"])
                if d is None:
                    continue
                score = d["weight"] * (1 + (growth or 0.0))
                if best is None or score > best[0]:
                    best = (score, d, tj)
            gaps.append({"skill_id": g["skill_id"], "name": names.get(g["skill_id"], g["name"]),
                         "score": g["score"], "weight": best[1]["weight"] if best else 0.0,
                         "share": (best[1]["cooc_postings"] / best[1]["job_postings"]) if best and best[1].get(
                             "job_postings") else None,
                         "best_job": best[2] if best else (g["for_jobs"][0] if g["for_jobs"] else None),
                         "for_jobs": list(g["for_jobs"]), **_gapfit_info(info)})
    return {"resolved": resolved, "have": have, "jobs": jobs, "within_reach": within,
            "targets": list(out["targets"]), "gaps": gaps}


def _banner(feed: str) -> str | None:
    if feed == "live":
        win = store.feed_window()
        return store.live_banner(win["forecast_as_of"], win["forecast_target_period"])
    if feed == "preview":
        return store.preview_envelope()["banner"]
    return (f"NO FORECASTS: no model has passed the validation gate ({store.no_live_reason()}). Outlook figures are "
            "unavailable; skill and job matching still works.")


def _stages(gaps: list[dict]) -> list[str]:
    n = len(gaps)
    k = min(3, n)
    if k == 0:
        return []
    titles = STAGE_TITLES[k]
    base, extra = divmod(n, k)
    out: list[str] = []
    for i in range(k):
        out += [titles[i]] * (base + (1 if i < extra else 0))
    return out


def _why(gap: dict, names: dict, validated: bool) -> str:
    label = "validated forecast" if validated else "unvalidated preview"
    best = names.get(gap["best_job"], gap["best_job"]) if gap["best_job"] else None
    k = len(gap["for_jobs"])
    if gap["share"] is not None and best:
        text = f"Appears in {gap['share'] * 100:.0f}% of {best} postings in our data"
    elif best:
        text = f"Listed in the {best} profile"
    else:
        text = "Listed in a target job profile"
    if k > 1:
        text += f"; also a gap for {k - 1} other target job{'s' if k > 2 else ''}"
    if gap["forecast_status"] == "ok" and gap["growth"] is not None and gap["direction"]:
        text += f"; outlook: {gap['direction']} ({gap['growth'] * 100:+.0f}%, {label}, made as of {gap['as_of']})"
    elif gap["forecast_status"] == "insufficient_data":
        text += "; outlook: not enough history to forecast"
    else:
        text += "; no forecast available"
    if gap.get("shares_signal_with"):
        text += "; " + C.DUPLICATE_CAVEAT.format(canon=gap["shares_signal_with"])
    return text


def _market(gaps: list[dict], warnings: list[str]) -> list[MarketItem]:
    if not gaps:
        return []
    try:
        snap = store.snapshot(None, limit=100000)
    except Exception as exc:
        warnings.append(f"The market snapshot could not be computed ({type(exc).__name__}).")
        return []
    best: dict[str, dict] = {}
    gap_ids = {g["skill_id"] for g in gaps}
    for it in snap.get("items", []):
        eid = it.get("entity_id")
        if eid in gap_ids and it.get("kind") == "skill":
            if eid not in best or it.get("support", 0) > best[eid].get("support", 0):
                best[eid] = it
    return [MarketItem(entity_id=g["skill_id"], name=g["name"], percent_change=best[g["skill_id"]].get("percent_change"),
                       mean_share_first=best[g["skill_id"]]["mean_share_first"],
                       mean_share_last=best[g["skill_id"]]["mean_share_last"])
            for g in gaps if g["skill_id"] in best]


def _skill_ref(sid: str, names: dict) -> SkillRef:
    return SkillRef(skill_id=sid, name=names.get(sid, sid))


def _assemble(engine: str, data: dict, feed: str, horizon: int, names: dict, warnings: list[str]) -> Report:
    validated = feed == "live"
    window = store.feed_window()
    banner = _banner(feed)
    disclaimer = store.disclaimer()
    resolved = data["resolved"]
    unresolved = [r.input for r in resolved if r.status == "unresolved"]
    caveats = [disclaimer, *BASE_CAVEATS]
    window = store.feed_window()
    if window["stale"]:
        caveats.append(f"Outlook figures are dated: made as of {window['forecast_as_of']} for "
                       f"{window['forecast_target_period']}. Data is {window['data_age_days']} days old, so they may "
                       "not describe current postings.")
    if validated:
        caveats.append("Outlook figures come from a validated model.")
    else:
        caveats.append("PREVIEW: outlook figures are unvalidated. The forecasting model did not pass its validation "
                       "gate, so they are shown only to demonstrate the pipeline. Do not use them for decisions.")
    if engine == "offline_alias":
        caveats.append("Matching engine: offline alias matcher (keyword rules against the taxonomy), not Qwen. "
                       "Short or common aliases can match ordinary words in free text.")
    else:
        caveats.append("Matching engine: the embeddings team's Qwen and pgvector matcher.")
    if horizon != FORECAST_HORIZON_MONTHS:
        warnings.append(f"Outlook figures exist only for a {FORECAST_HORIZON_MONTHS}-month horizon; "
                        f"{horizon} months shows no growth.")

    base = dict(engine=engine, feed=feed, validated=validated, banner=banner, disclaimer=disclaimer,
                horizon_months=horizon, caveats=caveats, warnings=warnings, resolved_skills=resolved,
                unresolved=unresolved, learning_path=LearningPath(note=LEARNING_NOTE, stages=[]), **window)
    if not data["have"]:
        return Report(**base, status="no_skills_resolved",
                      message="None of the skills could be matched to the taxonomy. Pick skills from the list, or "
                              "paste CV text that names skills directly.")

    jobs = data["jobs"]
    if not jobs:
        return Report(**base, status="no_overlap",
                      message="No job profiles are available in the job-skill graph, so no job can be matched.")
    if jobs[0]["coverage"] <= 0:
        return Report(**base, status="no_overlap",
                      message="None of your resolved skills appears in the skill profile of any job in our data.")

    within = [j for j in jobs if j["job_id"] in data["within_reach"]]
    shown = (within or jobs)[:BEST_FIT_SHOWN]
    targets = data["targets"]
    if within:
        status = "ok"
        message = (f"{len(within)} job(s) reach {REACH:.0%} coverage with a non-declining outlook. Gaps are ranked for "
                   f"the top {len(targets)}.")
    else:
        status = "nothing_within_reach"
        message = (f"No job reaches {REACH:.0%} coverage with a non-declining outlook. The closest jobs by coverage are "
                   "shown, and their gaps are used instead.")

    best_fit = [JobFit(job_id=j["job_id"], name=j["name"], coverage=round(j["coverage"], 4),
                       matched=[_skill_ref(s, names) for s in j["matched"]],
                       missing=[_skill_ref(s, names) for s in j["missing"]], growth=j["growth"],
                       forecast_status=j["forecast_status"], direction=j["direction"], confidence=j["confidence"],
                       signal_quality=j["signal_quality"], as_of=j.get("as_of"),
                       target_period=j.get("target_period"), shares_signal_with=j.get("shares_signal_with"))
                for j in shown]

    gaps = sorted(data["gaps"], key=lambda g: (-g["score"], -len(g["for_jobs"]), g["skill_id"]))
    stage_titles = _stages(gaps)
    stage_gaps = []
    for g, title in zip(gaps, stage_titles):
        stage_gaps.append(SkillGap(
            skill_id=g["skill_id"], name=g["name"], score=round(g["score"], 4), weight=round(g["weight"], 4),
            share=None if g["share"] is None else round(g["share"], 4), growth=g["growth"],
            forecast_status=g["forecast_status"], direction=g["direction"], confidence=g["confidence"],
            signal_quality=g["signal_quality"], for_jobs=g["for_jobs"], stage=title,
            as_of=g.get("as_of"), target_period=g.get("target_period"),
            shares_signal_with=g.get("shares_signal_with"), why=_why(g, names, validated)))
    stages: list[LearningStage] = []
    for title in dict.fromkeys(stage_titles):
        stages.append(LearningStage(title=title, skills=[SkillRef(skill_id=g.skill_id, name=g.name)
                                                         for g in stage_gaps if g.stage == title]))
    base["learning_path"] = LearningPath(note=LEARNING_NOTE, stages=stages)
    return Report(**base, status=status, message=message, best_fit_jobs=best_fit, targets=targets,
                  skill_gaps=stage_gaps, market_snapshot=_market(gaps, warnings))


def build_report(skills: list[str], text: str | None = None, horizon_months: int = FORECAST_HORIZON_MONTHS,
                 engine: str = "auto") -> dict:
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}")
    if len(skills) > MAX_SKILLS:
        raise ValueError(f"at most {MAX_SKILLS} skills")
    text = (text or "").strip()
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text must be at most {MAX_TEXT_CHARS} characters")
    warnings: list[str] = []
    items: list[str] = []
    for s in skills:
        s = (s or "").strip()
        if s and s.lower() not in {i.lower() for i in items}:
            items.append(s)

    vocab = T.load_vocabulary()
    names = dict(vocab.names)
    feed = store.serving_feed()
    blob = _blob_matches(text, vocab.skill_aliases) if text else {}

    data = None
    used = "offline_alias"
    if engine != "offline_alias":
        if os.environ.get("DATABASE_URL"):
            try:
                data = _qwen_data(items, blob, vocab, horizon_months, warnings)
                used = "qwen_pgvector"
            except Exception as exc:
                warnings.append(f"The Qwen/pgvector matcher failed ({type(exc).__name__}: {_short(exc)}); "
                                "the offline alias matcher was used instead.")
        elif engine == "qwen_pgvector":
            warnings.append("DATABASE_URL is not set, so the Qwen/pgvector matcher cannot run; the offline alias "
                            "matcher was used instead.")
    if data is None:
        data = _offline_data(items, blob, vocab, horizon_months, warnings)
        used = "offline_alias"
    if used == "qwen_pgvector":

        warnings.append("Growth values were read from the entity_forecast table and are not checked against the "
                        "validation gate here; the preview label reflects the local forecast store.")
    for g in data["gaps"]:
        names.setdefault(g["skill_id"], g["name"])
    report = _assemble(used, data, feed, horizon_months, names, warnings)
    return report.model_dump(mode="json")


def _short(exc: Exception) -> str:
    return re.sub(r"\s+", " ", str(exc))[:200]
