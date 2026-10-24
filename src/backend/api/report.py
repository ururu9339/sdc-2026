from __future__ import annotations

import re
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Query, Response
from sqlmodel import Session

from src.backend.api.common import strict_query
from src.backend.db import get_session
from src.backend.errors import ERROR_RESPONSES, ApiError
from src.backend.models.candidate import Candidate
from src.backend.models.report import Report, ReportRequest, TaxonomyPage
from src.backend.services import catalog
from src.backend.services import report as service

router = APIRouter()

_SPLIT = re.compile(r"[,;\n]+")
_ITEM_MAX = 200
TAXONOMY_PARAMS = {"q", "limit", "offset"}

REPORT_EXAMPLES = {
    "skills_only": {
        "summary": "Taxonomy ids",
        "description": "Skills as taxonomy ids. No CV text.",
        "value": {"skills": ["skill:python", "skill:sql", "skill:excel", "skill:data_analysis"],
                  "horizon_months": 3, "engine": "auto"},
    },
    "free_text": {
        "summary": "Free-text skills and CV text",
        "description": "Free-text skill names plus a short CV blob. Short aliases (under 3 characters) are not "
                       "matched in free text.",
        "value": {"skills": ["Accounting", "Excel"],
                  "text": "Prepared monthly management accounts and VAT returns; used pivot tables daily.",
                  "horizon_months": 3, "engine": "offline_alias"},
    },
}


def _candidate_skills(raw: str | None) -> tuple[list[str], int]:
    items = [s.strip()[:_ITEM_MAX] for s in _SPLIT.split(raw or "") if s.strip()]
    unique = list(dict.fromkeys(items))
    return unique, len(unique)


@router.post(
    "/report",
    response_model=Report,
    operation_id="create_report",
    summary="Skills (and optional CV text) to matches, gaps, learning order and outlook",
    description=("Resolves skills (taxonomy ids or free text) and returns best-fit jobs, skill gaps, a learning "
                 "order and a descriptive market snapshot. Outlook figures carry their feed: while no model is "
                 "validated they are an unvalidated preview (validated=false, banner set). status values: ok, "
                 "nothing_within_reach, no_overlap, no_skills_resolved. Responses are never cached."),
    tags=["report"],
    responses=ERROR_RESPONSES,
)
def create_report(
    response: Response,
    body: ReportRequest = Body(..., openapi_examples=REPORT_EXAMPLES),
):
    response.headers["Cache-Control"] = "no-store"
    try:
        return service.build_report(body.skills, text=body.text, horizon_months=body.horizon_months,
                                    engine=body.engine)
    except ValueError as exc:
        raise ApiError(422, str(exc)) from exc


@router.get(
    "/report/candidate/{candidate_id}",
    response_model=Report,
    operation_id="report_for_candidate",
    summary="Report built from a stored candidate",
    description="Skills are split on commas, semicolons and new lines; experience and summary text are used as "
                "the free-text blob. Personal data: responses are never cached.",
    tags=["report"],
    responses=ERROR_RESPONSES,
)
def report_for_candidate(
    response: Response,
    candidate_id: UUID,
    horizon_months: int = Query(3, ge=1, le=12),
    engine: str = Query("auto", pattern="^(auto|qwen_pgvector|offline_alias)$"),
    session: Session = Depends(get_session),
):
    response.headers["Cache-Control"] = "no-store"
    candidate = session.get(Candidate, candidate_id)
    if candidate is None:
        raise ApiError(404, "Candidate not found.")
    skills, total = _candidate_skills(candidate.skills)
    notes = []
    if total > service.MAX_SKILLS:
        skills = skills[:service.MAX_SKILLS]
        notes.append(f"Only the first {service.MAX_SKILLS} of {total} listed skills were used.")
    text = "\n".join(t for t in (candidate.experience, candidate.summary) if t)[:service.MAX_TEXT_CHARS]
    try:
        out = service.build_report(skills, text=text, horizon_months=horizon_months, engine=engine)
    except ValueError as exc:
        raise ApiError(422, str(exc)) from exc
    out["warnings"] = [*notes, *out["warnings"]]
    return out


def _taxonomy_list(kind: str, q: str | None, limit: int, offset: int) -> dict:
    return catalog.list_taxonomy(kind, q=q, offset=offset, limit=limit)


@router.get(
    "/taxonomy/skills",
    response_model=TaxonomyPage,
    operation_id="list_taxonomy_skills",
    summary="Skill vocabulary (taxonomy ids, names, alternative labels)",
    description="Sorted A to Z by name. q matches id, name or an alternative label. total counts all matches; "
                "count is the page size.",
    tags=["taxonomy"],
    dependencies=[Depends(strict_query(TAXONOMY_PARAMS))],
    responses=ERROR_RESPONSES,
)
def list_taxonomy_skills(
    q: str | None = Query(None, max_length=100),
    limit: int = Query(500, ge=1, le=500),
    offset: int = Query(0, ge=0, le=1_000_000),
):
    return _taxonomy_list("skill", q, limit, offset)


@router.get(
    "/taxonomy/jobs",
    response_model=TaxonomyPage,
    operation_id="list_taxonomy_jobs",
    summary="Job vocabulary (taxonomy ids, names, alternative labels)",
    description="Sorted A to Z by name. q matches id, name or an alternative label. total counts all matches; "
                "count is the page size.",
    tags=["taxonomy"],
    dependencies=[Depends(strict_query(TAXONOMY_PARAMS))],
    responses=ERROR_RESPONSES,
)
def list_taxonomy_jobs(
    q: str | None = Query(None, max_length=100),
    limit: int = Query(500, ge=1, le=500),
    offset: int = Query(0, ge=0, le=1_000_000),
):
    return _taxonomy_list("job", q, limit, offset)
