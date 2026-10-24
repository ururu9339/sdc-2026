"""Forecast, history, explanation, snapshot, model and source routes. Every forecast-bearing response carries
the feed and the validated flag; while no model has passed the gate, the preview banner is always present."""
from __future__ import annotations

from typing import Literal, Optional, Union

from fastapi import APIRouter, Depends, Query

from src.backend.api.common import strict_query
from src.backend.errors import ERROR_RESPONSES, ApiError
from src.backend.models.forecast import (EntityForecast, Explanation, History, LiveForecastList, ModelInfo,
                                         NoLiveModel, PreviewForecastList, Snapshot, SourceInfo)
from src.backend.services import forecast_store as store

router = APIRouter()

Kind = Literal["job", "skill"]
Direction = Literal["emerging", "declining", "stable"]
Status = Literal["ok", "insufficient_data"]
Confidence = Literal["low", "moderate", "high"]
Sort = Literal["growth", "-growth", "name", "-name"]

FORECAST_PARAMS = {"kind", "direction", "status", "min_confidence", "q", "sort", "offset", "limit"}
SNAPSHOT_PARAMS = {"kind", "limit"}


def _require_known(entity_id: str) -> None:
    if not store.entity_known(entity_id):
        raise ApiError(404, "Entity not found in observations.")


@router.get(
    "/forecast",
    response_model=Union[NoLiveModel, LiveForecastList, PreviewForecastList],
    operation_id="list_forecasts",
    summary="List forecasts for the current feed",
    description=(
        "Newest forecast per entity from the current feed. feed 'live': validated forecasts from a model that "
        "passed the gate. feed 'preview': unvalidated candidate forecasts (validated=false, banner set, never "
        "hide it). feed 'none': no model and no preview; the response explains why and carries no forecasts.\n\n"
        "Sorting: growth sorts by the median log change (p50) ascending, -growth descending (default: largest "
        "rises first). Entities with no number (insufficient_data) always come last. name and -name sort A-Z and "
        "Z-A by taxonomy name. q matches id or name, case-insensitively. min_confidence keeps records at or above "
        "low < moderate < high. Paging: offset and limit (max 500); total is the number of matches before paging, "
        "count the number on this page."),
    tags=["forecasts"],
    dependencies=[Depends(strict_query(FORECAST_PARAMS))],
    responses=ERROR_RESPONSES,
)
def list_forecasts(
    kind: Optional[Kind] = Query(None, description="Restrict to jobs or skills."),
    direction: Optional[Direction] = Query(None, description="emerging, declining or stable."),
    status: Optional[Status] = Query(None, description="ok (has a number) or insufficient_data."),
    min_confidence: Optional[Confidence] = Query(None, description="Lowest confidence to include."),
    q: Optional[str] = Query(None, max_length=100, description="Case-insensitive match on id or name."),
    sort: Sort = Query(store.DEFAULT_SORT, description="growth, -growth (default), name or -name."),
    offset: int = Query(0, ge=0, le=1_000_000, description="Records to skip."),
    limit: int = Query(100, ge=1, le=500, description="Records per page (max 500)."),
):
    feed = store.serving_feed()
    if feed == "none":
        return NoLiveModel(**store.no_live_envelope(), total=0, offset=offset, limit=limit)
    feed, total, rows = store.query_forecasts(kind=kind, direction=direction, status=status,
                                              min_confidence=min_confidence, q=q, sort=sort, offset=offset,
                                              limit=limit)
    if feed == "preview":
        return PreviewForecastList(**store.preview_envelope(), count=len(rows), total=total, offset=offset,
                                   limit=limit, forecasts=rows)
    return LiveForecastList(**store.forecast_envelope(), count=len(rows), total=total, offset=offset, limit=limit,
                            forecasts=rows)


@router.get(
    "/forecast/{entity_id}",
    response_model=EntityForecast,
    operation_id="get_forecast",
    summary="Forecast for one entity",
    description="Newest record for the entity in the current feed. Unknown ids give 404; known ids with no "
                "number give status insufficient_data with a reason.",
    tags=["forecasts"],
    responses=ERROR_RESPONSES,
)
def get_forecast(entity_id: str):
    _require_known(entity_id)
    kind = store.entity_kind(entity_id)
    disclaimer = store.disclaimer()
    feed, rec = store.record_for(entity_id)
    if feed == "none":
        return EntityForecast(entity_id=entity_id, kind=kind, status="no_live_model", feed="none", validated=False,
                              reason=store.no_live_reason(), disclaimer=disclaimer)
    validated = feed == "live"
    banner = None if validated else store.PREVIEW_BANNER.format(reason=(rec or {}).get("preview_reason")
                                                                or store.no_live_reason())
    if rec is None:
        return EntityForecast(entity_id=entity_id, kind=kind, status="insufficient_data", feed=feed,
                              validated=validated, banner=banner, disclaimer=disclaimer,
                              reason="no forecast has been logged for this entity yet")
    status = rec["status"] if validated else ("preview" if rec["status"] == "ok" else "insufficient_data")
    return EntityForecast(entity_id=entity_id, kind=kind, status=status, feed=feed, validated=validated,
                          banner=banner, reason=rec.get("reason"), forecast=store.api_record(rec, feed),
                          disclaimer=disclaimer)


@router.get(
    "/history/{entity_id}",
    response_model=History,
    operation_id="get_history",
    summary="Past share of postings for one entity",
    description="Descriptive history of the target source (share of postings), latest n_periods in the source's "
                "own granularity, with the entity's signal audit row. Not a forecast.",
    tags=["history"],
    responses=ERROR_RESPONSES,
)
def get_history(entity_id: str, n_periods: int = Query(52, ge=1, le=520, description="Latest periods (max 520).")):
    _require_known(entity_id)
    return store.history(entity_id, n_periods)


@router.get(
    "/explain/{entity_id}",
    response_model=Explanation,
    operation_id="get_explanation",
    summary="Plain-language explanation with drivers and caveats",
    description="Template text filled only from stored fields. Every number in the summary comes from the stored "
                "forecast; with no live model or preview the summary states that no number is issued.",
    tags=["forecasts"],
    responses=ERROR_RESPONSES,
)
def get_explain(entity_id: str):
    _require_known(entity_id)
    return store.explain(entity_id)


@router.get(
    "/snapshot",
    response_model=Snapshot,
    operation_id="get_snapshot",
    summary="Descriptive market snapshot (not a forecast)",
    description="Average share of postings in the first and last contiguous block of periods of the best-tier "
                "leading source. Every item has type descriptive_not_forecast. No trend is inferred.",
    tags=["market"],
    dependencies=[Depends(strict_query(SNAPSHOT_PARAMS))],
    responses=ERROR_RESPONSES,
)
def get_snapshot(kind: Optional[Kind] = Query(None), limit: int = Query(50, ge=1, le=500)):
    return store.snapshot(kind, limit)


@router.get(
    "/model",
    response_model=ModelInfo,
    operation_id="get_model",
    summary="Live model, gate verdicts and audit",
    description="The live model (null while none has passed the gate), the latest gate attempt, the audit "
                "summary and the source timeline.",
    tags=["model"],
)
def get_model():
    return store.model_info()


@router.get(
    "/sources",
    response_model=list[SourceInfo],
    operation_id="list_sources",
    summary="Data sources, tiers, licences and coverage",
    tags=["sources"],
)
def get_sources():
    return store.sources_info()
