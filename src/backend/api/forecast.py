from __future__ import annotations

from typing import Literal, Optional, Union

from fastapi import APIRouter, HTTPException, Query

from src.backend.models.forecast import (EntityForecast, Explanation, History, LiveForecastList, ModelInfo,
                                         NoLiveModel, PreviewForecastList, Snapshot, SourceInfo)
from src.backend.services import forecast_store as store

router = APIRouter(tags=["forecast"])

Kind = Literal["job", "skill"]
Direction = Literal["emerging", "declining", "stable"]
Status = Literal["ok", "insufficient_data"]


def _require_known(entity_id: str) -> None:
    if not store.entity_known(entity_id):
        raise HTTPException(status_code=404, detail="Entity not found in observations.")


@router.get("/forecast", response_model=Union[NoLiveModel, LiveForecastList, PreviewForecastList])
def list_forecasts(
    kind: Optional[Kind] = None,
    direction: Optional[Direction] = None,
    status: Optional[Status] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """feed 'live': validated forecasts. feed 'preview': no validated model exists, so unvalidated candidate
    forecasts are returned with status 'preview' and a banner. feed 'none': nothing to show."""
    feed = store.serving_feed()
    if feed == "none":
        return NoLiveModel(**store.no_live_envelope())
    rows = store.select_forecasts(kind, direction, status, limit)
    if feed == "preview":
        return PreviewForecastList(**store.preview_envelope(), count=len(rows), forecasts=rows)
    return LiveForecastList(**store.forecast_envelope(), count=len(rows), forecasts=rows)


@router.get("/forecast/{entity_id}", response_model=EntityForecast)
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


@router.get("/history/{entity_id}", response_model=History)
def get_history(entity_id: str, n_periods: int = Query(52, ge=1, le=520)):
    _require_known(entity_id)
    return store.history(entity_id, n_periods)


@router.get("/explain/{entity_id}", response_model=Explanation)
def get_explain(entity_id: str):
    _require_known(entity_id)
    return store.explain(entity_id)


@router.get("/snapshot", response_model=Snapshot)
def get_snapshot(kind: Optional[Kind] = None, limit: int = Query(50, ge=1, le=500)):
    """Descriptive market snapshot from the best-tier leading source that has data. Not a forecast."""
    return store.snapshot(kind, limit)


@router.get("/model", response_model=ModelInfo)
def get_model():
    return store.model_info()


@router.get("/sources", response_model=list[SourceInfo])
def get_sources():
    return store.sources_info()
