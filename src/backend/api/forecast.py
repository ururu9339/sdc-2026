from __future__ import annotations

from typing import Literal, Optional, Union

from fastapi import APIRouter, HTTPException, Query

from src.backend.models.forecast import (EntityForecast, Explanation, History, LiveForecastList, ModelInfo,
                                         NoLiveModel, SourceInfo)
from src.backend.services import forecast_store as store

router = APIRouter(tags=["forecast"])

Kind = Literal["job", "skill"]
Direction = Literal["emerging", "declining", "stable"]
Status = Literal["ok", "insufficient_data"]


def _require_known(entity_id: str) -> None:
    if not store.entity_known(entity_id):
        raise HTTPException(status_code=404, detail="Entity not found in observations.")


@router.get("/forecast", response_model=Union[NoLiveModel, LiveForecastList])
def list_forecasts(
    kind: Optional[Kind] = None,
    direction: Optional[Direction] = None,
    status: Optional[Status] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    if store.live_version() is None:
        return NoLiveModel(**store.no_live_envelope())
    rows = store.select_forecasts(kind, direction, status, limit)
    return LiveForecastList(**store.forecast_envelope(), count=len(rows), forecasts=rows)


@router.get("/forecast/{entity_id}", response_model=EntityForecast)
def get_forecast(entity_id: str):
    _require_known(entity_id)
    kind = store.entity_kind(entity_id)
    disclaimer = store.disclaimer()
    if store.live_version() is None:
        return EntityForecast(entity_id=entity_id, kind=kind, status="no_live_model",
                              reason=store.no_live_reason(), disclaimer=disclaimer)
    rec = store.forecast_for(entity_id)
    if rec is None:
        return EntityForecast(entity_id=entity_id, kind=kind, status="insufficient_data",
                              reason="no forecast has been logged for this entity yet", disclaimer=disclaimer)
    return EntityForecast(entity_id=entity_id, kind=kind, status=rec["status"], reason=rec.get("reason"),
                          forecast=rec, disclaimer=disclaimer)


@router.get("/history/{entity_id}", response_model=History)
def get_history(entity_id: str, n_periods: int = Query(52, ge=1, le=520)):
    _require_known(entity_id)
    return store.history(entity_id, n_periods)


@router.get("/explain/{entity_id}", response_model=Explanation)
def get_explain(entity_id: str):
    _require_known(entity_id)
    return store.explain(entity_id)


@router.get("/model", response_model=ModelInfo)
def get_model():
    return store.model_info()


@router.get("/sources", response_model=list[SourceInfo])
def get_sources():
    return store.sources_info()
