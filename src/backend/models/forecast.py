"""Response models for the read-only forecast API. Values are stored records; nothing is computed here."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ForecastRecord(BaseModel):
    forecast_id: str
    model_version: Optional[str] = None
    as_of: str
    target_period: Optional[str] = None
    horizon_periods: Optional[int] = None
    entity_id: str
    kind: Optional[str] = None
    status: Literal["ok", "insufficient_data"]
    reason: Optional[str] = None
    p50: Optional[float] = None
    p10: Optional[float] = None
    p90: Optional[float] = None
    direction: Optional[Literal["emerging", "declining", "stable"]] = None
    confidence: Optional[str] = None
    signal_quality: Optional[str] = None
    dispersion_ratio: Optional[float] = None
    gate_profile: Optional[str] = None
    source_contributions: Optional[dict[str, float]] = None
    baseline_persistence: Optional[float] = None
    baseline_trend: Optional[float] = None
    disclaimer: Optional[str] = None
    logged_at: Optional[str] = None
    hash: Optional[str] = None


class GateSummary(BaseModel):
    created_at: Optional[str] = None
    as_of: Optional[str] = None
    version: Optional[str] = None
    status: Optional[str] = None
    passed: Optional[bool] = None
    reason: Optional[str] = None
    profile: Optional[str] = None
    checks: Optional[dict[str, bool]] = None
    n_folds: Optional[int] = None
    model_mae: Optional[float] = None
    best_baseline: Optional[str] = None
    best_baseline_mae: Optional[float] = None
    improvement_pct: Optional[float] = None
    fold_win_rate: Optional[float] = None
    coverage: Optional[float] = None


class LiveForecastList(BaseModel):
    status: Literal["ok"] = "ok"
    model_version: str
    gate_profile: Optional[str] = None
    as_of: Optional[str] = None
    target_period: Optional[str] = None
    horizon_periods: Optional[int] = None
    disclaimer: Optional[str] = None
    count: int
    forecasts: list[ForecastRecord]


class NoLiveModel(BaseModel):
    status: Literal["no_live_model"] = "no_live_model"
    reason: str
    gate: Optional[GateSummary] = None
    disclaimer: Optional[str] = None
    forecasts: list[ForecastRecord] = Field(default_factory=list)


class EntityForecast(BaseModel):
    entity_id: str
    kind: Optional[str] = None
    status: Literal["ok", "insufficient_data", "no_live_model"]
    reason: Optional[str] = None
    forecast: Optional[ForecastRecord] = None
    disclaimer: Optional[str] = None


class HistoryPoint(BaseModel):
    period: str
    value: float
    support: Optional[float] = None


class AuditRow(BaseModel):
    source_id: str
    entity_id: str
    kind: Optional[str] = None
    n_periods: Optional[int] = None
    first_period: Optional[str] = None
    last_period: Optional[str] = None
    mean_share: Optional[float] = None
    dispersion_ratio: Optional[float] = None
    abs_corr_with_time: Optional[float] = None
    noise_like: Optional[bool] = None


class History(BaseModel):
    entity_id: str
    kind: str
    source_id: str
    granularity: str
    n_periods_returned: int
    series: list[HistoryPoint]
    audit: Optional[AuditRow] = None


class Neighbour(BaseModel):
    entity_id: str
    kind: str
    weight: float
    n_cooccur: int


class Drivers(BaseModel):
    source_contributions: Optional[dict[str, float]] = None
    source_contributions_units: str
    related: list[Neighbour]


class Explanation(BaseModel):
    entity_id: str
    kind: Optional[str] = None
    status: Literal["ok", "insufficient_data", "no_live_model"]
    summary: str
    drivers: Drivers
    caveats: list[str]


class SourceInfo(BaseModel):
    source_id: str
    name: Optional[str] = None
    role: Optional[str] = None
    tier: Optional[int] = None
    reliability: Optional[float] = None
    granularity: Optional[str] = None
    enabled: bool
    license: Optional[str] = None
    attribution: Optional[str] = None
    source_url: Optional[str] = None
    has_data: bool
    rows: int
    periods: Optional[int] = None
    overlap_periods: Optional[int] = None
    first_period: Optional[str] = None
    last_period: Optional[str] = None


class LiveModel(BaseModel):
    version: str
    gate_profile: Optional[str] = None
    trained_as_of: Optional[str] = None
    horizon_periods: Optional[int] = None
    n_feature_columns: Optional[int] = None
    sources_used: Optional[list[str]] = None
    source_importance: Optional[dict[str, float]] = None
    gate: Optional[GateSummary] = None


class ModelInfo(BaseModel):
    live: Optional[LiveModel] = None
    live_version: Optional[str] = None
    active_gate_profile: str
    attempts: int
    latest_attempt: Optional[GateSummary] = None
    audit_summary: Optional[list[dict[str, Any]]] = None
    timeline: Optional[dict[str, Any]] = None
    report_generated_at: Optional[str] = None
    disclaimer: Optional[str] = None
