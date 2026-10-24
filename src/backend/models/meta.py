"""Response shapes shared by every route: the error envelope and the health report."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None
    request_id: Optional[str] = None


class ErrorEnvelope(BaseModel):
    """Every non-2xx response has this shape. Tracebacks and request bodies are never included."""
    error: ErrorDetail


class BundleSummary(BaseModel):
    generated_at: Optional[str] = None
    as_of: Optional[str] = None
    feed: Optional[str] = None
    model_version: Optional[str] = None
    validated: Optional[bool] = None
    n_files: Optional[int] = None
    total_bytes: Optional[int] = None
    note: Optional[str] = None


class LastGate(BaseModel):
    passed: Optional[bool] = None
    reason: Optional[str] = None


class MatchingEngines(BaseModel):
    qwen_pgvector: bool
    offline_alias: bool = True


class HealthData(BaseModel):
    live_model: Optional[str] = None
    feed: Literal["live", "preview", "none"]
    forecast_as_of: Optional[str] = None
    n_forecasts: int
    gate_profile: str
    last_gate: LastGate
    data_layout: Literal["env", "bundle", "repo"]
    bundle: Optional[BundleSummary] = None
    sources_with_data: list[str]
    matching_engine_available: MatchingEngines


class Health(BaseModel):
    """ready is true when forecasts can be served, or when the no-model explanation can be served.
    It is false only when the stored files cannot be read. Always returned with HTTP 200."""
    status: Literal["ok"] = "ok"
    ready: bool
    api_version: str
    data: HealthData
