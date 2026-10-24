from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

Engine = Literal["qwen_pgvector", "offline_alias"]
Feed = Literal["live", "preview", "none"]
Direction = Literal["emerging", "declining", "stable"]
ForecastStatus = Literal["ok", "insufficient_data", "no_forecast"]
ReportStatus = Literal["ok", "nothing_within_reach", "no_overlap", "no_skills_resolved"]


class SkillRef(BaseModel):
    skill_id: str
    name: str


class ResolvedSkill(BaseModel):
    input: str
    skill_id: Optional[str] = None
    name: Optional[str] = None
    status: Literal["matched", "unresolved"]
    method: str


class JobFit(BaseModel):
    job_id: str
    name: str
    coverage: float
    matched: list[SkillRef]
    missing: list[SkillRef]
    growth: Optional[float] = None
    forecast_status: ForecastStatus
    direction: Optional[Direction] = None
    confidence: Optional[str] = None
    signal_quality: Optional[str] = None
    as_of: Optional[str] = None
    target_period: Optional[str] = None
    shares_signal_with: Optional[str] = None


class SkillGap(BaseModel):
    skill_id: str
    name: str
    score: float
    weight: float
    share: Optional[float] = None
    growth: Optional[float] = None
    forecast_status: ForecastStatus
    direction: Optional[Direction] = None
    confidence: Optional[str] = None
    signal_quality: Optional[str] = None
    for_jobs: list[str]
    stage: str
    as_of: Optional[str] = None
    target_period: Optional[str] = None
    shares_signal_with: Optional[str] = None
    why: str


class LearningStage(BaseModel):
    title: str
    skills: list[SkillRef]


class LearningPath(BaseModel):
    note: str
    stages: list[LearningStage]


class MarketItem(BaseModel):
    entity_id: str
    name: str
    percent_change: Optional[float] = None
    mean_share_first: float
    mean_share_last: float
    type: Literal["descriptive_not_forecast"] = "descriptive_not_forecast"


class Report(BaseModel):
    engine: Engine
    feed: Feed
    validated: bool
    banner: Optional[str] = None
    disclaimer: str
    horizon_months: int
    status: ReportStatus
    message: Optional[str] = None
    resolved_skills: list[ResolvedSkill] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    best_fit_jobs: list[JobFit] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    skill_gaps: list[SkillGap] = Field(default_factory=list)
    learning_path: LearningPath
    market_snapshot: list[MarketItem] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    forecast_as_of: Optional[str] = None
    forecast_target_period: Optional[str] = None
    data_age_days: Optional[int] = None
    stale: Optional[bool] = None
    forecast_window: Optional[str] = None


class ReportRequest(BaseModel):
    skills: list[Annotated[str, Field(max_length=200)]] = Field(default_factory=list, max_length=100)
    text: Optional[str] = Field(default=None, max_length=20000)
    horizon_months: int = Field(default=3, ge=1, le=12)
    engine: Literal["auto", "qwen_pgvector", "offline_alias"] = "auto"


class TaxonomyEntry(BaseModel):
    id: str
    name: str
    alt_labels: list[str]


class TaxonomyPage(BaseModel):
    kind: Literal["skill", "job"]
    count: int
    total: int
    offset: int
    limit: int
    items: list[TaxonomyEntry]
