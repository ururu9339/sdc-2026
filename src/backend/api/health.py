from __future__ import annotations

import importlib.util
import os

from fastapi import APIRouter

from src.backend.models.meta import BundleSummary, Health, HealthData, LastGate, MatchingEngines
from src.backend.services import forecast_store as store
from src.training import config as C
from src.pipeline.store import load_json

router = APIRouter()
API_VERSION = "v1"


def _bundle_summary() -> BundleSummary | None:
    manifest = load_json(C.SERVING_DIR / C.BUNDLE_MANIFEST, None)
    if not manifest:
        return None
    feed = manifest.get("feed")
    return BundleSummary(generated_at=manifest.get("generated_at"), as_of=manifest.get("as_of"), feed=feed,
                         model_version=manifest.get("model_version"), validated=manifest.get("validated"),
                         n_files=len(manifest.get("files", [])), total_bytes=manifest.get("total_bytes"),
                         note=manifest.get("note"))


def _qwen_available() -> bool:
    if not os.environ.get("DATABASE_URL"):
        return False
    try:
        return importlib.util.find_spec("psycopg") is not None and \
            importlib.util.find_spec("embeddings.match") is not None
    except (ImportError, ValueError):
        return False


@router.get(
    "/health",
    response_model=Health,
    operation_id="get_health",
    summary="Service health: what is served, from which feed, and which data layout",
    description="ready is true when forecasts or the no-model explanation can be served. feed is live, preview "
                "or none. data_layout is env, bundle or repo. bundle is the serving/ manifest summary when "
                "present.",
    tags=["meta"],
)
def get_health() -> Health:
    try:
        facts = store.health_facts()
        ready = True
    except Exception:
        facts = {"live_model": None, "feed": "none", "forecast_as_of": None, "forecast_age_days": None,
                 "stale": None, "n_forecasts": 0,
                 "gate_profile": C.ACTIVE_GATE_PROFILE, "last_gate": {"passed": None, "reason": "unavailable"},
                 "sources_with_data": []}
        ready = False
    data = HealthData(
        live_model=facts["live_model"], feed=facts["feed"], forecast_as_of=facts["forecast_as_of"],
        forecast_age_days=facts.get("forecast_age_days"), stale=facts.get("stale"),
        n_forecasts=facts["n_forecasts"], gate_profile=facts["gate_profile"],
        last_gate=LastGate(**facts["last_gate"]), data_layout=C.DATA_LAYOUT, bundle=_bundle_summary(),
        sources_with_data=facts["sources_with_data"],
        matching_engine_available=MatchingEngines(qwen_pgvector=_qwen_available()),
    )
    return Health(ready=ready, api_version=API_VERSION, data=data)
