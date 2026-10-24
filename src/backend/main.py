"""The HTTP API. Every public route is served under /api/v1 (the schema documents only these). The same routes are
also served unprefixed as hidden aliases for older clients.

Run:   uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
Env:   CORS_ORIGINS (comma-separated; default '*'), DATABASE_URL (optional; enables the Qwen matcher),
       SDC_DATA_DIR / SDC_MODELS_DIR (optional; see src/training/config.py resolve_layout).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.api import health as health_api
from src.backend.api.candidates import router as candidates_router
from src.backend.api.forecast import router as forecast_router
from src.backend.api.report import router as report_router
from src.backend.db import create_db_and_tables
from src.backend.errors import install_error_handlers
from src.backend.middleware import REQUEST_ID_HEADER, ETagMiddleware, RequestLogMiddleware, cors_origins

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_VERSION = "0.2.0"
API_PREFIX = "/api/v1"
OPENAPI_TAGS = [
    {"name": "forecasts", "description": "Forecasts and explanations for jobs and skills. Preview rows are "
                                         "labelled validated=false."},
    {"name": "history", "description": "Past share of postings (descriptive)."},
    {"name": "market", "description": "Descriptive snapshot of past shares. Not a forecast."},
    {"name": "model", "description": "Live model, gate verdicts and audit."},
    {"name": "sources", "description": "Data sources, tiers, licences and coverage."},
    {"name": "report", "description": "CV-to-report: matches, gaps, learning order and outlook. Never cached."},
    {"name": "taxonomy", "description": "Skill and job vocabulary."},
    {"name": "candidates", "description": "Candidate records (form and OCR upload). Personal data."},
    {"name": "meta", "description": "Service health."},
]
DESCRIPTION = (
    "Forecasts of job and skill demand in Malaysia. **No model is validated yet**: while the gate rejects every "
    "candidate, `/forecast` serves an unvalidated PREVIEW (`validated: false`, with a banner). Always show the "
    "banner when `validated` is false. See docs/API_FOR_FRONTEND.md for rendering rules and error shapes.\n\n"
    f"Errors share one shape: `{{\"error\": {{\"code\", \"message\", \"details\", \"request_id\"}}}}`. "
    f"Each response carries an `{REQUEST_ID_HEADER}` header."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()  # the candidates database; created at server start, never at import time
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="SDC-2026 API", version=APP_VERSION, description=DESCRIPTION, openapi_tags=OPENAPI_TAGS,
                  lifespan=lifespan)
    # Outermost first: CORS, then request id and logging, then the ETag rule, then the routes.
    app.add_middleware(ETagMiddleware)
    app.add_middleware(RequestLogMiddleware)
    origins = cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],  # browsers reject credentials with a wildcard origin
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER, "ETag"],
    )
    install_error_handlers(app)

    public = APIRouter()
    public.include_router(forecast_router)
    public.include_router(report_router)
    public.include_router(health_api.router)
    app.include_router(public, prefix=API_PREFIX)
    app.include_router(public, include_in_schema=False)  # unprefixed aliases, hidden from the schema
    app.include_router(candidates_router)  # already /api/v1/candidates
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=8000, reload=True)
