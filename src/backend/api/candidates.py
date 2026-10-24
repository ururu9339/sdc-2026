from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlmodel import Session, select

from src.backend.db import get_session
from src.backend.errors import ApiError
from src.backend.models.candidate import Candidate, CandidateCreate, CandidateRead, OCRResult
from src.backend.models.meta import ErrorEnvelope
from src.backend.services.ocr import extract_text_from_image, extract_text_from_pdf, normalise_to_candidate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/candidates", tags=["candidates"])
_ERRORS = {404: {"model": ErrorEnvelope, "description": "Candidate not found."},
           415: {"model": ErrorEnvelope, "description": "Unsupported file type."},
           422: {"model": ErrorEnvelope, "description": "Validation or OCR error."},
           503: {"model": ErrorEnvelope, "description": "OCR engine not installed."}}

_ALLOWED_MIME = {"application/pdf", "image/png", "image/jpeg", "image/jpg", "image/tiff"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post("/form", response_model=CandidateRead, status_code=status.HTTP_201_CREATED,
             operation_id="create_candidate_form", summary="Create a candidate from a form", responses=_ERRORS)
def submit_form(payload: CandidateCreate, session: Session = Depends(get_session)) -> CandidateRead:
    payload.source = "form"
    db_candidate = Candidate.model_validate(payload)
    session.add(db_candidate)
    session.commit()
    session.refresh(db_candidate)
    return db_candidate


@router.post("/pdf", response_model=OCRResult, status_code=status.HTTP_201_CREATED,
             operation_id="create_candidate_pdf", summary="Create a candidate from a PDF or image (OCR)",
             responses=_ERRORS)
async def submit_pdf(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> OCRResult:
    if file.content_type not in _ALLOWED_MIME:
        raise ApiError(415, f"Unsupported file type: {file.content_type}")

    file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not file_bytes:
        raise ApiError(400, "File is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "File is larger than 10 MB.")

    try:
        if file.content_type == "application/pdf":
            raw_text = extract_text_from_pdf(file_bytes)
        else:
            raw_text = extract_text_from_image(file_bytes)
    except RuntimeError as exc:
        raise ApiError(503, str(exc)) from exc
    except Exception as exc:
        logger.exception("OCR failed for %s", file.filename)
        raise ApiError(422, f"OCR error: {exc}") from exc

    candidate_data = normalise_to_candidate(raw_text, source="ocr_pdf")
    db_candidate = Candidate.model_validate(candidate_data)
    session.add(db_candidate)
    session.commit()
    session.refresh(db_candidate)

    return OCRResult(raw_text=raw_text, candidate=CandidateRead.model_validate(db_candidate))


@router.get("/", response_model=list[CandidateRead], operation_id="list_candidates",
            summary="List candidates", responses=_ERRORS)
def list_candidates(skip: int = 0, limit: int = 50, session: Session = Depends(get_session)):
    return session.exec(select(Candidate).offset(skip).limit(limit)).all()


@router.get("/{candidate_id}", response_model=CandidateRead, operation_id="get_candidate",
            summary="Get one candidate", responses=_ERRORS)
def get_candidate(candidate_id: UUID, session: Session = Depends(get_session)):
    candidate = session.get(Candidate, candidate_id)
    if not candidate:
        raise ApiError(404, "Candidate not found.")
    return candidate
