from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlmodel import Session, select

from src.backend.db import get_session
from src.backend.models.candidate import Candidate, CandidateCreate, CandidateRead, OCRResult
from src.backend.services.ocr import extract_text_from_image, extract_text_from_pdf, normalise_to_candidate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/candidates", tags=["candidates"])

_ALLOWED_MIME = {"application/pdf", "image/png", "image/jpeg", "image/jpg", "image/tiff"}


@router.post("/form", response_model=CandidateRead, status_code=status.HTTP_201_CREATED)
def submit_form(payload: CandidateCreate, session: Session = Depends(get_session)) -> CandidateRead:
    payload.source = "form"
    db_candidate = Candidate.model_validate(payload)
    session.add(db_candidate)
    session.commit()
    session.refresh(db_candidate)
    return db_candidate  # type: ignore[return-value]


@router.post("/pdf", response_model=OCRResult, status_code=status.HTTP_201_CREATED)
async def submit_pdf(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> OCRResult:
    if file.content_type not in _ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        if file.content_type == "application/pdf":
            raw_text = extract_text_from_pdf(file_bytes)
        else:
            raw_text = extract_text_from_image(file_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("OCR failed for %s", file.filename)
        raise HTTPException(status_code=422, detail=f"OCR error: {exc}") from exc

    candidate_data = normalise_to_candidate(raw_text, source="ocr_pdf")
    db_candidate = Candidate.model_validate(candidate_data)
    session.add(db_candidate)
    session.commit()
    session.refresh(db_candidate)

    return OCRResult(raw_text=raw_text, candidate=CandidateRead.model_validate(db_candidate))


@router.get("/", response_model=list[CandidateRead])
def list_candidates(skip: int = 0, limit: int = 50, session: Session = Depends(get_session)):
    return session.exec(select(Candidate).offset(skip).limit(limit)).all()


@router.get("/{candidate_id}", response_model=CandidateRead)
def get_candidate(candidate_id: UUID, session: Session = Depends(get_session)):
    candidate = session.get(Candidate, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return candidate
