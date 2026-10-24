from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class CandidateBase(SQLModel):
    full_name: str = Field(max_length=255)
    email: str = Field(max_length=255)
    phone: Optional[str] = Field(default=None, max_length=50)
    location: Optional[str] = Field(default=None, max_length=255)
    summary: Optional[str] = Field(default=None)
    experience: Optional[str] = Field(default=None)
    education: Optional[str] = Field(default=None)
    skills: Optional[str] = Field(default=None)
    certifications: Optional[str] = Field(default=None)
    languages: Optional[str] = Field(default=None)
    source: str = Field(default="form", max_length=20)  # "form" | "ocr_pdf"


class Candidate(CandidateBase, table=True):
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CandidateCreate(CandidateBase):
    pass


class CandidateRead(CandidateBase):
    id: uuid.UUID
    created_at: datetime


class OCRResult(SQLModel):
    raw_text: str
    candidate: CandidateRead
