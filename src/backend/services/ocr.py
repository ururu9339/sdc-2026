from __future__ import annotations

import io
import re
import logging
from typing import Optional

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from src.backend.models.candidate import CandidateCreate

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-().]{7,}\d)")

_SECTION_HEADINGS = {
    "experience":     re.compile(r"(?i)(work experience|experience|employment history)"),
    "education":      re.compile(r"(?i)(education|academic background|qualifications)"),
    "skills":         re.compile(r"(?i)(skills|technical skills|competencies|expertise)"),
    "certifications": re.compile(r"(?i)(certifications?|licenses?|credentials?)"),
    "summary":        re.compile(r"(?i)(summary|profile|objective|about me)"),
    "languages":      re.compile(r"(?i)(languages?|language proficiency)"),
}


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR packages not installed. Run: pip install pytesseract pdf2image && brew install tesseract poppler")
    pages = convert_from_bytes(pdf_bytes, dpi=300)
    return "\n\n".join(pytesseract.image_to_string(p, lang="eng") for p in pages)


def extract_text_from_image(image_bytes: bytes) -> str:
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR packages not installed.")
    return pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)), lang="eng")


def _extract_email(text: str) -> Optional[str]:
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None


def _extract_phone(text: str) -> Optional[str]:
    m = _PHONE_RE.search(text)
    return m.group(0).strip() if m else None


def _extract_name(lines: list[str]) -> str:
    name_re = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$")
    for line in lines[:10]:
        line = line.strip()
        if line and name_re.match(line):
            return line
    for line in lines[:5]:
        line = line.strip()
        if line and "@" not in line:
            return line
    return "Unknown"


def _extract_section(text: str, heading_pattern: re.Pattern) -> Optional[str]:
    all_patterns = list(_SECTION_HEADINGS.values())
    lines = text.splitlines()
    in_section = False
    section_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if heading_pattern.match(stripped):
            in_section = True
            continue
        if in_section:
            if any(p.match(stripped) for p in all_patterns if p is not heading_pattern):
                break
            section_lines.append(line)

    content = "\n".join(section_lines).strip()
    return content if content else None


def normalise_to_candidate(raw_text: str, source: str = "ocr_pdf") -> CandidateCreate:
    lines = raw_text.splitlines()
    return CandidateCreate(
        full_name=_extract_name(lines),
        email=_extract_email(raw_text) or "unknown@unknown.com",
        phone=_extract_phone(raw_text),
        summary=_extract_section(raw_text, _SECTION_HEADINGS["summary"]),
        experience=_extract_section(raw_text, _SECTION_HEADINGS["experience"]),
        education=_extract_section(raw_text, _SECTION_HEADINGS["education"]),
        skills=_extract_section(raw_text, _SECTION_HEADINGS["skills"]),
        certifications=_extract_section(raw_text, _SECTION_HEADINGS["certifications"]),
        languages=_extract_section(raw_text, _SECTION_HEADINGS["languages"]),
        source=source,
    )
