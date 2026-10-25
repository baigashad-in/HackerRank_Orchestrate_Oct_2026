"""Parser for PDF business documents (statements, support attachments, registers, journal extracts)."""

from __future__ import annotations

import subprocess
from typing import List, Tuple

from ..loaders import EvidenceFile
from .base import Observation, SnapshotScope
from .docfacts import observations_from_document_text


def extract_pdf_text(path: str) -> str:
    """Best-effort text extraction: pdfplumber, then pypdf, then the pdftotext CLI."""
    text = ""
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(path) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception:
        text = ""
    if len(text.strip()) < 20:
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(path)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            pass
    if len(text.strip()) < 20:
        try:
            text = subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True, timeout=60).stdout
        except Exception:
            pass
    return text or ""


def parse_pdf(ef: EvidenceFile, known_emails=None, aliases=None) -> Tuple[List[Observation], List[SnapshotScope]]:
    text = extract_pdf_text(ef.abs_path)
    return observations_from_document_text(ef, text, source="pdf", known_emails=known_emails, aliases=aliases), []
