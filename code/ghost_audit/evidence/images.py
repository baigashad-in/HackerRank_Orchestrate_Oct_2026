"""Parser for application screenshots.

The default reader runs local OCR (tesseract through pytesseract) after light
preprocessing (grayscale, upscale, inversion for dark themes).  A different
reader, such as a vision model, can be injected: it only has to return the text
visible in the screenshot.
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Set, Tuple

from ..loaders import EvidenceFile
from .base import Observation, SnapshotScope
from .docfacts import observations_from_document_text

ReaderFn = Callable[[str], str]

_OCR_CACHE_DIRNAME = ".ghost_audit_cache"

# Where installers usually put tesseract when it is not on PATH.
_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/usr/bin/tesseract",
]

# Where screenshot text came from in this process (reported in usage_run.json).
READ_STATS: dict = {"model_fresh": 0, "model_cached": 0, "ocr_fresh": 0, "ocr_cached": 0, "failed": 0}

# Filled in by ocr_text(); read by the pipeline and by --doctor to explain empty screenshots.
OCR_STATUS: dict = {"checked": False, "available": None, "command": None, "reason": None,
                    "attempts": 0, "empty": 0}


def find_tesseract() -> Optional[str]:
    """TESSERACT_CMD, then PATH, then the usual install locations."""
    import shutil

    cmd = os.environ.get("TESSERACT_CMD", "").strip().strip('"')
    if cmd:
        return cmd if os.path.isfile(cmd) or shutil.which(cmd) else cmd   # report a bad TESSERACT_CMD as-is
    found = shutil.which("tesseract")
    if found:
        return found
    for cand in _TESSERACT_CANDIDATES:
        if cand and "%" not in cand and os.path.isfile(cand):
            return cand
    return None


def ocr_text(path: str) -> str:
    """OCR a screenshot.  Returns an empty string when OCR tooling is unavailable (reason in OCR_STATUS)."""
    OCR_STATUS["attempts"] += 1
    try:
        from PIL import Image, ImageOps  # type: ignore
        import pytesseract  # type: ignore
    except Exception as exc:
        OCR_STATUS.update(checked=True, available=False,
                          reason=f"Python packages missing ({exc.__class__.__name__}: {exc}); "
                                 "run: python -m pip install -r code/requirements.txt")
        OCR_STATUS["empty"] += 1
        return ""
    cmd = find_tesseract()
    OCR_STATUS["command"] = cmd
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    else:
        OCR_STATUS.update(checked=True, available=False,
                          reason="tesseract program not found (not on PATH, not in the usual install folders); "
                                 "install it or set TESSERACT_CMD to the full path of tesseract(.exe)")
        OCR_STATUS["empty"] += 1
        return ""
    try:
        im = Image.open(path).convert("L")
        # dark admin themes OCR badly; invert when the page is mostly dark
        hist = im.histogram()
        dark = sum(hist[:128])
        light = sum(hist[128:])
        if dark > light:
            im = ImageOps.invert(im)
        w, h = im.size
        im = im.resize((w * 2, h * 2))
        text = pytesseract.image_to_string(im, config="--psm 4")
        text_alt = pytesseract.image_to_string(im, config="--psm 3")
        # keep both passes: different page segmentation modes recover different rows
        out = text + "\n" + text_alt
        OCR_STATUS.update(checked=True, available=True)
        if not out.strip():
            OCR_STATUS["empty"] += 1
        return out
    except Exception as exc:
        OCR_STATUS.update(checked=True, available=False,
                          reason=f"tesseract failed ({exc.__class__.__name__}: {str(exc)[:200]}); command={cmd}")
        OCR_STATUS["empty"] += 1
        return ""


def _cache_path(image_path: str, tag: str) -> str:
    from .. import config

    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    return os.path.join(config.REPO_ROOT, _OCR_CACHE_DIRNAME, "ocr", f"{os.path.basename(image_path)}.{tag}.txt")


def _cache_read(path: str) -> Optional[str]:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            cached = fh.read()
        if cached.strip():            # an empty entry means an earlier failure: try again
            return cached
    return None


def _cache_write(path: str, text: str) -> None:
    if not text.strip():
        return                        # never cache a failure, so installing tesseract later takes effect
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError:
        pass


def cached_ocr(image_path: str) -> str:
    """Local OCR with its own cache entry (tag 'ocr')."""
    path = _cache_path(image_path, "ocr")
    hit = _cache_read(path)
    if hit is not None:
        READ_STATS["ocr_cached"] += 1
        return hit
    text = ocr_text(image_path)
    READ_STATS["ocr_fresh" if text.strip() else "failed"] += 1
    _cache_write(path, text)
    return text


def cached_text(ef: EvidenceFile, reader: Optional[ReaderFn]) -> str:
    """Text of a screenshot, cached under <repo>/.ghost_audit_cache/ocr/ per reader.

    A model reader that had to fall back to OCR sets ``last_from_model = False``; that text is then
    not stored under the model's tag, so a later run with a working key still asks the model.
    """
    if reader is None:
        return cached_ocr(ef.abs_path)
    path = _cache_path(ef.abs_path, getattr(reader, "cache_tag", None) or "reader")
    hit = _cache_read(path)
    if hit is not None:
        READ_STATS["model_cached"] += 1
        return hit
    text = reader(ef.abs_path)
    if getattr(reader, "last_from_model", True):
        READ_STATS["model_fresh" if text.strip() else "failed"] += 1
        _cache_write(path, text)
    return text


_ID_WORDS = r"(account|search|analytics|support|export|data[ _]?flow|backup|audit[ _]?record|tombstone|marketing|legal[ _]?hold|object|user)"
_ID_FIX_RE = re.compile(r"\b" + _ID_WORDS + r"[ _.\-]?(\d{2,6})\b", re.I)


def normalise_ocr_text(text: str) -> str:
    """Repair the identifier spellings OCR most often breaks (lost underscores, dots)."""

    def fix(m: "re.Match[str]") -> str:
        word = re.sub(r"[ \-.]", "_", m.group(1).lower())
        return f"{word}_{m.group(2)}"

    text = _ID_FIX_RE.sub(fix, text)
    text = text.replace("@gmall.com", "@gmail.com").replace("@gmai1.com", "@gmail.com")
    return text


def parse_image(ef: EvidenceFile, reader: Optional[ReaderFn] = None,
                known_emails: Optional[Set[str]] = None, aliases=None) -> Tuple[List[Observation], List[SnapshotScope]]:
    text = cached_text(ef, reader)
    if not text.strip():
        return [], []
    text = normalise_ocr_text(text)
    return observations_from_document_text(ef, text, source="image", known_emails=known_emails, aliases=aliases), []
