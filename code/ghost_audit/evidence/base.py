"""Observation model and the dispatcher that parses every evidence file."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from .. import config
from ..loaders import EvidenceFile

NULLISH = {None, "", "null", "none", "nil", "-", "n/a", "[redacted]", "no direct identifier"}


def is_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in NULLISH
    return False


def norm_email(value: str) -> str:
    return value.strip().lower()


def norm_phone(value: str) -> str:
    """Keep digits only so formatting differences do not break matching."""
    return re.sub(r"\D", "", value or "")


def norm_name(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


@dataclass
class Identifiers:
    """Direct identifiers found on (or used to look up) a record."""

    emails: Set[str] = field(default_factory=set)
    phones: Set[str] = field(default_factory=set)
    names: Set[str] = field(default_factory=set)
    hashes: Set[str] = field(default_factory=set)
    user_refs: Set[str] = field(default_factory=set)

    def add_value(self, value, *, as_name: bool = False) -> None:
        """Classify an arbitrary identifier-ish value and store it."""
        if value is None or isinstance(value, (dict, list)):
            return
        text = str(value).strip()
        if is_null(text):
            return
        if config.USER_REF_RE.fullmatch(text):
            self.user_refs.add(text.lower())
            return
        if config.SHA256_RE.fullmatch(text.lower()):
            self.hashes.add(text.lower())
            return
        if config.EMAIL_RE.fullmatch(text):
            self.emails.add(norm_email(text))
            return
        if as_name:
            self.names.add(norm_name(text))
            return
        digits = norm_phone(text)
        if text.startswith("+") and len(digits) >= 7:
            self.phones.add(digits)
            return
        if re.fullmatch(r"[\d\s()\-]{7,}", text) and len(digits) >= 7:
            self.phones.add(digits)
            return
        # fall back: embedded emails / hashes / user refs inside free text
        for m in config.EMAIL_RE.findall(text):
            self.emails.add(norm_email(m))
        for m in config.SHA256_RE.findall(text.lower()):
            self.hashes.add(m)
        for m in config.USER_REF_RE.findall(text):
            self.user_refs.add(f"user_{m}")

    def add_email(self, value) -> None:
        if not is_null(value):
            self.emails.add(norm_email(str(value)))

    def add_phone(self, value) -> None:
        if not is_null(value):
            d = norm_phone(str(value))
            if len(d) >= 7:
                self.phones.add(d)

    def add_name(self, value) -> None:
        if not is_null(value):
            self.names.add(norm_name(str(value)))

    def add_hash(self, value) -> None:
        if not is_null(value):
            v = str(value).strip().lower()
            if config.SHA256_RE.fullmatch(v):
                self.hashes.add(v)

    def add_user_ref(self, value) -> None:
        if not is_null(value):
            v = str(value).strip().lower()
            if config.USER_REF_RE.fullmatch(v):
                self.user_refs.add(v)

    def has_direct(self) -> bool:
        """True when a *direct* identifier (not just a stable user reference) is present."""
        return bool(self.emails or self.phones or self.names or self.hashes)

    def is_empty(self) -> bool:
        return not (self.emails or self.phones or self.names or self.hashes or self.user_refs)

    def merge(self, other: "Identifiers") -> None:
        self.emails |= other.emails
        self.phones |= other.phones
        self.names |= other.names
        self.hashes |= other.hashes
        self.user_refs |= other.user_refs

    def as_dict(self) -> dict:
        return {
            "emails": sorted(self.emails), "phones": sorted(self.phones), "names": sorted(self.names),
            "hashes": sorted(self.hashes), "user_refs": sorted(self.user_refs),
        }


@dataclass
class Observation:
    file_id: str
    observed_at: str                   # most specific timestamp available for this fact
    system_id: str
    resource_id: str
    kind: str
    present: Optional[bool] = None      # True = record exists, False = record absent, None = unknown
    identifiers: Identifiers = field(default_factory=Identifiers)   # identifiers carried by the record itself
    lookup: Identifiers = field(default_factory=Identifiers)        # identifiers used to look the record up
    identifiable: Optional[bool] = None  # direct identifiers still exposed (for REDACT / ANONYMIZE records)
    enabled: Optional[bool] = None       # data flows
    restore_allowed: Optional[bool] = None   # backups
    available: Optional[bool] = None     # exports
    retention_until: Optional[str] = None
    hold_active: Optional[bool] = None   # legal holds: False when the evidence says the hold was released
    source: str = ""
    file_captured_at: str = ""
    snapshot: bool = False               # True when the file is a closed-world snapshot of a collection
    attachment_file: Optional[str] = None
    record_type: Optional[str] = None    # record_type as reported by the evidence itself (if any)
    raw: dict = field(default_factory=dict)

    def all_identifiers(self) -> Identifiers:
        merged = Identifiers()
        merged.merge(self.identifiers)
        merged.merge(self.lookup)
        return merged

    def short(self) -> str:
        flags = []
        if self.present is not None:
            flags.append("present" if self.present else "absent")
        if self.identifiable is not None:
            flags.append("identifiable" if self.identifiable else "clean")
        if self.enabled is not None:
            flags.append("enabled" if self.enabled else "disabled")
        if self.restore_allowed is not None:
            flags.append("restorable" if self.restore_allowed else "blocked")
        if self.available is not None:
            flags.append("available" if self.available else "unavailable")
        ids = self.all_identifiers()
        return (f"{self.file_id} {self.observed_at} {self.system_id}/{self.resource_id} [{self.kind}] "
                f"{' '.join(flags)} ids={ids.as_dict()} <{self.source}>")


@dataclass
class SnapshotScope:
    """A closed-world snapshot: the file lists every record of the collection at that time."""

    file_id: str
    observed_at: str
    system_id: str
    kinds: Set[str]
    resource_ids: Set[str]
    file_captured_at: str = ""


def parse_all_evidence(evidence_files: Iterable[EvidenceFile], *, verbose: bool = False,
                       image_reader=None, known_emails: Optional[Set[str]] = None,
                       aliases: Optional[Dict[str, str]] = None,
                       ) -> tuple[List[Observation], List[SnapshotScope]]:
    """Parse every evidence file listed in the manifest.

    ``image_reader`` may be a callable(path) -> text used instead of local OCR
    (for example a vision model); when it is None the OCR extractor is used.
    ``known_emails`` lets the OCR path snap slightly garbled addresses back to
    real identities.
    """
    from . import logs, jsonrecords, sqlsnap, textrecords, pdfs, images

    observations: List[Observation] = []
    scopes: List[SnapshotScope] = []
    for ef in evidence_files:
        path = ef.abs_path
        if not os.path.exists(path):
            if verbose:
                print(f"[evidence] missing file {ef.file_path}")
            continue
        ext = ef.ext
        try:
            if ext == ".log":
                obs, sc = logs.parse_log(ef)
            elif ext == ".json":
                obs, sc = jsonrecords.parse_json(ef)
            elif ext == ".sql":
                obs, sc = sqlsnap.parse_sql(ef)
            elif ext == ".txt":
                obs, sc = textrecords.parse_text(ef)
            elif ext == ".pdf":
                obs, sc = pdfs.parse_pdf(ef, known_emails=known_emails, aliases=aliases)
            elif ext in (".png", ".jpg", ".jpeg", ".webp"):
                obs, sc = images.parse_image(ef, reader=image_reader, known_emails=known_emails, aliases=aliases)
            else:
                obs, sc = [], []
        except Exception as exc:  # a broken file must never stop the audit
            if verbose:
                print(f"[evidence] failed to parse {ef.file_path}: {exc!r}")
            obs, sc = [], []
        for o in obs:
            o.file_captured_at = ef.captured_at
            if not o.observed_at:
                o.observed_at = ef.captured_at
            if not o.system_id:
                o.system_id = config.system_for_resource(o.resource_id)
        for s in sc:
            s.file_captured_at = ef.captured_at
        if verbose:
            print(f"[evidence] {ef.file_id} {ef.file_path}: {len(obs)} observations, {len(sc)} snapshot scopes")
        observations.extend(obs)
        scopes.extend(sc)
    return observations, scopes
