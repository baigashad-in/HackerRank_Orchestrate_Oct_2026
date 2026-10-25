"""Turn the text of a business document (PDF text or screenshot OCR) into observations.

Documents never state a deletion verdict.  They show a record with (or without)
direct identifiers, or a row in an admin console.  We locate the record the
document is about (the *focus* resource), attach the identifiers that belong to
it, and derive the operational flags that the console exposes (export ready,
connector continuous, archive restorable, ...).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Set

from .. import config
from ..loaders import EvidenceFile
from .base import Identifiers, Observation, norm_email

ORIGIN_PREFIX = {
    "account_admin_portal": "account",
    "account_history_service": "export",
    "analytics_platform": "analytics",
    "backup_service": "backup",
    "integration_admin": "data_flow",
    "integration_management_service": "data_flow",
    "export_service": "export",
    "customer_export_service": "export",
    "identity_service": "tombstone",
    "marketing_platform": "marketing",
    "object_storage": "object",
    "search_index_admin": "search",
    "security_service": "audit_record",
    "support_platform": "support",
    "legal_retention_service": "legal_hold",
}

PHONE_RE = re.compile(r"\+\d[\d\s().\-]{6,}\d")
STATEMENT_RE = re.compile(r"\b(NST-\d{4}-\d{4}-(\d+))\b")
ORDER_RE = re.compile(r"\b(ORD-2026(\d{4}))\b")
RETAIN_RE = re.compile(r"[Rr]etain(?:ed)?\s+until\s+([0-9]{1,2}\s+\w{3}\s+\d{4}(?:\s+[\d:]+)?|\d{4}-\d{2}-\d{2}(?:T[\d:]+Z)?)")
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

DISABLED_WORDS = re.compile(r"\b(disabled|stopped|paused|inactive|revoked|off)\b", re.I)
BLOCKED_WORDS = re.compile(r"\b(restore\s+blocked|blocked|restore\s+denied|legal\s+hold|quarantined|deletion\s+block)\b", re.I)
EXPIRED_WORDS = re.compile(r"\b(expired|deleted|purged|unavailable|removed)\b", re.I)
READY_WORDS = re.compile(r"\b(ready|available|download)\b", re.I)


def _iso_from_human(date_text: str) -> Optional[str]:
    date_text = date_text.strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", date_text):
        return date_text if "T" in date_text else date_text + "T23:59:59Z"
    m = re.match(r"(\d{1,2})\s+(\w{3})\w*\s+(\d{4})(?:\s+(\d{1,2}):(\d{2}))?", date_text)
    if not m:
        return None
    day, mon, year = int(m.group(1)), MONTHS.get(m.group(2).lower()), int(m.group(3))
    if not mon:
        return None
    hh, mm = (int(m.group(4)), int(m.group(5))) if m.group(4) else (23, 59)
    return f"{year:04d}-{mon:02d}-{day:02d}T{hh:02d}:{mm:02d}:00Z"


def _line_identifiers(line: str, known_emails: Optional[Set[str]] = None) -> Identifiers:
    ids = Identifiers()
    for m in config.EMAIL_RE.findall(line):
        ids.emails.add(_repair_email(norm_email(m), known_emails))
    for m in config.USER_REF_RE.findall(line):
        ids.user_refs.add(f"user_{m}")
    for m in config.SHA256_RE.findall(line.lower()):
        ids.hashes.add(m)
    for m in PHONE_RE.findall(line):
        ids.add_phone(m)
    return ids


def _repair_email(email: str, known_emails: Optional[Set[str]]) -> str:
    """OCR sometimes garbles one or two characters of an address; snap to a known address when close."""
    if not known_emails or email in known_emails:
        return email
    local, _, domain = email.partition("@")
    best, best_d = None, 3
    for cand in known_emails:
        c_local, _, c_domain = cand.partition("@")
        if c_domain != domain or abs(len(c_local) - len(local)) > 2:
            continue
        d = _edit_distance(local, c_local, best_d)
        if d < best_d:
            best, best_d = cand, d
    return best or email


def _edit_distance(a: str, b: str, limit: int) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def observations_from_document_text(ef: EvidenceFile, text: str, *, source: str,
                                    known_emails: Optional[Set[str]] = None,
                                    aliases: Optional[Dict[str, str]] = None) -> List[Observation]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    joined = "\n".join(lines)
    out: List[Observation] = []
    expected_prefix = ORIGIN_PREFIX.get(ef.origin, "")
    # catalog aliases (statement numbers, attachment file names, ...) name resources without their id
    for alias, rid in (aliases or {}).items():
        if alias and alias in joined and rid not in joined and config.resource_prefix(rid) == expected_prefix:
            joined = joined + f"\n{rid} {alias}"
            lines = lines + [f"{rid} {alias}"]

    # --- legal register rows: ORD-2026NNNN | user_XX | reason | date | amount | Retain until DATE
    legal_rows = [ln for ln in lines if ORDER_RE.search(ln) and config.USER_REF_RE.search(ln) and RETAIN_RE.search(ln)]
    if legal_rows and expected_prefix == "legal_hold":
        for ln in legal_rows:
            om = ORDER_RE.search(ln)
            rid = f"legal_hold_{int(om.group(2))}"
            obs = Observation(ef.file_id, ef.captured_at, "legal_service", rid, config.KIND_LEGAL_HOLD, present=True,
                              source=f"{source}:legal_register")
            obs.identifiers.merge(_line_identifiers(ln, known_emails))
            rm = RETAIN_RE.search(ln)
            obs.retention_until = _iso_from_human(rm.group(1)) if rm else None
            obs.raw = {"order_reference": om.group(1), "alias": om.group(1)}
            out.append(obs)
        return out

    counts: Counter = Counter()
    first_pos: Dict[str, int] = {}
    for i, ln in enumerate(lines):
        for m in config.RESOURCE_ID_RE.finditer(ln):
            rid = m.group(0)
            counts[rid] += 1
            first_pos.setdefault(rid, i)

    focus: Optional[str] = None
    alias: Optional[str] = None
    candidates = [r for r in counts if config.resource_prefix(r) == expected_prefix] if expected_prefix else list(counts)
    if candidates:
        focus = max(candidates, key=lambda r: (counts[r], -first_pos[r]))
    elif expected_prefix == "export":
        sm = STATEMENT_RE.search(joined)
        if sm:
            alias = sm.group(1)
            focus = f"export_{int(sm.group(2))}"
    elif expected_prefix == "tombstone":
        um = config.USER_REF_RE.search(joined)
        if um:
            alias = f"user_{um.group(1)}"
            focus = alias  # resolved through the data catalog alias table in the pipeline
    elif counts:
        focus = max(counts, key=lambda r: (counts[r], -first_pos[r]))
    if not focus:
        return out

    kind = config.kind_for_resource(focus) if config.RESOURCE_ID_RE.fullmatch(focus) else config.KIND_TOMBSTONE
    sys_id = config.system_for_resource(focus) or config.PREFIX_TO_SYSTEM.get(expected_prefix, ef.origin)
    if expected_prefix == "support":
        kind = config.KIND_SUPPORT_ATTACHMENT if re.search(r"attachment|upload", joined, re.I) else config.KIND_SUPPORT_MESSAGE
    obs = Observation(ef.file_id, ef.captured_at, sys_id, focus, kind, present=True, source=f"{source}:document")
    if alias:
        obs.raw["alias"] = alias

    # identifiers: everything on lines that mention the focus, plus identifiers that are either repeated
    # or the only one of their type in the document (table rows about other people are excluded)
    focus_lines = [ln for ln in lines if focus in ln or (alias and alias in ln)]
    other_rid_lines = [ln for ln in lines if config.RESOURCE_ID_RE.search(ln) and focus not in ln]
    all_ids = _line_identifiers(joined, known_emails)
    focus_ids = Identifiers()
    for ln in focus_lines:
        focus_ids.merge(_line_identifiers(ln, known_emails))
    other_ids = Identifiers()
    for ln in other_rid_lines:
        other_ids.merge(_line_identifiers(ln, known_emails))
    email_counts = Counter(_repair_email(norm_email(m), known_emails) for m in config.EMAIL_RE.findall(joined))
    for email, n in email_counts.items():
        if email in focus_ids.emails:
            continue
        if email in other_ids.emails:
            continue
        if n >= 2 or len(email_counts) == 1:
            focus_ids.emails.add(email)
    for ref in all_ids.user_refs:
        if ref not in other_ids.user_refs:
            focus_ids.user_refs.add(ref)
    for ph in all_ids.phones:
        if ph not in other_ids.phones:
            focus_ids.phones.add(ph)
    for h in all_ids.hashes:
        focus_ids.hashes.add(h)
    obs.identifiers.merge(focus_ids)

    focus_text = "\n".join(focus_lines) if focus_lines else joined
    if kind == config.KIND_DATA_FLOW:
        obs.enabled = not DISABLED_WORDS.search(focus_text)
    elif kind == config.KIND_BACKUP:
        obs.restore_allowed = not BLOCKED_WORDS.search(focus_text)
    elif kind == config.KIND_EXPORT:
        if EXPIRED_WORDS.search(focus_text) and not READY_WORDS.search(focus_text):
            obs.available = False
            obs.present = False
        else:
            obs.available = True
    elif kind in (config.KIND_ANALYTICS, config.KIND_TOMBSTONE, config.KIND_MARKETING,
                  config.KIND_SUPPORT_MESSAGE, config.KIND_SUPPORT_ATTACHMENT, config.KIND_PROFILE, config.KIND_SEARCH):
        obs.identifiable = obs.identifiers.has_direct()
    if kind == config.KIND_SUPPORT_ATTACHMENT:
        fm = re.search(r"\b([\w-]+\.(?:pdf|jpg|jpeg|png))\b", joined, re.I)
        obs.attachment_file = fm.group(1) if fm else None
    rm = RETAIN_RE.search(joined)
    if rm and kind in (config.KIND_AUDIT, config.KIND_LEGAL_HOLD):
        obs.retention_until = _iso_from_human(rm.group(1))
    out.append(obs)
    return out
