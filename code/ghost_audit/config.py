"""Constants shared by the deletion-audit engine."""

from __future__ import annotations

import os
import re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(CODE_DIR)
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
EVIDENCE_DIR = os.path.join(DATASET_DIR, "evidence")

# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------
PHASES = ("FINDING", "REMEDIATION", "VERIFICATION")

# Operation order inside each phase follows the table in problem_statement.md.
OPERATION_ORDER = {
    "REPORT_REMAINING_RECORD": 0,
    "REPORT_EXPOSED_PII": 1,
    "REPORT_ACTIVE_DATA_FLOW": 2,
    "REPORT_RESTORE_RISK": 3,
    "DISABLE_DATA_FLOW": 10,
    "BLOCK_RESTORE": 11,
    "DELETE_RECORD": 12,
    "REDACT_RECORD": 13,
    "ANONYMIZE_RECORD": 14,
    "VERIFY_RECORD_ABSENT": 20,
    "VERIFY_RECORD_PRESENT": 21,
    "VERIFY_RECORD_REDACTED": 22,
    "VERIFY_RECORD_ANONYMIZED": 23,
    "VERIFY_DATA_FLOW_DISABLED": 24,
    "VERIFY_RESTORE_BLOCKED": 25,
}

OPERATION_STATE = {
    "REPORT_REMAINING_RECORD": "PRESENT",
    "REPORT_EXPOSED_PII": "PRESENT",
    "REPORT_ACTIVE_DATA_FLOW": "ACTIVE",
    "REPORT_RESTORE_RISK": "ACTIVE",
    "DISABLE_DATA_FLOW": "DISABLED",
    "BLOCK_RESTORE": "BLOCKED",
    "DELETE_RECORD": "ABSENT",
    "REDACT_RECORD": "REDACTED",
    "ANONYMIZE_RECORD": "ANONYMIZED",
    "VERIFY_RECORD_ABSENT": "ABSENT",
    "VERIFY_RECORD_PRESENT": "PRESENT",
    "VERIFY_RECORD_REDACTED": "REDACTED",
    "VERIFY_RECORD_ANONYMIZED": "ANONYMIZED",
    "VERIFY_DATA_FLOW_DISABLED": "DISABLED",
    "VERIFY_RESTORE_BLOCKED": "BLOCKED",
}

OPERATION_PHASE = {op: ("FINDING" if op.startswith("REPORT_") else "VERIFICATION" if op.startswith("VERIFY_") else "REMEDIATION")
                   for op in OPERATION_ORDER}

# ---------------------------------------------------------------------------
# Resource naming
# ---------------------------------------------------------------------------
# resource_id prefix -> canonical system_id used in the output
PREFIX_TO_SYSTEM = {
    "account": "account_service",
    "search": "search_service",
    "analytics": "analytics_platform",
    "support": "support_platform",
    "export": "export_service",
    "data_flow": "integration_service",
    "backup": "backup_service",
    "audit_record": "security_service",
    "tombstone": "identity_service",
    "marketing": "marketing_platform",
    "legal_hold": "legal_service",
    "object": "object_storage",
    "cache": "cache_service",
    "email_delivery": "email_delivery_service",
}

RESOURCE_ID_RE = re.compile(r"\b(account|search|analytics|support|export|data_flow|backup|audit_record|tombstone|marketing|legal_hold|object|cache|email_delivery)_(\d+)\b")
USER_REF_RE = re.compile(r"\buser_(\d+)\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
SHA256_RE = re.compile(r"\b[0-9a-f]{64}\b")

# Record kinds produced by the evidence layer
KIND_PROFILE = "PROFILE"
KIND_SEARCH = "SEARCH"
KIND_ANALYTICS = "ANALYTICS"
KIND_SUPPORT = "SUPPORT"             # support record of unknown sub-type
KIND_SUPPORT_MESSAGE = "SUPPORT_MESSAGE"
KIND_SUPPORT_ATTACHMENT = "SUPPORT_ATTACHMENT"
KIND_EXPORT = "EXPORT"
KIND_DATA_FLOW = "DATA_FLOW"
KIND_BACKUP = "BACKUP"
KIND_AUDIT = "AUDIT_RECORD"
KIND_TOMBSTONE = "TOMBSTONE"
KIND_MARKETING = "MARKETING"
KIND_LEGAL_HOLD = "LEGAL_HOLD"
KIND_CONTACT = "CONTACT"             # marketing-style contact record living in another system
KIND_CACHE = "CACHE_ENTRY"
KIND_EMAIL_DELIVERY = "EMAIL_DELIVERY_LOG"
KIND_TELEMETRY = "PRODUCT_TELEMETRY"
KIND_OTHER = "OTHER"

PREFIX_TO_KIND = {
    "account": KIND_PROFILE,
    "search": KIND_SEARCH,
    "analytics": KIND_ANALYTICS,
    "support": KIND_SUPPORT,
    "export": KIND_EXPORT,
    "data_flow": KIND_DATA_FLOW,
    "backup": KIND_BACKUP,
    "audit_record": KIND_AUDIT,
    "tombstone": KIND_TOMBSTONE,
    "marketing": KIND_MARKETING,
    "legal_hold": KIND_LEGAL_HOLD,
    "object": KIND_SUPPORT_ATTACHMENT,
    "cache": KIND_CACHE,
    "email_delivery": KIND_EMAIL_DELIVERY,
}


def check_system_map(worker_rows, min_rows: int = 5, min_share: float = 0.9):
    """Compare PREFIX_TO_SYSTEM with the systems the deletion worker actually targets for each resource type.

    Where the activity data is clear (at least ``min_rows`` rows, ``min_share`` of them to one system) and
    disagrees with the table, the data wins and the difference is returned as a warning. Prefixes the worker
    never touches keep the table value. Returns (confirmed prefixes, warnings).
    """
    from collections import Counter, defaultdict

    seen = defaultdict(Counter)
    for a in worker_rows:
        p = resource_prefix(a.resource_id)
        if p and a.target_system:
            seen[p][a.target_system] += 1
    confirmed, warnings = [], []
    for p, cnt in sorted(seen.items()):
        system, n = cnt.most_common(1)[0]
        total = sum(cnt.values())
        if total < min_rows or n / total < min_share:
            continue
        if PREFIX_TO_SYSTEM.get(p) == system:
            confirmed.append(p)
        else:
            warnings.append(f"system map: {p}_* records are handled in {system} by the deletion worker "
                            f"({n}/{total} rows), not {PREFIX_TO_SYSTEM.get(p) or 'unknown'}; using {system}")
            PREFIX_TO_SYSTEM[p] = system
    return confirmed, warnings


def resource_prefix(resource_id: str) -> str:
    """Return the resource family prefix (``account`` for ``account_123``)."""
    m = RESOURCE_ID_RE.match(resource_id or "")
    return m.group(1) if m else ""


def system_for_resource(resource_id: str) -> str:
    return PREFIX_TO_SYSTEM.get(resource_prefix(resource_id), "")


def kind_for_resource(resource_id: str) -> str:
    return PREFIX_TO_KIND.get(resource_prefix(resource_id), KIND_OTHER)


def resource_sort_key(resource_id: str):
    """Ascending order by resource_id as required by the contract (plain string order)."""
    return resource_id


def case_sort_key(case_id: str):
    m = re.match(r"case_(\d+)$", case_id)
    return (0, int(m.group(1))) if m else (1, case_id)
