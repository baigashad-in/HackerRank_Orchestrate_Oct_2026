"""Shared interpretation of structured records (JSON snapshot bundles, SQL record_json rows,
inventory tables and record-lookup style rows)."""

from __future__ import annotations

from typing import Optional

from .. import config
from .base import Identifiers, Observation, is_null


def _bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "enabled", "active", "allowed"}:
        return True
    if text in {"0", "false", "no", "n", "disabled", "stopped", "blocked", "denied"}:
        return False
    return None


def _collection_kind(collection: str, record: dict) -> str:
    c = (collection or "").lower()
    if "profile" in c and "analytics" not in c:
        return config.KIND_PROFILE
    if "search" in c:
        return config.KIND_SEARCH
    if "analytics" in c:
        return config.KIND_ANALYTICS
    if "attachment" in c:
        return config.KIND_SUPPORT_ATTACHMENT
    if "conversation" in c or "message" in c or "ticket" in c:
        return config.KIND_SUPPORT_MESSAGE
    if "support" in c:
        return config.KIND_SUPPORT
    if "export" in c:
        return config.KIND_EXPORT
    if "connector" in c:
        return config.KIND_DATA_FLOW
    if "backup" in c or "restore" in c:
        return config.KIND_BACKUP
    if "audit" in c:
        return config.KIND_AUDIT
    if "deleted_customer_references" in c or "tombstone" in c:
        return config.KIND_TOMBSTONE
    if "retained_order" in c or "legal" in c:
        return config.KIND_LEGAL_HOLD
    if "marketing" in c:
        return config.KIND_MARKETING
    return config.kind_for_resource(record.get("resource_id", ""))


def observation_from_record(file_id: str, observed_at: str, store: str, collection: str, record: dict,
                            *, source: str, snapshot: bool = True) -> Optional[Observation]:
    """Interpret one structured record of a known collection."""
    rid = record.get("resource_id")
    if not rid:
        return None
    kind = _collection_kind(collection, record)
    system = config.system_for_resource(rid) or (store or "")
    obs = Observation(file_id=file_id, observed_at=observed_at, system_id=system, resource_id=rid,
                      kind=kind, present=True, source=source, snapshot=snapshot, raw=record)
    ids = obs.identifiers

    doc = record.get("document") if isinstance(record.get("document"), dict) else None
    fields = dict(record)
    if doc:
        fields.update(doc)

    ids.add_user_ref(fields.get("customer_reference"))
    ids.add_user_ref(fields.get("subject_reference"))
    ids.add_user_ref(fields.get("user_id"))
    for key in ("email", "requester_email", "owner_email", "customer_email", "uploaded_by"):
        ids.add_email(fields.get(key))
    for key in ("phone", "requester_phone", "owner_phone"):
        ids.add_phone(fields.get(key))
    for key in ("full_name", "display_name", "requester_name", "name"):
        ids.add_name(fields.get(key))
    ids.add_hash(fields.get("email_hash"))
    # subject_selector / lookup_value can be an email or a user reference
    for key in ("subject_selector", "lookup_value"):
        if key in fields:
            ids.add_value(fields.get(key))

    # free-text message bodies can embed identifiers
    text = fields.get("message_text") or fields.get("message_excerpt")
    if isinstance(text, str) and not is_null(text):
        tmp = Identifiers()
        tmp.add_value(text)
        ids.merge(tmp)
        if kind in (config.KIND_SUPPORT, config.KIND_SUPPORT_MESSAGE):
            obs.identifiable = tmp.has_direct() or any(
                not is_null(fields.get(k)) for k in ("requester_email", "requester_name", "requester_phone"))
    if kind in (config.KIND_SUPPORT, config.KIND_SUPPORT_MESSAGE) and obs.identifiable is None:
        obs.identifiable = any(not is_null(fields.get(k)) for k in ("requester_email", "requester_name", "requester_phone"))
        if isinstance(text, str) and text.strip().lower() == "[redacted]":
            obs.identifiable = obs.identifiable or False

    if kind == config.KIND_ANALYTICS:
        obs.identifiable = ids.has_direct()
    if kind == config.KIND_TOMBSTONE:
        obs.identifiable = ids.has_direct()
    if kind == config.KIND_DATA_FLOW:
        obs.enabled = _bool(fields.get("enabled"))
        if obs.enabled is None and not is_null(fields.get("last_delivery_status")):
            obs.enabled = str(fields.get("last_delivery_status")).lower() not in {"stopped", "disabled"}
    if kind == config.KIND_BACKUP:
        obs.restore_allowed = _bool(fields.get("restore_allowed"))
    if kind == config.KIND_EXPORT:
        if "download_available" in fields:
            obs.available = _bool(fields.get("download_available"))
        obs.present = True if obs.available is None else obs.available
    if kind in (config.KIND_AUDIT, config.KIND_LEGAL_HOLD):
        obs.retention_until = fields.get("retention_until") or fields.get("retained_until")
    if kind == config.KIND_LEGAL_HOLD:
        hold = fields.get("hold_status") or fields.get("status")
        if isinstance(hold, str) and hold.strip():
            obs.hold_active = hold.strip().upper() not in INACTIVE_HOLD_STATUSES
        if not is_null(fields.get("released_at")) and str(fields.get("released_at")) <= (observed_at_of(obs) or "9999"):
            obs.hold_active = False
    if kind == config.KIND_SUPPORT_ATTACHMENT:
        obs.attachment_file = fields.get("file_name")
    status = fields.get("account_status")
    if isinstance(status, str):
        obs.raw["account_status"] = status
    return obs


INACTIVE_HOLD_STATUSES = {"RELEASED", "EXPIRED", "LIFTED", "INACTIVE", "CLOSED", "ENDED", "CANCELLED", "CANCELED"}


def observed_at_of(obs) -> str:
    return obs.observed_at or obs.file_captured_at or ""


RESULT_PRESENT = {"FOUND": True, "NOT_FOUND": False, "IDENTIFIABLE": True, "ANONYMIZED": True, "REDACTED": True,
                  "ACTIVE": True, "DISABLED": True, "BLOCKED": True, "PRESENT": True, "ABSENT": False,
                  "MISSING": False, "DELETED": False}

RECORD_TYPE_KIND = {
    "USER_PROFILE": config.KIND_PROFILE,
    "SEARCH_INDEX_ENTRY": config.KIND_SEARCH,
    "ANALYTICS_PROFILE": config.KIND_ANALYTICS,
    "SUPPORT_MESSAGE": config.KIND_SUPPORT_MESSAGE,
    "SUPPORT_ATTACHMENT": config.KIND_SUPPORT_ATTACHMENT,
    "USER_EXPORT": config.KIND_EXPORT,
    "DATA_FLOW": config.KIND_DATA_FLOW,
    "BACKUP_ARCHIVE": config.KIND_BACKUP,
    "SECURITY_AUDIT_LOG": config.KIND_AUDIT,
    "ACCOUNT_TOMBSTONE": config.KIND_TOMBSTONE,
    "LEGAL_HOLD_RECORD": config.KIND_LEGAL_HOLD,
    "MARKETING_PROFILE": config.KIND_MARKETING,
    "CACHE_ENTRY": config.KIND_CACHE,
    "EMAIL_DELIVERY_LOG": config.KIND_EMAIL_DELIVERY,
    "SECONDARY_IDENTIFIER": config.KIND_CONTACT,
    "PRODUCT_TELEMETRY": config.KIND_TELEMETRY,
}


def observation_from_lookup(file_id: str, observed_at: str, system: str, resource_id: str, record_type: str,
                            result: str, subject, *, source: str) -> Optional[Observation]:
    """Interpret a record-lookup style row: system | resource | record_type | result | subject."""
    if not resource_id:
        return None
    record_type = (record_type or "").strip().upper()
    result = (result or "").strip().upper()
    kind = RECORD_TYPE_KIND.get(record_type, config.kind_for_resource(resource_id))
    sys_id = config.system_for_resource(resource_id) or system
    obs = Observation(file_id=file_id, observed_at=observed_at, system_id=sys_id, resource_id=resource_id,
                      kind=kind, source=source, snapshot=False, record_type=record_type)
    obs.present = RESULT_PRESENT.get(result)
    obs.lookup.add_value(subject)
    if result == "IDENTIFIABLE":
        obs.identifiable = True
        obs.identifiers.add_value(subject)
    elif result in ("ANONYMIZED", "REDACTED"):
        obs.identifiable = False
    if kind == config.KIND_DATA_FLOW:
        obs.enabled = {"ACTIVE": True, "DISABLED": False}.get(result)
    if kind == config.KIND_BACKUP:
        obs.restore_allowed = {"ACTIVE": True, "BLOCKED": False}.get(result)
    if kind in (config.KIND_PROFILE, config.KIND_SEARCH, config.KIND_EXPORT, config.KIND_SUPPORT_ATTACHMENT,
                config.KIND_AUDIT, config.KIND_LEGAL_HOLD, config.KIND_MARKETING) and result == "FOUND":
        obs.identifiers.add_value(subject)
    return obs
