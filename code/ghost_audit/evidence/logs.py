"""Parser for the many log dialects used by the observability exports.

Dialects seen in the dataset (all start with an ISO-8601 timestamp):

* logfmt            ``TS level=INFO service=x trace_id=t event=e k=v k="v"``
* JSON lines        ``TS {"severity": "INFO", "logger": "x", ...}``
* bracketed         ``TS [INFO] [service] [trace] event: k="v" ...``
* cloud runtime     ``TS requestId=t level=INFO function=x message="k=\\"v\\" ..."``
* pipe separated    ``TS | INFO | service | trace | event | k="v" ...``
* syslog            ``TS <14>1 node service trace - - level=INFO k="v" ...``

Lines are normalised into a flat dict, traces are joined on their trace id so
the lookup key of a ``*_started`` line is attached to the ``*_completed`` line,
and each completed lookup becomes an :class:`Observation`.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from .. import config
from ..loaders import EvidenceFile
from .base import Identifiers, Observation, SnapshotScope
from .records import observation_from_lookup

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+(.*)$")
KV_RE = re.compile(r'([A-Za-z_][\w.-]*)=("(?:[^"\\]|\\.)*"|\'[^\']*\'|\S+)')
BRACKET_RE = re.compile(r"^\[(\w+)\]\s+\[([^\]]+)\]\s+\[([^\]]+)\]\s+([\w.-]+):\s*(.*)$")
PIPE_RE = re.compile(r"^(\w+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.*)$")
SYSLOG_RE = re.compile(r"^<\d+>\d+\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+\s+\S+\s+(.*)$")
LEVEL_RE = re.compile(r"^(DEBUG|INFO|WARN|WARNING|ERROR|TRACE)\s+(.*)$")

LOOKUP_KEYS = ("lookup_key", "match_key", "subject_key", "subject_ref", "scan_key", "owner_key",
               "correlation_key", "lookup_value", "query")
IDENTIFIER_KEYS = ("email", "customer_email", "owner_email", "requester_email", "phone", "requester_phone",
                   "owner_phone", "email_hash", "customer_reference", "subject_reference", "requester_name",
                   "display_name", "full_name", "matched_value", "user_id")


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        inner = value[1:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\")
    return value


def _parse_kv(text: str) -> Dict[str, str]:
    return {k: _unquote(v) for k, v in KV_RE.findall(text)}


def normalise_line(line: str) -> Optional[Dict[str, str]]:
    """Turn one log line into a flat dict with at least ``ts``; None for blank lines."""
    line = line.rstrip("\r\n")
    if not line.strip():
        return None
    m = TS_RE.match(line)
    if not m:
        return None
    ts, rest = m.group(1), m.group(2)
    if not ts.endswith("Z"):
        ts += "Z"
    fields: Dict[str, str] = {"ts": ts}
    rest_stripped = rest.strip()
    if rest_stripped.startswith("{"):
        try:
            data = json.loads(rest_stripped)
            for k, v in data.items():
                fields[k] = "" if v is None else str(v)
            fields.setdefault("level", fields.get("severity", ""))
            fields.setdefault("service", fields.get("logger", ""))
            return fields
        except json.JSONDecodeError:
            pass
    if rest_stripped.startswith("|"):
        rest_stripped = rest_stripped.lstrip("| ")
    bm = BRACKET_RE.match(rest_stripped)
    if bm:
        fields.update({"level": bm.group(1), "service": bm.group(2), "trace_id": bm.group(3), "event": bm.group(4)})
        fields.update(_parse_kv(bm.group(5)))
        return fields
    pm = PIPE_RE.match(rest_stripped)
    if pm and "|" in rest_stripped[:60]:
        fields.update({"level": pm.group(1), "service": pm.group(2), "trace_id": pm.group(3), "event": pm.group(4)})
        fields.update(_parse_kv(pm.group(5)))
        return fields
    sm = SYSLOG_RE.match(rest_stripped)
    if sm:
        fields.update({"host": sm.group(1), "service": sm.group(2), "trace_id": sm.group(3)})
        fields.update(_parse_kv(sm.group(4)))
        return fields
    lm = LEVEL_RE.match(rest_stripped)
    if lm:
        fields["level"] = lm.group(1)
        rest_stripped = lm.group(2)
    kv = _parse_kv(rest_stripped)
    if "message" in kv and "=" in kv["message"]:
        inner = _parse_kv(kv.pop("message"))
        kv.update(inner)
    if "requestId" in kv and "trace_id" not in kv:
        kv["trace_id"] = kv["requestId"]
    if "function" in kv and "service" not in kv:
        kv["service"] = kv["function"]
    fields.update(kv)
    return fields


def _truthy(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"true", "1", "yes", "hit", "found", "allowed", "enabled", "active", "passed"}:
        return True
    if v in {"false", "0", "no", "miss", "not_found", "denied", "disabled", "stopped", "blocked"}:
        return False
    return None


def _int(value: Optional[str]) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _service_system(service: str, rid: str) -> str:
    sys_id = config.system_for_resource(rid)
    if sys_id:
        return sys_id
    s = (service or "").lower()
    if s in config.PREFIX_TO_SYSTEM.values():
        return s
    if s in ("profile_service",):
        return "account_service"
    return s


class _Trace:
    def __init__(self) -> None:
        self.lookup = Identifiers()
        self.resource_id: Optional[str] = None
        self.started_at: Optional[str] = None


def _collect_identifiers(fields: Dict[str, str], obs: Observation) -> None:
    ids = obs.identifiers
    for key in IDENTIFIER_KEYS:
        if key not in fields:
            continue
        val = fields[key]
        if key in ("requester_name", "display_name", "full_name"):
            ids.add_name(val)
        elif key in ("phone", "requester_phone", "owner_phone"):
            ids.add_phone(val)
        elif key == "email_hash":
            ids.add_hash(val)
        elif key in ("customer_reference", "subject_reference", "user_id"):
            ids.add_user_ref(val)
        else:
            ids.add_value(val)


def _lookup_identifiers(fields: Dict[str, str], obs: Observation, trace: Optional[_Trace]) -> None:
    for key in LOOKUP_KEYS:
        if key in fields:
            obs.lookup.add_value(fields[key])
    if trace is not None:
        obs.lookup.merge(trace.lookup)


def build_observation(fields: Dict[str, str], trace: Optional[_Trace], file_id: str) -> Optional[Observation]:
    event = (fields.get("event") or "").strip().strip('"')
    rid = fields.get("resource_id") or (trace.resource_id if trace else None)
    ts = fields["ts"]
    service = fields.get("service", "")
    if event == "record_lookup":
        return observation_from_lookup(file_id, ts, fields.get("system"), rid, fields.get("record_type"),
                                       fields.get("result"), fields.get("subject_reference") or fields.get("lookup_value"),
                                       source=f"log:{event}")
    if not rid or not config.RESOURCE_ID_RE.fullmatch(rid):
        return None
    kind = config.kind_for_resource(rid)
    sys_id = _service_system(service, rid)
    src = f"log:{event}"
    obs = Observation(file_id, ts, sys_id, rid, kind, source=src, raw=dict(fields))

    if event == "profile_lookup_completed":
        status = _int(fields.get("http_status"))
        rows = _int(fields.get("rows"))
        obs.present = (status == 200) if status is not None else (rows is not None and rows > 0)
        if rows is not None and rows > 0:
            obs.present = True
    elif event == "index_query_completed":
        hits = _int(fields.get("total_hits"))
        res = _truthy(fields.get("result"))
        obs.present = (hits > 0) if hits is not None else res
        if hits == 0:
            obs.present = False
    elif event in ("analytics_profile_read_completed", "analytics_profile_read"):
        obs.kind = config.KIND_ANALYTICS
        obs.present = True
    elif event == "analytics_identity_scan":
        obs.kind = config.KIND_ANALYTICS
        obs.present = True
        obs.identifiable = True
    elif event == "contact_lookup_completed":
        obs.kind = config.KIND_CONTACT
        rows = _int(fields.get("rows"))
        obs.present = rows is None or rows > 0
    elif event == "support_ticket_read_completed":
        obs.kind = config.KIND_SUPPORT_MESSAGE
        obs.present = True
    elif event == "message_pii_scan":
        obs.kind = config.KIND_SUPPORT_MESSAGE
        obs.present = True
        obs.identifiable = True
    elif event == "message_read":
        obs.kind = config.KIND_SUPPORT_MESSAGE
        obs.present = True
        tmp = Identifiers()
        tmp.add_value(fields.get("message_text"))
        obs.identifiers.merge(tmp)
        obs.identifiable = tmp.has_direct()
    elif event == "attachment_pii_scan":
        obs.kind = config.KIND_SUPPORT_ATTACHMENT
        obs.present = True
        obs.identifiable = True
        obs.attachment_file = fields.get("file")
    elif event == "audit_record_read_completed":
        obs.kind = config.KIND_AUDIT
        rows = _int(fields.get("rows"))
        obs.present = rows is None or rows > 0
        obs.retention_until = fields.get("retention_until") or None
    elif event in ("connector_config_read_completed", "connector_heartbeat", "connector_config_loaded"):
        obs.kind = config.KIND_DATA_FLOW
        obs.present = True
        obs.enabled = _truthy(fields.get("enabled"))
        if obs.enabled is None and fields.get("last_delivery_status"):
            obs.enabled = fields["last_delivery_status"].lower() not in {"stopped", "disabled"}
    elif event == "restore_policy_read_completed":
        obs.kind = config.KIND_BACKUP
        obs.present = _truthy(fields.get("archive_present"))
        if obs.present is None:
            obs.present = True
        obs.restore_allowed = _truthy(fields.get("restore_allowed"))
    elif event == "restore_candidate_evaluated":
        obs.kind = config.KIND_BACKUP
        obs.present = True
        eligible = _truthy(fields.get("eligible"))
        block = _truthy(fields.get("deletion_block"))
        if eligible is not None or block is not None:
            obs.restore_allowed = bool(eligible) and not bool(block)
    elif event == "restore_authorization_completed":
        obs.kind = config.KIND_BACKUP
        obs.present = True
        decision = (fields.get("decision") or "").lower()
        if decision:
            obs.restore_allowed = decision in {"allow", "allowed", "approved", "granted", "permit", "permitted"}
    elif event in ("export_download_check_completed", "export_metadata_read_completed"):
        obs.kind = config.KIND_EXPORT
        obs.available = _truthy(fields.get("available"))
        obs.present = obs.available if obs.available is not None else True
        obs.attachment_file = fields.get("file") or fields.get("object_key")
    elif event == "contact_query_completed":
        obs.kind = config.KIND_SUPPORT
        rows = _int(fields.get("rows"))
        obs.present = rows is None or rows > 0
    elif event == "deletion_marker_check":
        obs.kind = config.KIND_SUPPORT
        marker = _truthy(fields.get("marker_found"))
        obs.present = (not marker) if marker is not None else None
    else:
        return None

    _collect_identifiers(fields, obs)
    _lookup_identifiers(fields, obs, trace)
    if obs.kind in (config.KIND_ANALYTICS, config.KIND_SUPPORT_MESSAGE, config.KIND_SUPPORT, config.KIND_TOMBSTONE) \
            and obs.identifiable is None:
        direct = _int(fields.get("direct_identifiers"))
        if direct is not None:
            obs.identifiable = direct > 0 or obs.identifiers.has_direct()
        elif event in ("analytics_profile_read_completed", "support_ticket_read_completed"):
            obs.identifiable = obs.identifiers.has_direct()
    return obs


def parse_log(ef: EvidenceFile) -> Tuple[List[Observation], List[SnapshotScope]]:
    traces: Dict[str, _Trace] = {}
    out: List[Observation] = []
    with open(ef.abs_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            fields = normalise_line(line)
            if not fields:
                continue
            event = (fields.get("event") or "").strip().strip('"')
            fields["event"] = event
            tid = fields.get("trace_id") or ""
            trace = traces.get(tid) if tid else None
            if event.endswith("_started") or event.endswith("_scan_started"):
                if tid:
                    trace = traces.setdefault(tid, _Trace())
                    for key in LOOKUP_KEYS:
                        if key in fields:
                            trace.lookup.add_value(fields[key])
                    if fields.get("resource_id"):
                        trace.resource_id = fields["resource_id"]
                    trace.started_at = fields["ts"]
                continue
            obs = build_observation(fields, trace, ef.file_id)
            if obs is not None:
                out.append(obs)
    return out, []
