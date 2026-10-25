"""Parsers for JSON evidence: service resource exports, records-assurance snapshot
bundles, record-lookup exports, analytics reports and export job listings."""

from __future__ import annotations

import json
from typing import List, Tuple

from .. import config
from ..loaders import EvidenceFile
from .base import Identifiers, Observation, SnapshotScope
from .records import observation_from_lookup, observation_from_record


def _sre_observations(ef: EvidenceFile, data: dict) -> List[Observation]:
    """service_resource_export: one lookup per resource with the observed state."""
    out: List[Observation] = []
    exported_at = data.get("exported_at") or ef.captured_at
    for service, groups in (data.get("service_exports") or {}).items():
        for group, entries in (groups or {}).items():
            for e in entries or []:
                rid = e.get("resource_id")
                if not rid:
                    continue
                observed = e.get("observed_at") or exported_at
                src = f"service_resource_export:{service}.{group}"
                sys_id = config.system_for_resource(rid) or service
                if group == "profile_lookups":
                    obs = Observation(ef.file_id, observed, sys_id, rid, config.KIND_PROFILE, source=src, raw=e)
                    status = e.get("status_code")
                    profile = e.get("profile") or {}
                    obs.present = (status == 200) or bool(profile)
                    obs.identifiers.add_user_ref(profile.get("customer_reference"))
                    obs.identifiers.add_email(profile.get("email"))
                    obs.identifiers.add_phone(profile.get("phone"))
                    obs.identifiers.add_name(profile.get("full_name"))
                    obs.identifiers.add_hash(profile.get("email_hash"))
                    out.append(obs)
                elif group == "profiles":  # analytics
                    obs = Observation(ef.file_id, observed, sys_id, rid, config.KIND_ANALYTICS, present=True, source=src, raw=e)
                    obs.identifiers.add_user_ref(e.get("customer_reference"))
                    obs.identifiers.add_email(e.get("email"))
                    obs.identifiers.add_phone(e.get("phone"))
                    obs.identifiers.add_hash(e.get("email_hash"))
                    obs.identifiable = obs.identifiers.has_direct()
                    out.append(obs)
                elif group == "restore_entries":
                    obs = Observation(ef.file_id, observed, sys_id, rid, config.KIND_BACKUP, present=True, source=src, raw=e)
                    obs.identifiers.add_user_ref(e.get("customer_reference"))
                    obs.identifiers.add_email(e.get("customer_email"))
                    obs.restore_allowed = bool(e.get("restore_allowed"))
                    out.append(obs)
                elif group == "exports":
                    obs = Observation(ef.file_id, observed, sys_id, rid, config.KIND_EXPORT, present=True, source=src, raw=e)
                    obs.identifiers.add_user_ref(e.get("customer_reference"))
                    obs.identifiers.add_email(e.get("owner_email"))
                    obs.identifiers.add_phone(e.get("owner_phone"))
                    obs.available = True
                    out.append(obs)
                elif group == "connectors":
                    obs = Observation(ef.file_id, observed, sys_id, rid, config.KIND_DATA_FLOW, present=True, source=src, raw=e)
                    obs.identifiers.add_user_ref(e.get("customer_reference"))
                    obs.identifiers.add_email(e.get("customer_email"))
                    obs.enabled = bool(e.get("enabled"))
                    out.append(obs)
                elif group == "index_queries":
                    obs = Observation(ef.file_id, observed, sys_id, rid, config.KIND_SEARCH, source=src, raw=e)
                    hits = e.get("hits") or {}
                    docs = hits.get("documents") or []
                    total = hits.get("total")
                    obs.present = bool(docs) or (isinstance(total, int) and total > 0)
                    obs.lookup.add_value(e.get("query"))
                    for d in docs:
                        obs.identifiers.add_user_ref(d.get("customer_reference"))
                        obs.identifiers.add_email(d.get("email"))
                        obs.identifiers.add_phone(d.get("phone"))
                        obs.identifiers.add_name(d.get("display_name"))
                        obs.identifiers.add_hash(d.get("email_hash"))
                    out.append(obs)
                elif group == "audit_records":
                    obs = Observation(ef.file_id, observed, sys_id, rid, config.KIND_AUDIT, present=True, source=src, raw=e)
                    obs.identifiers.add_user_ref(e.get("subject_reference"))
                    obs.retention_until = e.get("retention_until")
                    out.append(obs)
                elif group == "tickets":
                    obs = Observation(ef.file_id, observed, sys_id, rid, config.KIND_SUPPORT_MESSAGE, present=True, source=src, raw=e)
                    obs.identifiers.add_user_ref(e.get("customer_reference"))
                    obs.identifiers.add_email(e.get("requester_email"))
                    obs.identifiers.add_phone(e.get("requester_phone"))
                    obs.identifiers.add_name(e.get("requester_name"))
                    excerpt = e.get("message_excerpt")
                    tmp = Identifiers()
                    tmp.add_value(excerpt)
                    obs.identifiers.merge(tmp)
                    obs.identifiable = obs.identifiers.has_direct()
                    out.append(obs)
                else:  # unknown group: generic record interpretation
                    obs = observation_from_record(ef.file_id, observed, service, group, e, source=src, snapshot=False)
                    if obs:
                        out.append(obs)
    return out


def _bundle_observations(ef: EvidenceFile, data: dict) -> Tuple[List[Observation], List[SnapshotScope]]:
    """records_assurance snapshot bundle: several point-in-time collection snapshots."""
    out: List[Observation] = []
    scopes: List[SnapshotScope] = []
    for snap in data.get("snapshots") or []:
        captured = snap.get("captured_at") or data.get("exported_at") or ef.captured_at
        store = snap.get("data_store") or ""
        collection = snap.get("collection") or ""
        records = snap.get("records") or []
        src = f"snapshot_bundle:{store}.{collection}@{captured}"
        if collection == "operational_record_history":
            for r in records:
                obs = observation_from_lookup(ef.file_id, r.get("recorded_at") or captured, r.get("system"),
                                              r.get("resource_id"), r.get("record_type"), r.get("result"),
                                              r.get("lookup_value"), source=src)
                if obs:
                    out.append(obs)
            continue
        rids = set()
        kinds = set()
        for r in records:
            obs = observation_from_record(ef.file_id, captured, store, collection, r, source=src, snapshot=True)
            if obs:
                out.append(obs)
                rids.add(obs.resource_id)
                kinds.add(obs.kind)
        system = config.PREFIX_TO_SYSTEM.get(store.split("_service")[0], store)
        if store in config.PREFIX_TO_SYSTEM.values():
            system = store
        scopes.append(SnapshotScope(ef.file_id, captured, system, kinds or {_kind_for_collection(collection)}, rids))
    return out, scopes


def _kind_for_collection(collection: str) -> str:
    from .records import _collection_kind
    return _collection_kind(collection, {})


def _lookup_records(ef: EvidenceFile, data: dict) -> List[Observation]:
    out = []
    default_time = data.get("exported_at") or data.get("captured_at") or ef.captured_at
    src = f"record_lookup:{data.get('origin') or data.get('source') or ef.origin}"
    for r in data.get("records") or []:
        obs = observation_from_lookup(ef.file_id, r.get("recorded_at") or default_time, r.get("system"),
                                      r.get("resource_id"), r.get("record_type"), r.get("result"),
                                      r.get("subject_reference") or r.get("lookup_value"), source=src)
        if obs:
            out.append(obs)
    return out


def parse_json(ef: EvidenceFile) -> Tuple[List[Observation], List[SnapshotScope]]:
    with open(ef.abs_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        return [], []
    if "service_exports" in data:
        return _sre_observations(ef, data), []
    if "snapshots" in data:
        return _bundle_observations(ef, data)
    records = data.get("records")
    if isinstance(records, list):
        if records and isinstance(records[0], dict) and "record_type" in records[0] and "result" in records[0]:
            return _lookup_records(ef, data), []
        captured = data.get("captured_at") or data.get("exported_at") or ef.captured_at
        store = data.get("source") or data.get("origin") or ""
        collection = data.get("collection") or ""
        out = []
        rids = set()
        kinds = set()
        for r in records:
            obs = observation_from_record(ef.file_id, captured, store, collection, r,
                                          source=f"json_collection:{collection}", snapshot=True)
            if obs:
                out.append(obs)
                rids.add(obs.resource_id)
                kinds.add(obs.kind)
        scope = []
        if out:
            scope.append(SnapshotScope(ef.file_id, captured, out[0].system_id, kinds, rids))
        return out, scope
    return [], []
