"""Parser for SQLite-compatible SQL snapshot exports (primary + replica tables)."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Dict, List, Tuple

from .. import config
from ..loaders import EvidenceFile
from .base import Observation, SnapshotScope
from .records import observation_from_record

CAPTURED_RE = re.compile(r"--\s*Captured at:\s*(\S+)")


def _table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    return [c[1] for c in con.execute(f'PRAGMA table_info("{table}")')]


def _topology(table: str) -> str:
    t = table.lower()
    if "replica" in t:
        return "replica"
    if "primary" in t:
        return "primary"
    return "table"


def parse_sql(ef: EvidenceFile) -> Tuple[List[Observation], List[SnapshotScope]]:
    sql = open(ef.abs_path, encoding="utf-8").read()
    m = CAPTURED_RE.search(sql)
    captured = m.group(1) if m else ef.captured_at
    con = sqlite3.connect(":memory:")
    con.executescript(sql)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]

    scope_times: Dict[str, str] = {}
    if "snapshot_scope" in tables:
        for origin, cap in con.execute("SELECT origin_system, captured_at FROM snapshot_scope"):
            scope_times[origin] = cap
    completed = re.search(r"snapshot_completed_at=(\S+)", sql)
    if completed:
        captured = completed.group(1)

    out: List[Observation] = []
    scopes: List[SnapshotScope] = []
    for table in tables:
        if table == "snapshot_scope":
            continue
        cols = _table_columns(con, table)
        rows = [dict(zip(cols, r)) for r in con.execute(f'SELECT * FROM "{table}"')]
        topo = _topology(table)
        src_base = f"sql:{table}"
        per_system_kinds: Dict[str, set] = {}
        per_system_rids: Dict[str, set] = {}
        if "record_json" in cols:
            for row in rows:
                try:
                    rec = json.loads(row["record_json"])
                except Exception:
                    continue
                observed = row.get("updated_at") or captured
                obs = observation_from_record(ef.file_id, observed, row.get("origin_system") or "",
                                              row.get("collection_name") or "", rec,
                                              source=f"{src_base}:{row.get('collection_name')}", snapshot=True)
                if obs:
                    obs.raw["topology"] = topo
                    out.append(obs)
                    per_system_kinds.setdefault(obs.system_id, set()).add(obs.kind)
                    per_system_rids.setdefault(obs.system_id, set()).add(obs.resource_id)
        else:
            # map inventory / bespoke tables onto the shared record interpreter
            for row in rows:
                rec = dict(row)
                if "user_id" in rec and "customer_reference" not in rec:
                    rec["customer_reference"] = rec.get("user_id")
                if "source_resource_id" in rec:
                    rec.setdefault("index_name", "search")
                observed = row.get("updated_at") or row.get("indexed_at") or scope_times.get(
                    config.system_for_resource(row.get("resource_id", "")), captured)
                obs = observation_from_record(ef.file_id, observed, "", _collection_name(table), rec,
                                              source=src_base, snapshot=True)
                if obs:
                    obs.raw["topology"] = topo
                    if "source_resource_id" in rec:
                        obs.raw["source_resource_id"] = rec["source_resource_id"]
                    out.append(obs)
                    per_system_kinds.setdefault(obs.system_id, set()).add(obs.kind)
                    per_system_rids.setdefault(obs.system_id, set()).add(obs.resource_id)
            # an inventory table with zero rows still proves absence for its collection
            if not rows:
                kind = _table_kind(table)
                system = _table_system(table)
                if kind and system:
                    scopes.append(SnapshotScope(ef.file_id, scope_times.get(system, captured), system, {kind}, set()))
        for system, kinds in per_system_kinds.items():
            scopes.append(SnapshotScope(ef.file_id, scope_times.get(system, captured), system, kinds,
                                        per_system_rids.get(system, set())))
    # snapshot_scope rows name systems that were fully captured even when no rows exist for them
    for origin, cap in scope_times.items():
        if origin in config.PREFIX_TO_SYSTEM.values() and not any(s.system_id == origin for s in scopes):
            kinds = {k for p, k in config.PREFIX_TO_KIND.items() if config.PREFIX_TO_SYSTEM[p] == origin}
            scopes.append(SnapshotScope(ef.file_id, cap, origin, kinds, set()))
    con.close()
    return out, scopes


def _collection_name(table: str) -> str:
    t = table.lower()
    if "search" in t:
        return "customer_search_documents"
    if "analytics" in t:
        return "customer_analytics_profiles"
    if "support" in t:
        return "support_records"
    if "export" in t:
        return "customer_exports"
    if "connector" in t:
        return "connector_configurations"
    if "backup" in t:
        return "backup_catalog"
    if "audit" in t:
        return "security_audit_events"
    if "profile" in t:
        return "customer_profiles"
    return table


def _table_kind(table: str):
    from .records import _collection_kind
    return _collection_kind(_collection_name(table), {})


def _table_system(table: str):
    kind = _table_kind(table)
    for prefix, k in config.PREFIX_TO_KIND.items():
        if k == kind:
            return config.PREFIX_TO_SYSTEM[prefix]
    if kind in (config.KIND_SUPPORT_MESSAGE, config.KIND_SUPPORT_ATTACHMENT):
        return "support_platform"
    return ""
