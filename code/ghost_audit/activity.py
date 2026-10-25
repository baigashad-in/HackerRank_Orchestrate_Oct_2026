"""Indexes over ``system_activity.csv``.

The activity stream is a set of *claims*: what the deletion worker tried to do,
what synchronisation and restore jobs did, and which catalog links exist.  We
use it for three things:

* the required treatment the workflow itself chose for a resource
  (``DELETE_RECORD`` -> delete, ``REDACT_RECORD`` -> redact, ...);
* grouping the worker's actions into runs so unlabeled resources (an anonymised
  analytics profile, a 404 account) can be attributed to the request that
  triggered them;
* catalog links (``LINK_RESOURCE_IDENTITY``) that pin a resource to a person.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .evidence.base import Identifiers
from .loaders import Activity

ACTION_FOR_EVENT = {
    "DELETE_RECORD": "DELETE",
    "REDACT_RECORD": "REDACT",
    "ANONYMIZE_RECORD": "ANONYMIZE",
    "DISABLE_DATA_FLOW": "DISABLE",
    "BLOCK_RESTORE": "BLOCK",
    "RETAIN_RECORD": "RETAIN",
    "CHECK_DATA_FLOW": "DISABLE",       # the worker only checked: the flow still needs disabling
    "CHECK_RESTORE_ACCESS": "BLOCK",     # the worker only checked: restore access still needs blocking
}

LINK_RE = re.compile(r"linked\s+(\S+)\s+\(([^)]+)\)\s+to\s+(\S+)\s+and\s+(user_\d+)", re.I)
ATTACH_LINK_RE = re.compile(r"confirmed\s+(\S+)\s+\(([^)]+)\)\s+belongs\s+only\s+to\s+(\S+)\s+and\s+(user_\d+)", re.I)
REPOPULATED_RE = re.compile(r"repopulated\s+downstream\s+resource\s+(\S+?)\.?$", re.I)


def parse_ts(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class CatalogLink:
    resource_id: str
    alias: str
    email: str
    user_id: str
    activity_id: str
    event_time: str
    exclusive: bool = False      # "belongs only to" (attachment ownership)


@dataclass
class ActivityIndex:
    rows: List[Activity]
    by_resource: Dict[str, List[Activity]] = field(default_factory=lambda: defaultdict(list))
    worker_rows: List[Activity] = field(default_factory=list)
    links: List[CatalogLink] = field(default_factory=list)
    alias_to_resource: Dict[str, str] = field(default_factory=dict)
    non_deletion_accounts: Dict[str, Activity] = field(default_factory=dict)
    repopulated: Dict[str, List[Activity]] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self) -> None:
        for a in self.rows:
            self.by_resource[a.resource_id].append(a)
            if a.source_system == "deletion_worker":
                self.worker_rows.append(a)
            if a.event_type == "LINK_RESOURCE_IDENTITY":
                m = LINK_RE.search(a.details) or ATTACH_LINK_RE.search(a.details)
                if m:
                    link = CatalogLink(m.group(1), m.group(2), m.group(3).lower(), m.group(4).lower(), a.activity_id,
                                       a.event_time, exclusive="belongs only" in a.details.lower())
                    self.links.append(link)
                    if link.alias != link.resource_id:
                        self.alias_to_resource[link.alias] = link.resource_id
            if a.event_type == "CLASSIFY_REQUEST" and "non-deletion" in a.details.lower():
                self.non_deletion_accounts[a.resource_id] = a
            if a.event_type in ("RESTORE_BACKUP", "SYNC_RECORD"):
                m = REPOPULATED_RE.search(a.details)
                if m:
                    self.repopulated[a.resource_id].append(a)
                    self.repopulated[m.group(1)].append(a)
        self.worker_rows.sort(key=lambda a: a.event_time)

    def worker_events_for(self, resource_id: str) -> List[Activity]:
        return [a for a in self.by_resource.get(resource_id, []) if a.source_system == "deletion_worker"]

    def required_action(self, resource_id: str) -> Optional[str]:
        """The treatment the deletion workflow itself applied or attempted for the resource."""
        actions = [ACTION_FOR_EVENT.get(a.event_type) for a in self.worker_events_for(resource_id)]
        actions = [x for x in actions if x]
        if not actions:
            return None
        # prefer a concrete treatment over a check
        for pref in ("DELETE", "REDACT", "ANONYMIZE", "RETAIN", "DISABLE", "BLOCK"):
            if pref in actions:
                return pref
        return actions[-1]

    def links_for_user(self, user_id: str) -> List[CatalogLink]:
        return [l for l in self.links if l.user_id == user_id]

    def link_for_resource(self, resource_id: str) -> Optional[CatalogLink]:
        for l in self.links:
            if l.resource_id == resource_id:
                return l
        return None

    def worker_rows_between(self, start: datetime, end: datetime) -> List[Activity]:
        return [a for a in self.worker_rows if start <= parse_ts(a.event_time) <= end]


def identifiers_in_text(text: str) -> Identifiers:
    ids = Identifiers()
    ids.add_value(text)
    return ids
