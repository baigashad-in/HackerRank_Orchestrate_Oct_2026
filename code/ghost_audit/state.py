"""State reconstruction: what did a resource look like at ``audit_at``?"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import config
from .evidence.base import Observation, SnapshotScope
from .loaders import Case


@dataclass
class ResourceState:
    resource_id: str
    system_id: str
    kind: str
    present: Optional[bool] = None
    identifiable: Optional[bool] = None
    enabled: Optional[bool] = None
    restore_allowed: Optional[bool] = None
    available: Optional[bool] = None
    retention_until: Optional[str] = None
    hold_active: Optional[bool] = None
    evidence_id: str = ""              # file that proves the state (empty when nothing eligible)
    observed_at: str = ""
    basis: str = ""                    # human-readable explanation
    observations: List[Observation] = field(default_factory=list)
    stale: bool = False                # only evidence from before the request exists


def _eligible(obs: Observation, case: Case) -> bool:
    """Usable for this case: observed no later than audit_at (record-level time)."""
    return bool(obs.observed_at) and obs.observed_at <= case.audit_at


def _citable(obs: Observation, case: Case) -> bool:
    """Citable: the whole file was captured no later than audit_at."""
    return _eligible(obs, case) and (not obs.file_captured_at or obs.file_captured_at <= case.audit_at)


def _rank(obs: Observation, case: Case):
    """Later observations win; among equals prefer citable files, then snapshots over lookups."""
    return (obs.observed_at, 1 if _citable(obs, case) else 0, 1 if obs.snapshot else 0, obs.file_captured_at)


def reconstruct(case: Case, resource_id: str, observations: List[Observation],
                scopes: List[SnapshotScope], kind_hint: str = "") -> ResourceState:
    system_id = config.system_for_resource(resource_id)
    kind = kind_hint or config.kind_for_resource(resource_id)
    st = ResourceState(resource_id, system_id, kind)
    usable = [o for o in observations if _eligible(o, case)]
    # evidence from before the request describes the pre-deletion world; keep it only as a last resort
    fresh = [o for o in usable if o.observed_at >= case.requested_at]
    if fresh:
        usable = fresh
    else:
        st.stale = bool(usable)
    st.observations = sorted(usable, key=lambda o: _rank(o, case))
    if not usable:
        # closed-world snapshots: a collection dump that does not list the record proves absence
        snaps = [s for s in scopes if s.system_id == system_id and (kind in s.kinds or not s.kinds)
                 and s.observed_at <= case.audit_at and s.observed_at >= case.requested_at
                 and resource_id not in s.resource_ids and (not s.file_captured_at or s.file_captured_at <= case.audit_at)]
        if snaps and kind in (config.KIND_PROFILE, config.KIND_SEARCH, config.KIND_EXPORT, config.KIND_MARKETING):
            best = max(snaps, key=lambda s: (s.observed_at, s.file_captured_at))
            st.present = False
            st.evidence_id = best.file_id
            st.observed_at = best.observed_at
            st.basis = "absent from closed-world snapshot"
        return st

    latest_time = max(o.observed_at for o in usable)
    latest = [o for o in usable if o.observed_at == latest_time]
    # presence: any table/replica copy at the latest time counts as present
    presence_votes = [o.present for o in latest if o.present is not None]
    if presence_votes:
        st.present = any(presence_votes)
    else:
        earlier = [o for o in usable if o.present is not None]
        if earlier:
            st.present = max(earlier, key=lambda o: _rank(o, case)).present

    def pick(attr: str):
        vals = [o for o in latest if getattr(o, attr) is not None]
        if not vals:
            vals = [o for o in usable if getattr(o, attr) is not None]
        if not vals:
            return None, None
        best = max(vals, key=lambda o: _rank(o, case))
        return getattr(best, attr), best

    for attr in ("identifiable", "enabled", "restore_allowed", "available", "retention_until", "hold_active"):
        val, src = pick(attr)
        if attr == "identifiable" and val is not None:
            # any copy that still exposes identifiers makes the record identifiable
            exposed = [o for o in latest if o.identifiable is True]
            val = bool(exposed) or val
        setattr(st, attr, val)

    # citation: the decisive observation, preferring files captured before audit_at
    decisive = [o for o in latest]
    if st.present is not None:
        decisive = [o for o in latest if o.present == st.present] or decisive
    if st.identifiable is True:
        decisive = [o for o in decisive if o.identifiable is True] or decisive
    citable = [o for o in decisive if _citable(o, case)]
    chosen = max(citable or decisive, key=lambda o: _rank(o, case))
    st.evidence_id = chosen.file_id
    if not citable:
        # nothing citable at the latest time: fall back to the latest citable observation that agrees
        agree = [o for o in usable if _citable(o, case) and (o.present == st.present or o.present is None)
                 and _agrees(o, st)]
        if agree:
            st.evidence_id = max(agree, key=lambda o: _rank(o, case)).file_id
        elif st.present is False and kind in (config.KIND_PROFILE, config.KIND_SEARCH, config.KIND_EXPORT, config.KIND_MARKETING):
            snap = _absence_snapshot(case, resource_id, system_id, kind, scopes)
            st.evidence_id = snap.file_id if snap else chosen.file_id
            st.basis = (chosen.source + "; absence also proven by closed-world snapshot") if snap else chosen.source
        else:
            # last resort: the record-level eligible file (its entry predates audit_at even if the export is later)
            st.evidence_id = chosen.file_id
    st.observed_at = chosen.observed_at
    st.basis = st.basis or chosen.source
    return st


def _agrees(obs: Observation, st: ResourceState) -> bool:
    for attr in ("identifiable", "enabled", "restore_allowed", "available"):
        ov, sv = getattr(obs, attr), getattr(st, attr)
        if ov is not None and sv is not None and ov != sv:
            return False
    return True


def _absence_snapshot(case: Case, resource_id: str, system_id: str, kind: str, scopes: List[SnapshotScope]):
    snaps = [s for s in scopes if s.system_id == system_id and (kind in s.kinds or not s.kinds)
             and s.observed_at <= case.audit_at and s.observed_at >= case.requested_at
             and resource_id not in s.resource_ids and (not s.file_captured_at or s.file_captured_at <= case.audit_at)]
    if not snaps:
        return None
    return max(snaps, key=lambda s: (s.observed_at, s.file_captured_at))
