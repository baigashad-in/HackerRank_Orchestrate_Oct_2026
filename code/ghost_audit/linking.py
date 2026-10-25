"""Resource <-> user linking.

Three sources of truth, in decreasing strength:

1. **Identifier links** - an evidence observation carries (or was looked up by)
   one of the user's current or historical identifiers, or a stable user
   reference.  Catalog links from ``LINK_RESOURCE_IDENTITY`` count here too.
2. **Workflow runs** - the deletion worker processed the request shortly before
   ``reported_at``.  Its ``DELETE_RECORD account_*`` row is the user's account;
   the rows that follow at the worker's cadence are the user's other resources
   (this is how an anonymised analytics profile that no longer carries an
   identifier is still attributed to the right person).
3. **Elimination** - inside one worker run, accounts already pinned to other
   people cannot be this person's.

The worker cadence is learned at run time (``cadence.py``) from the requests
whose account *is* pinned by evidence. On the October 2026 dataset it comes out
as: policy confirmations start N minutes + 41 s after the account step and
follow each other every 6 min 30 s or 7 min 29 s; results and status checks
start N minutes + 52 s later and follow every 18 min 30 s or 19 min 29 s.
``set_cadence`` installs the schedule the functions below use.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple

from .activity import ActivityIndex, parse_ts
from .cadence import DEFAULT_CADENCE, DRIFT, Cadence
from .evidence.base import Observation
from .identity import IdentityIndex, UserProfile
from .loaders import Activity, Case

# How the deletion workflow's own status claim maps onto the worker result of the account step.
STATUS_RESULT = {"COMPLETE": "SUCCESS", "PARTIALLY_COMPLETE": "PARTIAL", "NOT_STARTED": "FAILED"}

ACCOUNT_WINDOW_BEFORE = timedelta(hours=3)
ACCOUNT_WINDOW_AFTER = timedelta(minutes=30)
CHAIN_MAX_SPAN = timedelta(hours=6)
ORPHAN_WINDOW_BEFORE = timedelta(hours=2)
ORPHAN_WINDOW_AFTER = timedelta(minutes=30)


def row_family(row: Activity) -> str:
    return _ACTIVE.family(row.details)


def _build_lattice(base: int, steps: Tuple[int, ...], max_k: int = 6) -> Dict[int, Dict[int, int]]:
    """seconds-of-minute offset -> {chain position: rounding drift needed to reach it}."""
    lattice: Dict[int, Dict[int, int]] = defaultdict(dict)
    frontier = {base}
    for k in range(max_k + 1):
        for s in frontier:
            for d in range(-DRIFT, DRIFT + 1):
                sec = (s + d) % 60
                prev = lattice[sec].get(k)
                if prev is None or abs(d) < prev:
                    lattice[sec][k] = abs(d)
        frontier = {(s + x) % 60 for s in frontier for x in steps}
    return lattice


_ACTIVE: Cadence = DEFAULT_CADENCE
LATTICE: Dict[str, Dict[int, Dict[int, int]]] = {}


def set_cadence(cadence: Cadence) -> None:
    """Install the worker schedule (learned or default) used by every function in this module."""
    global _ACTIVE, LATTICE
    _ACTIVE = cadence
    LATTICE = {fam: _build_lattice(base, cadence.step_seconds) for fam, base in cadence.family_base.items()}


def active_cadence() -> Cadence:
    return _ACTIVE


set_cadence(DEFAULT_CADENCE)


def positions_for(row: Activity, delta_seconds: int) -> Dict[int, int]:
    """{position: drift} for the offset from the account step (empty when off-cadence)."""
    if delta_seconds <= 0:
        return {}
    return dict(LATTICE.get(row_family(row), {}).get(delta_seconds % 60, {}))


def lattice_position(row: Activity, delta_seconds: int) -> Optional[int]:
    """Smallest chain position implied by the offset from the account step, or None when off-cadence."""
    if delta_seconds <= 0:
        return None
    if delta_seconds < 120:
        return 0   # immediate follow-up (older worker style): no cadence information
    pos = positions_for(row, delta_seconds)
    return min(pos) if pos else None


def plausible_offset(row: Activity, delta_seconds: int, pos: int) -> bool:
    lo, hi, step_lo, step_hi = _ACTIVE.first_offset[row_family(row)]
    minutes = delta_seconds / 60.0
    return lo + pos * step_lo <= minutes <= hi + pos * step_hi


def fit_score(row: Activity, delta_seconds: int) -> float:
    """Reward rows that sit on the cadence; exact seconds and few unseen steps are the strongest evidence."""
    if delta_seconds <= 0:
        return -3.0
    if delta_seconds < 120:
        return 1.0
    pos = positions_for(row, delta_seconds)
    if not pos:
        return -3.0
    k = min(pos)
    score = 3.0 - k - pos[k]
    if not plausible_offset(row, delta_seconds, k):
        score -= 4.0
    return score


def chain_positions(account: Activity, rows: List[Activity]) -> List[Tuple[Activity, int]]:
    """Assign increasing cadence positions to pinned rows (same rule as sequential_fit)."""
    t0 = parse_ts(account.event_time)
    out: List[Tuple[Activity, int]] = []
    prev_pos = -1
    for w in sorted(rows, key=lambda a: a.event_time):
        d = int((parse_ts(w.event_time) - t0).total_seconds())
        if d < 120:
            out.append((w, prev_pos + 1))
            prev_pos += 1
            continue
        pos = positions_for(w, d)
        cands = [k for k in pos if k > prev_pos]
        k = min(cands, key=lambda x: (pos[x], x)) if cands else prev_pos + 1
        out.append((w, k))
        prev_pos = k
    return out


def insertion_fit(account: Activity, placed: List[Tuple[Activity, int]], row: Activity) -> Optional[float]:
    """Score for inserting an unlabeled row into a chain whose pinned rows already hold positions.

    Returns None when the row cannot sit on the cadence between its neighbours.
    """
    t0 = parse_ts(account.event_time)
    t = parse_ts(row.event_time)
    d = int((t - t0).total_seconds())
    if d <= 0:
        return None
    if d < 120:
        return 1.0 if not placed else None
    pos = positions_for(row, d)
    if not pos:
        return None
    before = [(w, k) for w, k in placed if parse_ts(w.event_time) < t]
    after = [(w, k) for w, k in placed if parse_ts(w.event_time) > t]
    lo_pos = max((k for _, k in before), default=-1)
    hi_pos = min((k for _, k in after), default=10 ** 6)
    cands = [k for k in pos if lo_pos < k < hi_pos]
    if not cands:
        return None
    gmin, gmax = _ACTIVE.gaps[row_family(row)]
    best: Optional[float] = None
    for k in sorted(cands):
        ok = True
        if before:
            w_prev, k_prev = max(before, key=lambda x: x[0].event_time)
            n = k - k_prev
            gap = int((t - parse_ts(w_prev.event_time)).total_seconds())
            if not (n * gmin - 2 * n <= gap <= n * gmax + 2 * n):
                ok = False
        if after and ok:
            w_next, k_next = min(after, key=lambda x: x[0].event_time)
            n = k_next - k
            gap = int((parse_ts(w_next.event_time) - t).total_seconds())
            if not (n * gmin - 2 * n <= gap <= n * gmax + 2 * n):
                ok = False
        if not ok:
            continue
        score = 3.0 - k - pos[k]
        if not before and not plausible_offset(row, d, k):
            score -= 4.0
        if before or after:
            score += 1.0   # anchored by a pinned neighbour on the cadence
        best = score if best is None else max(best, score)
    return best


def sequential_fit(account: Activity, rows: List[Activity]) -> float:
    """Fit a whole chain: positions must increase along time, each step at most a few gaps apart."""
    t0 = parse_ts(account.event_time)
    score = 0.0
    prev_pos = -1
    for w in sorted(rows, key=lambda a: a.event_time):
        d = int((parse_ts(w.event_time) - t0).total_seconds())
        if d <= 0:
            score -= 3.0
            continue
        if d < 120:
            score += 1.0
            continue
        pos = positions_for(w, d)
        cands = [k for k in pos if k > prev_pos]
        if not cands:
            score -= 3.0
            prev_pos += 1
            continue
        k = min(cands, key=lambda x: (pos[x], x))
        skipped = k - prev_pos - 1
        score += 3.0 - min(skipped, 3) - pos[k]
        if prev_pos == -1 and not plausible_offset(w, d, k):
            score -= 4.0
        prev_pos = k
    return score


@dataclass
class LinkResult:
    user: UserProfile
    account_ids: List[str] = field(default_factory=list)
    account_activity: Optional[Activity] = None
    account_method: str = ""
    notes: List[str] = field(default_factory=list)


class Linker:
    def __init__(self, idx: IdentityIndex, act: ActivityIndex, observations: List[Observation]) -> None:
        self.idx = idx
        self.act = act
        self.observations = observations
        self.obs_by_resource: Dict[str, List[Observation]] = defaultdict(list)
        for o in observations:
            self.obs_by_resource[o.resource_id].append(o)
        self.resource_users: Dict[str, Set[str]] = defaultdict(set)
        for o in observations:
            for uid in idx.users_for(o.all_identifiers()):
                self.resource_users[o.resource_id].add(uid)
        for link in act.links:
            self.resource_users[link.resource_id].add(link.user_id)
        self.user_resources: Dict[str, Set[str]] = defaultdict(set)
        for rid, users in self.resource_users.items():
            for uid in users:
                self.user_resources[uid].add(rid)
        self.account_rows = [a for a in act.worker_rows
                             if a.event_type == "DELETE_RECORD" and a.target_system == "account_service"]
        self.assigned_accounts: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def anchors(self, cases: List[Case], users: Dict[str, Optional[UserProfile]]):
        """(user_id, account row, pinned worker rows) for requests whose account an identifier pins down."""
        out = []
        for case in cases:
            user = users.get(case.case_id)
            if user is None:
                continue
            anchored = [a for a in self.account_candidates(case) if user.user_id in self.resource_users.get(a.resource_id, set())]
            if not anchored:
                continue
            head = max(anchored, key=lambda a: (self.status_matches(case, a), a.event_time))
            rows = [w for w in self.user_chain_events(user, head) if w.source_system == "deletion_worker"]
            out.append((user.user_id, head, rows))
        return out

    def identifier_resources(self, user: UserProfile) -> Set[str]:
        return set(self.user_resources.get(user.user_id, set()))

    def account_candidates(self, case: Case) -> List[Activity]:
        rep = parse_ts(case.reported_at)
        lo, hi = rep - ACCOUNT_WINDOW_BEFORE, rep + ACCOUNT_WINDOW_AFTER
        return [a for a in self.account_rows if lo <= parse_ts(a.event_time) <= hi]

    def user_chain_events(self, user: UserProfile, after: Activity) -> List[Activity]:
        """Rows (any source) for resources already pinned to the user that follow ``after`` within one run."""
        t0 = parse_ts(after.event_time)
        rows = []
        for rid in self.identifier_resources(user):
            if rid.startswith("account_"):
                continue   # the account's own follow-up rows fit any owner: no discriminating power
            for w in self.act.by_resource.get(rid, []):
                if w is after:
                    continue
                if w.source_system not in ("deletion_worker", "account_service", "integration_service", "backup_service"):
                    continue
                if w.event_type in ("INDEX_RECORD", "UPDATE_RECORD", "REFRESH_CACHE", "SEND_EMAIL", "CREATE_EXPORT",
                                    "LINK_RESOURCE_IDENTITY", "CLASSIFY_REQUEST"):
                    continue
                tw = parse_ts(w.event_time)
                if t0 < tw <= t0 + CHAIN_MAX_SPAN:
                    rows.append(w)
        rows.sort(key=lambda a: a.event_time)
        return rows

    def chain_score(self, account: Activity, user: UserProfile) -> Tuple[float, int]:
        rows = self.user_chain_events(user, account)
        if not rows:
            return 0.0, 0
        return sequential_fit(account, rows), len(rows)

    def status_score(self, case: Case, account: Activity) -> float:
        expected = STATUS_RESULT.get(case.reported_status)
        if expected == account.reported_result:
            return 10.0
        if case.reported_status == "PARTIALLY_COMPLETE" and account.reported_result == "SUCCESS":
            return 5.0
        return -10.0

    def status_matches(self, case: Case, account: Activity) -> bool:
        return STATUS_RESULT.get(case.reported_status) == account.reported_result

    # ------------------------------------------------------------------
    def assign_accounts(self, cases: List[Case], users: Dict[str, Optional[UserProfile]]) -> Dict[str, Tuple[Optional[Activity], str]]:
        """Pin one account activity to every deletion case (identifier links first, then runs)."""
        result: Dict[str, Tuple[Optional[Activity], str]] = {}
        pending: List[Case] = []
        taken: Set[str] = set()
        for case in cases:
            user = users.get(case.case_id)
            if user is None:
                result[case.case_id] = (None, "no-user")
                continue
            cands = self.account_candidates(case)
            anchored = [a for a in cands if user.user_id in self.resource_users.get(a.resource_id, set())]
            if anchored:
                best = max(anchored, key=lambda a: (self.status_matches(case, a), a.event_time))
                result[case.case_id] = (best, "identifier")
                taken.add(best.resource_id)
                continue
            # accounts known from evidence that already existed at audit_at (a later re-registration is
            # a different record and must not be pulled into an earlier request)
            known = [r for r in self.identifier_resources(user) if r.startswith("account_")
                     and any(o.observed_at <= case.audit_at for o in self.obs_by_resource.get(r, []))]
            if known and not cands:
                result[case.case_id] = (None, "identifier-no-activity:" + ",".join(sorted(known)))
                continue
            pending.append(case)
        by_run: Dict[str, List[Case]] = defaultdict(list)
        for case in pending:
            by_run[case.reported_at[:10]].append(case)
        for day, run_cases in sorted(by_run.items()):
            cand_pool: Dict[str, Activity] = {}
            for case in run_cases:
                for a in self.account_candidates(case):
                    if a.resource_id in taken or self.resource_users.get(a.resource_id):
                        continue
                    cand_pool[a.resource_id] = a
            scored: List[Tuple[float, str, str]] = []
            for case in run_cases:
                user = users[case.case_id]
                rep = parse_ts(case.reported_at)
                for rid, a in cand_pool.items():
                    s = self.status_score(case, a)
                    chain, n = self.chain_score(a, user)
                    s += 2.0 * chain
                    delta = abs((rep - parse_ts(a.event_time)).total_seconds())
                    if delta <= 600:
                        s += 4.0
                    s -= delta / 36000.0
                    scored.append((s, case.case_id, rid))
            scored.sort(key=lambda x: (-x[0], x[1], x[2]))
            used_cases: Set[str] = set()
            for s, cid, rid in scored:
                if cid in used_cases or rid in taken:
                    continue
                if s < 0.0:
                    continue
                result[cid] = (cand_pool[rid], f"run-assignment(score={s:.1f})")
                used_cases.add(cid)
                taken.add(rid)
            for case in run_cases:
                if case.case_id not in result:
                    result[case.case_id] = (None, "unresolved")
        self.assigned_accounts = {a.resource_id: cid for cid, (a, _) in result.items() if a is not None}
        return result
