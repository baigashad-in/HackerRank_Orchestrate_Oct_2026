"""End-to-end orchestration: dataset -> evidence -> identity -> linking -> state -> plan -> rows."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import config
from .activity import ActivityIndex, parse_ts
from .evidence import parse_all_evidence
from .evidence.base import Observation
from .identity import IdentityIndex, UserProfile
from .cadence import DEFAULT_CADENCE, Cadence, learn_cadence
from .linking import (CHAIN_MAX_SPAN, ORPHAN_WINDOW_AFTER, ORPHAN_WINDOW_BEFORE, Linker, chain_positions,
                      insertion_fit, set_cadence)
from .loaders import Activity, Case, Dataset, load_cases
from .exposure import exposed_fields, lookup_fields
from .planner import ResourcePlan, Step, assemble_steps, decide_treatment, plan_resource
from .policy import Policy, default_policy_path
from .request_classifier import Classification, classify_request
from .state import ResourceState, reconstruct


@dataclass
class CaseResult:
    case: Case
    status: str
    steps: List[Step]
    user: Optional[UserProfile] = None
    classification: Optional[Classification] = None
    account_id: Optional[str] = None
    account_method: str = ""
    plans: List[ResourcePlan] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def steps_json(self) -> str:
        return json.dumps([s.to_json(i + 1) for i, s in enumerate(self.steps)], separators=(",", ":"), ensure_ascii=False)

    def row(self) -> Tuple[str, str, str]:
        return self.case.case_id, self.status, self.steps_json()


class AuditEngine:
    def __init__(self, ds: Dataset, *, verbose: bool = False, image_reader=None, classifier=None,
                 cadence: Optional[str] = "learned") -> None:
        self.ds = ds
        self.verbose = verbose
        self.idx = IdentityIndex(ds.identities, ds.audit_log)
        self.act = ActivityIndex(ds.activity)
        # resource type -> system names: confirmed against the systems the deletion worker targets
        self.system_map_confirmed, self.system_map_warnings = config.check_system_map(self.act.worker_rows)
        obs, scopes = parse_all_evidence(ds.evidence_files, verbose=verbose, image_reader=image_reader,
                                         known_emails=self.idx.all_emails(), aliases=self.act.alias_to_resource)
        self.observations = self._remap_aliases(obs)
        self.scopes = scopes
        self.obs_by_resource: Dict[str, List[Observation]] = defaultdict(list)
        for o in self.observations:
            self.obs_by_resource[o.resource_id].append(o)
        self.linker = Linker(self.idx, self.act, self.observations)
        self.cadence: Cadence = self._learn_cadence() if cadence == "learned" else DEFAULT_CADENCE
        set_cadence(self.cadence)
        self.classifier = classifier or classify_request
        self.evidence_by_id = ds.evidence_by_id
        image_ids = [e.file_id for e in ds.evidence_files if e.ext in (".png", ".jpg", ".jpeg", ".webp")]
        read_ids = {o.file_id for o in obs}
        self.unread_images: List[str] = [fid for fid in image_ids if fid not in read_ids]
        self.image_count = len(image_ids)
        self.used_model_reader = image_reader is not None
        self.unverifiable: List[str] = []
        # deletion policy: always the rules file of the dataset being audited
        self.policy = (Policy.from_rules(ds.rules, source="deletion_rules.csv") if ds.rules
                       else Policy.load(default_policy_path()))
        self.retained_missing: List[Tuple[str, str]] = []      # (case_id, resource_id)
        self.policy_fallbacks: List[Tuple[str, str, str]] = []  # (case_id, resource_id, note)

    def warnings(self) -> List[str]:
        """Problems a person must know about before trusting output.csv."""
        out: List[str] = []
        if self.unread_images:
            from .evidence.images import OCR_STATUS

            reason = OCR_STATUS.get("reason") or "no recognisable record in the text"
            out.append(f"{len(self.unread_images)} of {self.image_count} screenshots produced no usable evidence "
                       f"({reason}). Run  python code/main.py --doctor  and fix OCR (or a vision model) before submitting.")
        out.extend(self.policy.warnings)
        out.extend(self.system_map_warnings)
        if self.cadence.source.startswith("default:"):
            out.append(f"worker schedule not learned ({self.cadence.source}); measured defaults used for "
                       f"attributing unlabeled worker rows")
        if self.retained_missing:
            shown = ", ".join(f"{c}:{r}" for c, r in self.retained_missing[:10])
            out.append(f"{len(self.retained_missing)} record(s) the policy requires to keep are absent or unavailable "
                       f"at audit_at; those cases are INCOMPLETE with the presence check that must pass: {shown}"
                       + (" ..." if len(self.retained_missing) > 10 else ""))
        if self.policy_fallbacks:
            shown = ", ".join(f"{c}:{r}" for c, r, _ in self.policy_fallbacks[:10])
            out.append(f"{len(self.policy_fallbacks)} record(s) lost their retention basis (retention ended, hold "
                       f"released or no rule) and are treated as DELETE: {shown}"
                       + (" ..." if len(self.policy_fallbacks) > 10 else ""))
        if self.unverifiable:
            out.append(f"{len(self.unverifiable)} deletion case(s) had no evidence-backed check and were marked INCOMPLETE "
                       f"with the checks that must pass: {', '.join(self.unverifiable[:15])}"
                       + (" ..." if len(self.unverifiable) > 15 else ""))
        return out

    # ------------------------------------------------------------------
    def _learn_cadence(self) -> Cadence:
        """Learn the worker schedule from every request file of the dataset (requests only, never answers).

        Using all request files keeps the schedule the same whichever subset of cases is audited.
        """
        import os

        seen, cases = set(), []
        for name in ("deletion_requests.csv", "sample_deletion_requests.csv"):
            path = os.path.join(self.ds.dataset_dir, name) if getattr(self.ds, "dataset_dir", "") else ""
            for c in (load_cases(path) if path and os.path.exists(path) else []):
                if c.case_id not in seen:
                    seen.add(c.case_id)
                    cases.append(c)
        for c in self.ds.cases:
            if c.case_id not in seen:
                seen.add(c.case_id)
                cases.append(c)
        users = {c.case_id: self.idx.resolve_email(c.email) for c in cases}
        deletion = [c for c in cases if users[c.case_id] is not None and classify_request(c.request_text).authorises_deletion]
        anchors = self.linker.anchors(deletion, users)
        return learn_cadence(anchors, self.act.worker_rows, lambda rid: self.linker.resource_users.get(rid, set()),
                             parse_ts)

    def _remap_aliases(self, observations: List[Observation]) -> List[Observation]:
        out = []
        for o in observations:
            if config.RESOURCE_ID_RE.fullmatch(o.resource_id):
                out.append(o)
                continue
            alias = o.raw.get("alias") if isinstance(o.raw, dict) else None
            target = self.act.alias_to_resource.get(o.resource_id) or (alias and self.act.alias_to_resource.get(alias))
            if target:
                o.resource_id = target
                o.system_id = config.system_for_resource(target) or o.system_id
                out.append(o)
        return out

    # ------------------------------------------------------------------
    def run(self, cases: Optional[List[Case]] = None) -> List[CaseResult]:
        cases = list(cases if cases is not None else self.ds.cases)
        users: Dict[str, Optional[UserProfile]] = {}
        classifications: Dict[str, Classification] = {}
        deletion_cases: List[Case] = []
        for case in cases:
            cls = self.classifier(case.request_text)
            users[case.case_id] = self.idx.resolve_email(case.email)
            cls = self._corroborate_classification(case, users[case.case_id], cls)
            classifications[case.case_id] = cls
            if cls.authorises_deletion and users[case.case_id] is not None:
                deletion_cases.append(case)
        accounts = self.linker.assign_accounts(deletion_cases, users)
        chains = self._assign_chains(deletion_cases, users, accounts)
        results: List[CaseResult] = []
        for case in cases:
            cls = classifications[case.case_id]
            user = users[case.case_id]
            if not cls.authorises_deletion:
                results.append(CaseResult(case, "NOT_APPLICABLE", [], user, cls, notes=[cls.reason]))
                continue
            if user is None:
                results.append(CaseResult(case, "COMPLETE", [], None, cls, notes=["unresolved identity"]))
                continue
            acct, method = accounts.get(case.case_id, (None, ""))
            res = self._audit_case(case, user, cls, acct, method, chains.get(case.case_id, []))
            results.append(res)
        results.sort(key=lambda r: config.case_sort_key(r.case.case_id))
        return results

    # ------------------------------------------------------------------
    def _corroborate_classification(self, case: Case, user: Optional[UserProfile],
                                    cls: Classification) -> Classification:
        """The privacy intake's own classification (CLASSIFY_REQUEST rows) corroborates a weak text verdict."""
        if user is None or cls.confidence >= 0.9:
            return cls
        accounts = [r for r in self.linker.identifier_resources(user) if r.startswith("account_")]
        flagged = [a for a in accounts if a in self.act.non_deletion_accounts
                   and self.act.non_deletion_accounts[a].event_time <= case.audit_at]
        if flagged and cls.authorises_deletion:
            row = self.act.non_deletion_accounts[flagged[0]]
            return Classification(False, 0.85, f"{cls.reason}; privacy intake classified it as non-deletion ({row.activity_id})",
                                  method=cls.method + "+intake")
        return cls

    # ------------------------------------------------------------------
    def _assign_chains(self, cases: List[Case], users: Dict[str, Optional[UserProfile]],
                       accounts: Dict[str, Tuple[Optional[Activity], str]]) -> Dict[str, List[Activity]]:
        """Attribute unlabeled worker rows to the request run they belong to."""
        chains: Dict[str, List[Activity]] = {}
        claims: Dict[str, List[Tuple[float, str, Activity]]] = defaultdict(list)   # resource -> (score, case, row)
        orphan_cases: List[Case] = []
        for case in cases:
            acct, _ = accounts.get(case.case_id, (None, ""))
            user = users[case.case_id]
            if user is None:
                chains[case.case_id] = []
                continue
            if acct is None:
                chains[case.case_id] = []
                orphan_cases.append(case)
                continue
            pinned = self.linker.user_chain_events(user, acct)
            chains[case.case_id] = list(pinned)
            placed = chain_positions(acct, pinned)
            t0 = parse_ts(acct.event_time)
            for w in self.act.worker_rows:
                if w.target_system == "account_service" or self.linker.resource_users.get(w.resource_id):
                    continue
                tw = parse_ts(w.event_time)
                if not (t0 < tw <= t0 + CHAIN_MAX_SPAN):
                    continue
                score = insertion_fit(acct, placed, w)
                if score is None:
                    continue
                claims[w.resource_id].append((score, case.case_id, w))
        for rid, cl in claims.items():
            cl.sort(key=lambda x: (-x[0], x[1]))
            best_score, best_case, row = cl[0]
            if best_score < 2.0:
                continue
            if len(cl) > 1 and cl[1][0] == best_score:
                continue   # ambiguous between two runs: leave unattributed
            chains[best_case].append(row)
        claimed = {w.resource_id for rows in chains.values() for w in rows}
        # requests whose account row is missing: unlabeled worker rows right around reported_at are theirs
        head_times = [parse_ts(a.event_time) for a, _ in accounts.values() if a is not None]
        for case in orphan_cases:
            rep = parse_ts(case.reported_at)
            lo, hi = rep - ORPHAN_WINDOW_BEFORE, rep + ORPHAN_WINDOW_AFTER
            if any(lo <= t <= hi for t in head_times):
                continue
            if any(c is not case and lo <= parse_ts(c.reported_at) <= hi
                   for c in orphan_cases):
                continue
            for w in self.act.worker_rows:
                if w.target_system == "account_service" or w.resource_id in claimed:
                    continue
                if self.linker.resource_users.get(w.resource_id):
                    continue
                if lo <= parse_ts(w.event_time) <= hi:
                    chains[case.case_id].append(w)
                    claimed.add(w.resource_id)
        for cid in chains:
            chains[cid].sort(key=lambda a: a.event_time)
        return chains

    # ------------------------------------------------------------------
    def _resource_universe(self, case: Case, user: UserProfile, account: Optional[Activity],
                           chain: List[Activity], method: str) -> Dict[str, str]:
        """resource_id -> link reason."""
        universe: Dict[str, str] = {}
        if account is not None:
            universe[account.resource_id] = f"account:{method}"
        elif method.startswith("identifier-no-activity:"):
            for rid in method.split(":", 1)[1].split(","):
                if rid:
                    universe[rid] = "account:identifier"
        for rid in self.linker.identifier_resources(user):
            obs = [o for o in self.obs_by_resource.get(rid, [])
                   if o.observed_at <= case.audit_at and o.observed_at >= case.requested_at]
            if obs:
                universe.setdefault(rid, "identifier")
            elif rid.startswith("account_") and account is not None and rid != account.resource_id:
                continue
        for w in chain:
            universe.setdefault(w.resource_id, "workflow-run")
        # catalog links pinned to this user with fresh evidence
        for link in self.act.links_for_user(user.user_id):
            obs = [o for o in self.obs_by_resource.get(link.resource_id, [])
                   if o.observed_at <= case.audit_at and o.observed_at >= case.requested_at]
            if obs:
                universe.setdefault(link.resource_id, "catalog-link")
        return universe

    def _audit_case(self, case: Case, user: UserProfile, cls: Classification, account: Optional[Activity],
                    method: str, chain: List[Activity]) -> CaseResult:
        universe = self._resource_universe(case, user, account, chain, method)
        plans: List[ResourcePlan] = []
        notes: List[str] = []
        for rid, reason in sorted(universe.items()):
            kind = config.kind_for_resource(rid)
            if kind == config.KIND_OTHER:
                continue
            obs = self.obs_by_resource.get(rid, [])
            # observations that belong to another person's copy of a shared record are ignored
            own_obs = [o for o in obs if self._observation_is_about(o, user, rid)]
            state = reconstruct(case, rid, own_obs, self.scopes, kind_hint=self._kind_from_observations(own_obs, kind))
            workflow_action = self.act.required_action(rid)
            decision = decide_treatment(state, workflow_action, self.policy, case.audit_at)
            treatment = decision.treatment
            if state.stale and reason not in ("workflow-run",) and not rid.startswith("account_"):
                notes.append(f"{rid}: only stale evidence, skipped")
                continue
            if not state.observations and state.evidence_id == "" and not rid.startswith("account_") \
                    and reason != "workflow-run":
                notes.append(f"{rid}: no usable evidence, skipped")
                continue
            if state.present is None and not state.observations and rid.startswith("account_"):
                # account without any eligible evidence: fall back to the workflow's own claim only for ordering;
                # the state stays unknown, which yields a verification step without a finding
                notes.append(f"{rid}: account state unknown (no evidence)")
            self._apply_workflow_hints(case, state, rid)
            plan = plan_resource(case, state, treatment)
            plan.rule_id = decision.rule_id
            plan.rationale = (f"{reason}; treatment={treatment} ({decision.rule_id or 'fallback'}"
                              + (f": {decision.note}" if decision.note else "") + f"); basis={state.basis}")
            if decision.note and not decision.rule_id and state.kind != config.KIND_DATA_FLOW:
                self.policy_fallbacks.append((case.case_id, rid, decision.note))
                notes.append(f"{rid}: {decision.note}")
            lookups = lookup_fields(user, state.observations)
            plan.linked_by = reason + (f"; found by {', '.join(lookups)}" if lookups else "")
            if plan.finding is not None and plan.finding.operation in ("REPORT_EXPOSED_PII", "REPORT_REMAINING_RECORD"):
                plan.exposed = exposed_fields(user, state.observations,
                                              pii_finding=plan.finding.operation == "REPORT_EXPOSED_PII", idx=self.idx)
            if plan.retained_missing:
                self.retained_missing.append((case.case_id, rid))
                notes.append(f"{rid}: required by {decision.rule_id or 'policy'} but absent or unavailable at audit_at")
            plans.append(plan)
        findings = [p for p in plans if p.finding is not None]
        retained_gone = [p for p in plans if p.retained_missing]
        status = "INCOMPLETE" if (findings or retained_gone) else "COMPLETE"
        steps = assemble_steps(status, plans)
        if status == "COMPLETE":
            # verification steps of a COMPLETE case must cite evidence; drop steps we cannot prove
            steps = [s for s in steps if s.evidence_id]
        if status == "COMPLETE" and not steps:
            # Nothing could be proven (typically: the only evidence is a screenshot that could not be read).
            # A deletion that cannot be verified is not complete: emit the checks that must pass, flag the case.
            check_plans = plans or self._catalog_plans(case, user)
            status = "INCOMPLETE"
            steps = assemble_steps("INCOMPLETE", check_plans)
            notes.append("no evidence-backed verification was possible; emitted the required checks as INCOMPLETE")
            self.unverifiable.append(case.case_id)
        res = CaseResult(case, status, steps, user, cls, account.resource_id if account else None, method, plans, notes)
        return res

    def _catalog_plans(self, case: Case, user: UserProfile) -> List[ResourcePlan]:
        """Resources the data catalog links to this person during the request window (no state known)."""
        plans: List[ResourcePlan] = []
        for link in self.act.links_for_user(user.user_id):
            if not (case.requested_at <= link.event_time <= case.audit_at):
                continue
            rid = link.resource_id
            kind = config.kind_for_resource(rid)
            if kind == config.KIND_OTHER or not config.system_for_resource(rid):
                continue
            state = ResourceState(rid, config.system_for_resource(rid), kind)
            decision = decide_treatment(state, self.act.required_action(rid), self.policy, case.audit_at)
            plan = plan_resource(case, state, decision.treatment)
            plan.rule_id = decision.rule_id
            plans.append(plan)
        return plans

    def _observation_is_about(self, obs: Observation, user: UserProfile, rid: str) -> bool:
        ids = obs.all_identifiers()
        if ids.is_empty():
            return True
        linked = self.idx.users_for(ids)
        return (user.user_id in linked) or not linked

    def _kind_from_observations(self, obs: List[Observation], default: str) -> str:
        kinds = [o.kind for o in obs if o.kind and o.kind != config.KIND_SUPPORT]
        if not kinds:
            return default
        # prefer the most specific support sub-type seen
        if config.KIND_SUPPORT_ATTACHMENT in kinds:
            return config.KIND_SUPPORT_ATTACHMENT
        if config.KIND_SUPPORT_MESSAGE in kinds:
            return config.KIND_SUPPORT_MESSAGE
        return kinds[-1]

    def _apply_workflow_hints(self, case: Case, state: ResourceState, rid: str) -> None:
        """Activity rows are claims, but a 'check' row is direct evidence that a risk is still open."""
        def in_window(a: Activity) -> bool:
            return case.requested_at <= a.event_time <= case.audit_at

        if state.kind == config.KIND_DATA_FLOW and state.enabled is None:
            for a in self.act.worker_events_for(rid):
                if a.event_type == "CHECK_DATA_FLOW" and "remains enabled" in a.details and in_window(a):
                    state.enabled = True
                    state.evidence_id = state.evidence_id or a.activity_id
        if state.kind == config.KIND_BACKUP and state.restore_allowed is None:
            for a in self.act.worker_events_for(rid):
                if a.event_type == "CHECK_RESTORE_ACCESS" and "can still be restored" in a.details and in_window(a):
                    state.restore_allowed = True
                    state.evidence_id = state.evidence_id or a.activity_id
