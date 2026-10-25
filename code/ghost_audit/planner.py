"""Policy application and ordered step generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import config
from .loaders import Case
from .policy import Decision, Policy, default_policy_path
from .state import ResourceState

@dataclass
class Step:
    phase: str
    system_id: str
    resource_id: str
    operation: str
    state: str
    evidence_id: str = ""

    def sort_key(self):
        return (config.PHASES.index(self.phase), config.OPERATION_ORDER[self.operation], self.system_id,
                config.resource_sort_key(self.resource_id))

    def to_json(self, step_no: int) -> dict:
        return {"step_no": step_no, "phase": self.phase, "system_id": self.system_id, "resource_id": self.resource_id,
                "operation": self.operation, "state": self.state, "evidence_id": self.evidence_id}


@dataclass
class ResourcePlan:
    state: ResourceState
    treatment: str
    compliant: Optional[bool]
    finding: Optional[Step] = None
    remediation: Optional[Step] = None
    verification: Optional[Step] = None
    rationale: str = ""
    rule_id: str = ""                 # deletion_rules.csv row that decided the treatment
    retained_missing: bool = False    # policy requires the record, but it is absent or unavailable
    exposed: List[str] = field(default_factory=list)   # identifier fields that still expose the user
    linked_by: str = ""               # how the record was tied to the person


_DEFAULT_POLICY: Optional[Policy] = None


def _default_policy() -> Policy:
    global _DEFAULT_POLICY
    if _DEFAULT_POLICY is None:
        _DEFAULT_POLICY = Policy.load(default_policy_path())
    return _DEFAULT_POLICY


# The worker's action on a record tells which sub-type it is when the evidence cannot (a support item that
# the worker deletes is an attachment, one it redacts is a message; an analytics-store record it deletes is a
# contact record keyed by an identifier, one it anonymises is an analytics profile). It only refines the
# record kind; the treatment still comes from deletion_rules.csv for that kind.
_SUPPORT = (config.KIND_SUPPORT, config.KIND_SUPPORT_MESSAGE, config.KIND_SUPPORT_ATTACHMENT)
_ANALYTICS = (config.KIND_ANALYTICS, config.KIND_CONTACT)
SUBTYPE_FROM_WORKER = {
    **{(k, "DELETE"): config.KIND_SUPPORT_ATTACHMENT for k in _SUPPORT},
    **{(k, "REDACT"): config.KIND_SUPPORT_MESSAGE for k in _SUPPORT},
    **{(k, "DELETE"): config.KIND_CONTACT for k in _ANALYTICS},
    **{(k, "ANONYMIZE"): config.KIND_ANALYTICS for k in _ANALYTICS},
}


def decide_treatment(state: ResourceState, workflow_action: Optional[str], policy: Optional[Policy] = None,
                     audit_at: str = "") -> Decision:
    """Treatment required by deletion_rules.csv for the record's kind.

    The workflow's action is a claim about what happened, never about what policy requires; it is used only
    to tell record sub-types apart (see SUBTYPE_FROM_WORKER). For a record kind the engine does not know,
    the worker's action is the only available signal and is used as is.
    """
    policy = policy or _default_policy()
    kind = state.kind
    if kind == config.KIND_OTHER and workflow_action:
        return Decision(workflow_action, "", "unknown record kind: worker's action used")
    sub = SUBTYPE_FROM_WORKER.get((kind, workflow_action or ""))
    decision = policy.decide(sub or kind, state, audit_at)
    if sub and sub != kind and decision.rule_id:
        decision.note = decision.note or f"worker's action identifies the record as {sub}"
    return decision


def plan_resource(case: Case, state: ResourceState, treatment: str) -> ResourcePlan:
    rid, sys_id = state.resource_id, state.system_id
    ev = state.evidence_id
    plan = ResourcePlan(state=state, treatment=treatment, compliant=None)

    if treatment == "DELETE":
        if state.present is True:
            plan.compliant = False
            plan.finding = Step("FINDING", sys_id, rid, "REPORT_REMAINING_RECORD", "PRESENT", ev)
            plan.remediation = Step("REMEDIATION", sys_id, rid, "DELETE_RECORD", "ABSENT")
        elif state.present is False:
            plan.compliant = True
        plan.verification = Step("VERIFICATION", sys_id, rid, "VERIFY_RECORD_ABSENT", "ABSENT", ev)
    elif treatment == "REDACT":
        if state.present is False:
            plan.compliant = True
            plan.verification = Step("VERIFICATION", sys_id, rid, "VERIFY_RECORD_REDACTED", "REDACTED", ev)
        elif state.identifiable is True:
            plan.compliant = False
            plan.finding = Step("FINDING", sys_id, rid, "REPORT_EXPOSED_PII", "PRESENT", ev)
            plan.remediation = Step("REMEDIATION", sys_id, rid, "REDACT_RECORD", "REDACTED")
            plan.verification = Step("VERIFICATION", sys_id, rid, "VERIFY_RECORD_REDACTED", "REDACTED", ev)
        else:
            plan.compliant = True if state.identifiable is False else None
            plan.verification = Step("VERIFICATION", sys_id, rid, "VERIFY_RECORD_REDACTED", "REDACTED", ev)
    elif treatment == "ANONYMIZE":
        if state.present is False:
            plan.compliant = True
            plan.verification = Step("VERIFICATION", sys_id, rid, "VERIFY_RECORD_ANONYMIZED", "ANONYMIZED", ev)
        elif state.identifiable is True:
            plan.compliant = False
            plan.finding = Step("FINDING", sys_id, rid, "REPORT_EXPOSED_PII", "PRESENT", ev)
            plan.remediation = Step("REMEDIATION", sys_id, rid, "ANONYMIZE_RECORD", "ANONYMIZED")
            plan.verification = Step("VERIFICATION", sys_id, rid, "VERIFY_RECORD_ANONYMIZED", "ANONYMIZED", ev)
        else:
            plan.compliant = True if state.identifiable is False else None
            plan.verification = Step("VERIFICATION", sys_id, rid, "VERIFY_RECORD_ANONYMIZED", "ANONYMIZED", ev)
    elif treatment == "DISABLE":
        if state.enabled is True:
            plan.compliant = False
            plan.finding = Step("FINDING", sys_id, rid, "REPORT_ACTIVE_DATA_FLOW", "ACTIVE", ev)
            plan.remediation = Step("REMEDIATION", sys_id, rid, "DISABLE_DATA_FLOW", "DISABLED")
        elif state.enabled is False:
            plan.compliant = True
        plan.verification = Step("VERIFICATION", sys_id, rid, "VERIFY_DATA_FLOW_DISABLED", "DISABLED", ev)
    elif treatment == "BLOCK":
        if state.restore_allowed is True:
            plan.compliant = False
            plan.finding = Step("FINDING", sys_id, rid, "REPORT_RESTORE_RISK", "ACTIVE", ev)
            plan.remediation = Step("REMEDIATION", sys_id, rid, "BLOCK_RESTORE", "BLOCKED")
        elif state.restore_allowed is False:
            plan.compliant = True
        plan.verification = Step("VERIFICATION", sys_id, rid, "VERIFY_RESTORE_BLOCKED", "BLOCKED", ev)
    elif treatment == "RETAIN":
        plan.verification = Step("VERIFICATION", sys_id, rid, "VERIFY_RECORD_PRESENT", "PRESENT", ev)
        if state.present is False or state.available is False:
            # A record the policy requires to keep is gone or unusable. The contract has no finding or
            # remediation operation for "restore a retained record", so the case cannot be COMPLETE: it is
            # reported INCOMPLETE with the presence check that must pass, and the engine raises a warning.
            plan.compliant = False
            plan.retained_missing = True
        else:
            plan.compliant = True
    return plan


def assemble_steps(status: str, plans: List[ResourcePlan]) -> List[Step]:
    steps: List[Step] = []
    if status == "NOT_APPLICABLE":
        return steps
    if status == "COMPLETE":
        for p in plans:
            if p.verification is not None:
                steps.append(p.verification)
    else:
        for p in plans:
            if p.finding is not None:
                steps.append(p.finding)
            if p.remediation is not None:
                steps.append(p.remediation)
            if p.verification is not None:
                v = p.verification
                steps.append(Step(v.phase, v.system_id, v.resource_id, v.operation, v.state, ""))
    steps.sort(key=lambda s: s.sort_key())
    # de-duplicate (two observations of the same record never produce two identical steps)
    seen = set()
    unique = []
    for s in steps:
        key = (s.phase, s.system_id, s.resource_id, s.operation)
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique
