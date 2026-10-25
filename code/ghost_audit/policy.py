"""Deletion policy, read from ``dataset/deletion_rules.csv`` at run time.

The engine does not carry its own copy of the policy. Every treatment comes
from a row of the rules file:

* ``data_category`` is mapped to the record kinds the evidence parsers produce
  (``CATEGORY_KINDS`` below). This table translates vocabulary only; it holds
  no decisions.
* ``required_action`` becomes the treatment (DELETE, REDACT, ANONYMIZE,
  RETAIN). A retained backup archive becomes BLOCK, because the contract makes
  a retained backup safe by blocking its restore path.
* ``condition`` is checked against the reconstructed record state when the
  evidence can decide it (retention period, active legal hold). Conditions the
  evidence cannot test are applied as written, and conditions the engine does
  not recognise are applied as written with a warning, so a changed policy is
  never ignored silently.

When no rule applies (a category is missing from the file, or the only rule's
condition is false, for example an audit log whose retention period ended),
the record is personal data without a retention basis and is deleted. Each
such fallback is reported in ``Policy.warnings`` / ``Policy.notes``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import config

# data_category -> record kinds produced by the evidence parsers (vocabulary mapping only)
CATEGORY_KINDS: Dict[str, Tuple[str, ...]] = {
    "USER_PROFILE": (config.KIND_PROFILE,),
    "SECONDARY_IDENTIFIER": (config.KIND_CONTACT,),
    "SEARCH_INDEX_ENTRY": (config.KIND_SEARCH,),
    "CACHE_ENTRY": (config.KIND_CACHE,),
    "SUPPORT_MESSAGE": (config.KIND_SUPPORT_MESSAGE, config.KIND_SUPPORT),
    "SUPPORT_ATTACHMENT": (config.KIND_SUPPORT_ATTACHMENT,),
    "ANALYTICS_PROFILE": (config.KIND_ANALYTICS,),
    "PRODUCT_TELEMETRY": (config.KIND_TELEMETRY,),
    "USER_EXPORT": (config.KIND_EXPORT,),
    "MARKETING_PROFILE": (config.KIND_MARKETING,),
    "EMAIL_DELIVERY_LOG": (config.KIND_EMAIL_DELIVERY,),
    "SECURITY_AUDIT_LOG": (config.KIND_AUDIT,),
    "LEGAL_HOLD_RECORD": (config.KIND_LEGAL_HOLD,),
    "BACKUP_ARCHIVE": (config.KIND_BACKUP,),
    "ACCOUNT_TOMBSTONE": (config.KIND_TOMBSTONE,),
}

ACTIONS = ("DELETE", "REDACT", "ANONYMIZE", "RETAIN")

# Data flows are processes, not stored personal data: the contract always disables them.
PROCESS_KINDS = {config.KIND_DATA_FLOW: "DISABLE"}


@dataclass
class PolicyRule:
    rule_id: str
    data_category: str
    condition: str
    required_action: str


@dataclass
class Decision:
    treatment: str
    rule_id: str = ""            # the rules-file row that decided it ("" for a fallback)
    note: str = ""               # why, when the rule's condition changed the outcome


# ---------------------------------------------------------------------------
# condition checks
# ---------------------------------------------------------------------------
def _before(ts: Optional[str], cutoff: str) -> Optional[bool]:
    if not ts:
        return None
    return ts < cutoff


def check_condition(condition: str, state, audit_at: str) -> Tuple[Optional[bool], bool]:
    """(holds, understood). ``holds`` is None when the evidence cannot decide it (the rule then applies)."""
    text = (condition or "").strip().lower()
    if text in ("", "always"):
        return True, True
    if "retention period" in text:
        ended = _before(getattr(state, "retention_until", None), audit_at)
        return (None if ended is None else not ended), True
    if "legal hold" in text and "active" in text:
        active = getattr(state, "hold_active", None)
        ended = _before(getattr(state, "retention_until", None), audit_at)
        if active is False or ended is True:
            return False, True
        return (True if active else None), True
    # conditions that describe the record or the remediation and are satisfied by the treatment itself
    descriptive = ("restore access can be blocked", "still attributable", "contains a user email",
                   "belongs only to", "shared customer", "agent, or order context", "account is deleted",
                   "stable internal user_id")
    if any(p in text for p in descriptive):
        return None, True
    return None, False


class Policy:
    def __init__(self, rules: List[PolicyRule], source: str = "") -> None:
        self.rules = rules
        self.source = source
        self.warnings: List[str] = []
        self.by_kind: Dict[str, List[PolicyRule]] = {}
        unknown_actions = []
        for r in rules:
            action = r.required_action.strip().upper()
            if action not in ACTIONS:
                unknown_actions.append(f"{r.rule_id} ({r.required_action})")
                continue
            kinds = CATEGORY_KINDS.get(r.data_category.strip().upper())
            if not kinds:
                self.warnings.append(f"policy: {r.rule_id} uses data category {r.data_category!r}, which no evidence "
                                     f"parser produces; the rule cannot be applied")
                continue
            for k in kinds:
                self.by_kind.setdefault(k, []).append(r)
            understood = check_condition(r.condition, None, "")[1]
            if not understood:
                self.warnings.append(f"policy: condition of {r.rule_id} is not recognised ({r.condition!r}); "
                                     f"the rule is applied as written")
        if unknown_actions:
            self.warnings.append("policy: unsupported required_action in " + ", ".join(unknown_actions)
                                 + f"; supported actions are {', '.join(ACTIONS)}")
        missing = sorted(c for c, ks in CATEGORY_KINDS.items() if not any(k in self.by_kind for k in ks))
        if missing:
            self.warnings.append("policy: no rule for " + ", ".join(missing)
                                 + "; records of these categories are deleted (no retention basis)")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "Policy":
        if not os.path.exists(path):
            pol = cls([], source=path)
            pol.warnings.insert(0, f"policy: {path} not found; every record is deleted except data flows")
            return pol
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        return cls([PolicyRule(**{k: (r.get(k) or "").strip() for k in PolicyRule.__dataclass_fields__})
                    for r in rows], source=path)

    @classmethod
    def from_rules(cls, rules, source: str = "") -> "Policy":
        return cls([PolicyRule(r.rule_id, r.data_category, r.condition, r.required_action) for r in rules], source)

    # ------------------------------------------------------------------
    def decide(self, kind: str, state=None, audit_at: str = "") -> Decision:
        if kind in PROCESS_KINDS:
            return Decision(PROCESS_KINDS[kind], "", "data flow: the contract disables processes")
        rules = self.by_kind.get(kind, [])
        failed: List[str] = []
        for r in rules:
            holds, _ = check_condition(r.condition, state, audit_at) if state is not None else (None, True)
            if holds is False:
                failed.append(r.rule_id)
                continue
            action = r.required_action.strip().upper()
            treatment = "BLOCK" if (kind == config.KIND_BACKUP and action == "RETAIN") else action
            return Decision(treatment, r.rule_id)
        if failed:
            return Decision("DELETE", "", f"condition of {', '.join(failed)} no longer holds at audit_at "
                                          f"(retention ended or hold released): no retention basis, delete")
        return Decision("DELETE", "", f"no rule for {kind}: no retention basis, delete")

    def summary(self) -> Dict[str, str]:
        """kind -> 'TREATMENT (rule_id)' for the record kinds the policy covers."""
        out = {}
        for kind in sorted(set(self.by_kind) | set(PROCESS_KINDS)):
            d = self.decide(kind)
            out[kind] = f"{d.treatment} ({d.rule_id or 'contract'})"
        return out


def default_policy_path() -> str:
    return os.path.join(config.DATASET_DIR, "deletion_rules.csv")

