#!/usr/bin/env python3
"""Offline checks for the policy layer, retained records, exposure reporting and the learned worker schedule.

    python code/evaluation/test_engine_rules.py

No keys, no network. Uses the participant dataset only (no reference answers).
"""

from __future__ import annotations

import copy
import csv
import io
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ghost_audit import config, loaders  # noqa: E402
from ghost_audit.activity import parse_ts  # noqa: E402
from ghost_audit.cadence import DEFAULT_CADENCE, learn_cadence  # noqa: E402
from ghost_audit.request_classifier import classify_request  # noqa: E402
from ghost_audit.evidence.base import Identifiers  # noqa: E402
from ghost_audit.evidence.records import observation_from_record  # noqa: E402
from ghost_audit.exposure import describe_identifiers  # noqa: E402
from ghost_audit.identity import UserProfile, sha256_email  # noqa: E402
from ghost_audit.pipeline import AuditEngine  # noqa: E402
from ghost_audit.planner import decide_treatment, plan_resource  # noqa: E402
from ghost_audit.policy import Policy, PolicyRule  # noqa: E402
from ghost_audit.state import ResourceState  # noqa: E402

PASSED = []
FAILED = []

# The treatment table the engine used to hardcode. Kept here only as a regression reference: the engine
# now derives every treatment from deletion_rules.csv and must still arrive at the same table.
REFERENCE = {
    config.KIND_PROFILE: "DELETE", config.KIND_SEARCH: "DELETE", config.KIND_EXPORT: "DELETE",
    config.KIND_MARKETING: "DELETE", config.KIND_CONTACT: "DELETE", config.KIND_SUPPORT_ATTACHMENT: "DELETE",
    config.KIND_SUPPORT_MESSAGE: "REDACT", config.KIND_SUPPORT: "REDACT", config.KIND_ANALYTICS: "ANONYMIZE",
    config.KIND_DATA_FLOW: "DISABLE", config.KIND_BACKUP: "BLOCK", config.KIND_AUDIT: "RETAIN",
    config.KIND_LEGAL_HOLD: "RETAIN", config.KIND_TOMBSTONE: "REDACT",
    config.KIND_CACHE: "DELETE", config.KIND_EMAIL_DELIVERY: "REDACT", config.KIND_TELEMETRY: "ANONYMIZE",
}


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASSED if cond else FAILED).append(name)
    print(("  ok  " if cond else "  FAIL") + f" {name}" + (f": {detail}" if detail and not cond else ""))


def dataset_policy() -> Policy:
    return Policy.load(os.path.join(config.DATASET_DIR, "deletion_rules.csv"))


def edited_rules(ds, category: str, action: str, condition: str = None):
    rules = copy.deepcopy(ds.rules)
    for r in rules:
        if r.data_category == category:
            r.required_action = action
            if condition is not None:
                r.condition = condition
    return rules


def main() -> int:
    print("policy from deletion_rules.csv")
    pol = dataset_policy()
    check("dataset rules load without warnings", not pol.warnings, str(pol.warnings))
    got = {k: pol.decide(k).treatment for k in REFERENCE}
    check("derived treatments equal the former hardcoded table (plus cache, e-mail log, telemetry)",
          got == REFERENCE, str({k: (got[k], v) for k, v in REFERENCE.items() if got[k] != v}))
    check("every decision names its rule", all(pol.decide(k).rule_id for k in REFERENCE if k != config.KIND_DATA_FLOW))
    check("email delivery log follows rule_11", pol.decide(config.KIND_EMAIL_DELIVERY).rule_id == "rule_11")
    check("cache entry follows rule_04", pol.decide(config.KIND_CACHE).rule_id == "rule_04")

    print("conditions")
    audit = ResourceState("audit_record_1", "security_service", config.KIND_AUDIT, present=True,
                          retention_until="2026-01-31T00:00:00Z")
    d = pol.decide(config.KIND_AUDIT, audit, "2026-03-01T00:00:00Z")
    check("audit log past its retention period loses RETAIN (DELETE, with a note)",
          d.treatment == "DELETE" and not d.rule_id and "no longer holds" in d.note, str(d))
    d = pol.decide(config.KIND_AUDIT, audit, "2026-01-01T00:00:00Z")
    check("audit log within its retention period is RETAIN (rule_12)", d.treatment == "RETAIN" and d.rule_id == "rule_12")
    audit.retention_until = None
    check("unknown retention date keeps the rule (RETAIN)", pol.decide(config.KIND_AUDIT, audit, "2026-03-01").treatment == "RETAIN")
    hold = ResourceState("legal_hold_1", "legal_service", config.KIND_LEGAL_HOLD, present=True, hold_active=False)
    check("released legal hold loses RETAIN", pol.decide(config.KIND_LEGAL_HOLD, hold, "2026-03-01").treatment == "DELETE")
    hold.hold_active = True
    check("active legal hold is RETAIN (rule_13)", pol.decide(config.KIND_LEGAL_HOLD, hold, "2026-03-01").rule_id == "rule_13")
    hold.hold_active, hold.retention_until = None, "2026-02-01T00:00:00Z"
    check("legal hold past its retained_until date loses RETAIN",
          pol.decide(config.KIND_LEGAL_HOLD, hold, "2026-03-01T00:00:00Z").treatment == "DELETE")
    obs = observation_from_record("evidence_x", "2026-02-01T00:00:00Z", "legal_service", "retained_order_records",
                                  {"resource_id": "legal_hold_1", "subject_reference": "user_1", "status": "RELEASED"},
                                  source="test")
    check("parser reads a released hold status", obs is not None and obs.hold_active is False)
    obs = observation_from_record("evidence_x", "2026-02-01T00:00:00Z", "legal_service", "retained_order_records",
                                  {"resource_id": "legal_hold_1", "retained_until": "2033-12-31T23:59:59Z"},
                                  source="test")
    check("parser reads retained_until on legal holds", obs is not None and obs.retention_until == "2033-12-31T23:59:59Z")

    print("policy file problems are reported, never ignored")
    bad = Policy([PolicyRule("rule_a", "USER_PROFILE", "When the moon is full", "DELETE"),
                  PolicyRule("rule_b", "SEARCH_INDEX_ENTRY", "Always", "SHRED"),
                  PolicyRule("rule_c", "SHOE_SIZE", "Always", "DELETE")])
    text = " | ".join(bad.warnings)
    check("unrecognised condition warned", "rule_a" in text and "not recognised" in text)
    check("unsupported action warned", "rule_b" in text and "SHRED" in text)
    check("unknown category warned", "SHOE_SIZE" in text)
    check("categories without a rule listed", "no rule for" in text and "SECURITY_AUDIT_LOG" in text)
    missing = Policy.load(os.path.join(tempfile.gettempdir(), "does_not_exist_rules.csv"))
    check("missing rules file warned", any("not found" in w for w in missing.warnings))
    check("missing rules file: records deleted, data flows disabled",
          missing.decide(config.KIND_AUDIT).treatment == "DELETE" and missing.decide(config.KIND_DATA_FLOW).treatment == "DISABLE")

    print("retained records")
    st = ResourceState("audit_record_9", "security_service", config.KIND_AUDIT, present=False, evidence_id="evidence_1")
    p = plan_resource(None, st, decide_treatment(st, None, pol, "2026-01-01").treatment)
    check("absent retained record is not compliant and is flagged", p.retained_missing and p.compliant is False)
    check("its presence check is still emitted", p.verification is not None and p.verification.operation == "VERIFY_RECORD_PRESENT")
    st.present, st.available = True, False
    p = plan_resource(None, st, "RETAIN")
    check("unavailable retained record is flagged too", p.retained_missing)

    print("end to end on the participant dataset")
    ds = loaders.load_dataset(config.DATASET_DIR)
    base = AuditEngine(ds)
    base_res = {r.case.case_id: r for r in base.run()}
    complete_with_audit = [r for r in base_res.values() if r.status == "COMPLETE"
                           and any(p.treatment == "RETAIN" and p.state.resource_id.startswith("audit_record_") for p in r.plans)]
    check("there are COMPLETE cases with a retained audit record", bool(complete_with_audit))
    target = complete_with_audit[0]
    rid = next(p.state.resource_id for p in target.plans if p.treatment == "RETAIN")
    eng = AuditEngine(ds)
    for o in eng.obs_by_resource.get(rid, []):
        o.present = False            # simulate: the audit record was wrongly deleted
    res = {r.case.case_id: r for r in eng.run([target.case])}[target.case.case_id]
    check("a wrongly deleted audit record turns COMPLETE into INCOMPLETE", res.status == "INCOMPLETE", res.status)
    v = [s for s in res.steps if s.resource_id == rid]
    check("the case lists the presence check that must pass (empty evidence)",
          bool(v) and v[0].operation == "VERIFY_RECORD_PRESENT" and v[0].evidence_id == "")
    check("the engine warns about it", any("requires to keep" in w for w in eng.warnings()))

    ds2 = loaders.load_dataset(config.DATASET_DIR)
    ds2.rules = edited_rules(ds2, "ANALYTICS_PROFILE", "DELETE")
    eng2 = AuditEngine(ds2)
    res2 = {r.case.case_id: r for r in eng2.run()}
    anon_cases = [cid for cid, r in base_res.items() if any(s.operation == "ANONYMIZE_RECORD" for s in r.steps)]
    changed = [cid for cid in anon_cases
               if not any(s.operation == "ANONYMIZE_RECORD" and s.resource_id.startswith("analytics_") for s in res2[cid].steps)]
    check("editing the rules file (analytics -> DELETE) changes the plans", bool(anon_cases) and len(changed) == len(anon_cases),
          f"{len(changed)}/{len(anon_cases)}")
    check("... and the new plans delete the analytics records",
          any(s.operation == "DELETE_RECORD" and s.resource_id.startswith("analytics_") for cid in changed for s in res2[cid].steps))

    ds3 = loaders.load_dataset(config.DATASET_DIR)
    ds3.rules = edited_rules(ds3, "SECURITY_AUDIT_LOG", "RETAIN", "Record is within the supplied retention period")
    eng3 = AuditEngine(ds3)
    for obs_list in eng3.obs_by_resource.values():
        for o in obs_list:
            if o.kind == config.KIND_AUDIT and o.retention_until:
                o.retention_until = "2026-01-01T00:00:00Z"      # simulate: every retention period has ended
    res3 = {r.case.case_id: r for r in eng3.run()}
    flipped = [cid for cid, r in base_res.items()
               if any(s.operation == "VERIFY_RECORD_PRESENT" and s.resource_id.startswith("audit_record_") for s in r.steps)
               and any(s.resource_id.startswith("audit_record_") and s.operation in ("VERIFY_RECORD_ABSENT", "REPORT_REMAINING_RECORD")
                       for s in res3[cid].steps)]
    check("ended retention periods turn audit records into deletions", bool(flipped), "no case changed")
    check("... and the engine reports how many lost their retention basis", any("retention basis" in w for w in eng3.warnings()))

    print("exposed identifier fields")
    user = UserProfile("user_1", "Ana Lima", "ana.new@example.com", "+55 11 90000 0000", sha256_email("ana.new@example.com"))
    user.emails |= {"ana.new@example.com", "ana.old@example.com"}
    user.phones |= {"5511900000000", "5511911111111"}
    user.hashes |= {sha256_email("ana.new@example.com"), sha256_email("ana.old@example.com")}
    user.names |= {"ana lima"}
    ids = Identifiers()
    ids.emails |= {"ana.old@example.com", "someone.else@example.com"}
    ids.phones |= {"5511900000000"}
    ids.hashes |= {sha256_email("ana.old@example.com")}
    ids.user_refs |= {"user_1"}
    got = describe_identifiers(user, ids)
    check("current and historical values are told apart",
          got == ["email (historical)", "phone (current)", "email_hash (historical)"], str(got))
    check("another person's identifier is not listed", not any("else" in g for g in got))
    check("stable user_id is listed only when asked (remaining records)",
          "user_id (stable reference)" in describe_identifiers(user, ids, include_user_ref=True))

    with tempfile.TemporaryDirectory() as tmp:
        import main as entry  # noqa: E402  (defines write_output / write_exposures; running it is not needed)
        results = sorted(base_res.values(), key=lambda r: config.case_sort_key(r.case.case_id))
        exp = os.path.join(tmp, "exposures.csv")
        n = entry.write_exposures(results, exp)
        rows = list(csv.DictReader(open(exp, encoding="utf-8")))
        findings = sum(1 for r in results for s in r.steps if s.phase == "FINDING")
        check("exposures report has one row per finding", n == len(rows) == findings, f"{len(rows)} vs {findings}")
        record_rows = [r for r in rows if r["operation"] in ("REPORT_EXPOSED_PII", "REPORT_REMAINING_RECORD")]
        check("every record finding names its fields (or says the evidence does not show them)",
              bool(record_rows) and all(r["exposed_fields"] for r in record_rows))
        check("historical identifiers are reported where they still expose the person",
              any("historical" in r["exposed_fields"] for r in record_rows))
        blob = io.StringIO()
        csv.writer(blob).writerows([[v for v in r.values()] for r in rows])
        check("no e-mail address or phone number value is written", "@" not in blob.getvalue()
              and not re.search(r"\+\d{6,}|\d{9,}", blob.getvalue()))

    print("learned worker schedule")
    eng = AuditEngine(ds)
    cad = eng.cadence
    check("schedule is learned from the data", cad.source.startswith("learned") and cad.anchors >= 20, cad.source)
    check("learned first-step seconds are 41 and 52", sorted(cad.family_base.values()) == [41, 52], str(cad.family_base))
    check("learned step adds 29 or 30 s", tuple(cad.step_seconds) == (29, 30), str(cad.step_seconds))
    check("learned single-step gaps are 390-449 s and 1110-1169 s",
          sorted(cad.gaps.values()) == [(390, 449), (1110, 1169)], str(cad.gaps))
    rows = [w for w in eng.act.worker_rows if w.target_system != "account_service"]
    pairs = {(cad.family(w.details), DEFAULT_CADENCE.family(w.details)) for w in rows}
    check("learned families group the worker messages exactly like the measured rule",
          len(pairs) == 2 and len({a for a, _ in pairs}) == 2, str(pairs))
    few = learn_cadence(eng.linker.anchors([c for c in ds.cases if classify_request(c.request_text).authorises_deletion][:30],
                                           {c.case_id: eng.idx.resolve_email(c.email) for c in ds.cases})[:5],
                        eng.act.worker_rows, lambda rid: eng.linker.resource_users.get(rid, set()), parse_ts)
    check("too few pinned requests fall back to the measured defaults", few.source.startswith("default:"), few.source)
    learned_rows = [r.row() for r in AuditEngine(ds, cadence="learned").run()]
    default_rows = [r.row() for r in AuditEngine(ds, cadence="default").run()]
    check("learned and measured schedules give the same 518 answers", learned_rows == default_rows,
          f"{sum(a != b for a, b in zip(learned_rows, default_rows))} cases differ")

    print("system names")
    check("resource-type to system names are confirmed by the worker's own target systems",
          len(eng.system_map_confirmed) >= 9 and not eng.system_map_warnings, str(eng.system_map_warnings))
    saved = dict(config.PREFIX_TO_SYSTEM)
    try:
        fake = [loaders.Activity(**{**{k: "" for k in loaders.Activity.__dataclass_fields__},
                                    "activity_id": f"a{i}", "resource_id": f"account_{i}", "target_system": "identity_store",
                                    "source_system": "deletion_worker"}) for i in range(10)]
        _, warns = config.check_system_map(fake)
        check("a contradicting activity stream wins and is reported", config.PREFIX_TO_SYSTEM["account"] == "identity_store"
              and any("identity_store" in w for w in warns))
    finally:
        config.PREFIX_TO_SYSTEM.clear()
        config.PREFIX_TO_SYSTEM.update(saved)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
        return 1
    print(f"all {len(PASSED)} policy, retention, exposure, schedule and system-name checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
