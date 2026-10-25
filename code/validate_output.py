#!/usr/bin/env python3
"""Validate output.csv against the submission contract.

    python3 code/validate_output.py                 # checks <repo>/output.csv against dataset/deletion_requests.csv
    python3 code/validate_output.py --output x.csv --requests dataset/sample_deletion_requests.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from ghost_audit import config  # noqa: E402

ALLOWED = {
    "FINDING": {"REPORT_REMAINING_RECORD": "PRESENT", "REPORT_EXPOSED_PII": "PRESENT",
                "REPORT_ACTIVE_DATA_FLOW": "ACTIVE", "REPORT_RESTORE_RISK": "ACTIVE"},
    "REMEDIATION": {"DISABLE_DATA_FLOW": "DISABLED", "BLOCK_RESTORE": "BLOCKED", "DELETE_RECORD": "ABSENT",
                    "REDACT_RECORD": "REDACTED", "ANONYMIZE_RECORD": "ANONYMIZED"},
    "VERIFICATION": {"VERIFY_RECORD_ABSENT": "ABSENT", "VERIFY_RECORD_PRESENT": "PRESENT",
                     "VERIFY_RECORD_REDACTED": "REDACTED", "VERIFY_RECORD_ANONYMIZED": "ANONYMIZED",
                     "VERIFY_DATA_FLOW_DISABLED": "DISABLED", "VERIFY_RESTORE_BLOCKED": "BLOCKED"},
}
FIELDS = {"step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(REPO_ROOT, "output.csv"))
    ap.add_argument("--requests", default=os.path.join(REPO_ROOT, "dataset", "deletion_requests.csv"))
    ap.add_argument("--dataset", default=os.path.join(REPO_ROOT, "dataset"))
    args = ap.parse_args(argv)

    errors = []
    warnings = []
    with open(args.requests, encoding="utf-8-sig", newline="") as fh:
        requests = {r["case_id"]: r for r in csv.DictReader(fh)}
    with open(os.path.join(args.dataset, "evidence_files.csv"), encoding="utf-8-sig", newline="") as fh:
        evidence = {r["file_id"]: r for r in csv.DictReader(fh)}
    with open(os.path.join(args.dataset, "system_activity.csv"), encoding="utf-8-sig", newline="") as fh:
        activity = {r["activity_id"]: r for r in csv.DictReader(fh)}
    with open(os.path.join(args.dataset, "user_audit_log.csv"), encoding="utf-8-sig", newline="") as fh:
        audits = {r["audit_id"] for r in csv.DictReader(fh)}

    with open(args.output, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    if header != ["case_id", "deletion_status", "steps_json"]:
        errors.append(f"bad header: {header}")
    ids = [r[0] for r in rows]
    if len(rows) != len(requests):
        errors.append(f"expected {len(requests)} rows, found {len(rows)}")
    if set(ids) != set(requests):
        errors.append(f"case ids differ from requests: missing={sorted(set(requests) - set(ids))[:5]} extra={sorted(set(ids) - set(requests))[:5]}")
    if ids != sorted(ids, key=config.case_sort_key):
        errors.append("rows are not sorted by case_id")
    if len(set(ids)) != len(ids):
        errors.append("duplicate case ids")

    for case_id, status, steps_json in rows:
        req = requests.get(case_id)
        if status not in ("COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"):
            errors.append(f"{case_id}: bad status {status}")
            continue
        try:
            steps = json.loads(steps_json)
        except json.JSONDecodeError as exc:
            errors.append(f"{case_id}: steps_json is not valid JSON ({exc})")
            continue
        if not isinstance(steps, list):
            errors.append(f"{case_id}: steps_json is not an array")
            continue
        if status == "NOT_APPLICABLE" and steps:
            errors.append(f"{case_id}: NOT_APPLICABLE must have []")
        if status != "NOT_APPLICABLE" and not steps:
            errors.append(f"{case_id}: {status} with no steps")
        last_key = None
        for i, s in enumerate(steps, 1):
            if set(s) != FIELDS:
                errors.append(f"{case_id}: step {i} fields {sorted(s)}")
                break
            if s["step_no"] != i:
                errors.append(f"{case_id}: step_no {s['step_no']} at position {i}")
            phase, op, state = s["phase"], s["operation"], s["state"]
            if phase not in ALLOWED or op not in ALLOWED[phase]:
                errors.append(f"{case_id}: invalid phase/operation {phase}/{op}")
                continue
            if ALLOWED[phase][op] != state:
                errors.append(f"{case_id}: {op} must have state {ALLOWED[phase][op]}, got {state}")
            key = (config.PHASES.index(phase), config.OPERATION_ORDER[op], s["system_id"], s["resource_id"])
            if last_key is not None and key < last_key:
                errors.append(f"{case_id}: step {i} breaks the required ordering")
            last_key = key
            ev = s["evidence_id"]
            if status == "COMPLETE":
                if phase != "VERIFICATION":
                    errors.append(f"{case_id}: COMPLETE case has a {phase} step")
                if not ev:
                    errors.append(f"{case_id}: COMPLETE verification step {i} has no evidence")
            if status == "INCOMPLETE":
                if phase == "FINDING" and not ev:
                    errors.append(f"{case_id}: finding step {i} has no evidence")
                if phase != "FINDING" and ev:
                    errors.append(f"{case_id}: {phase} step {i} must have empty evidence")
            if ev:
                if ev in evidence:
                    if req and evidence[ev]["captured_at"] > req["audit_at"]:
                        # the file export is later than audit_at; the engine only does this when the cited
                        # record inside the file is itself timestamped before audit_at (see state.py)
                        warnings.append(f"{case_id}: cites {ev} whose export time is after audit_at (record-level timestamp used)")
                elif ev in activity:
                    if req and activity[ev]["event_time"] > req["audit_at"]:
                        errors.append(f"{case_id}: cites {ev} after audit_at")
                elif ev not in audits:
                    errors.append(f"{case_id}: unknown evidence id {ev}")
            if config.system_for_resource(s["resource_id"]) and config.system_for_resource(s["resource_id"]) != s["system_id"]:
                errors.append(f"{case_id}: {s['resource_id']} listed under {s['system_id']}")
    for w in warnings[:10]:
        print("  warning:", w)
    if len(warnings) > 10:
        print(f"  ... {len(warnings)} warnings in total")
    if errors:
        print(f"FAILED with {len(errors)} problem(s):")
        for e in errors[:50]:
            print("  -", e)
        return 1
    print(f"OK: {len(rows)} rows, header and ordering valid, all steps well-formed, evidence ids resolve and respect audit_at"
          f" ({len(warnings)} record-level citations noted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
