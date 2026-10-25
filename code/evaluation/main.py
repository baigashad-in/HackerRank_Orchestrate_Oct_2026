#!/usr/bin/env python3
"""Evaluation harness: score predictions against reference answers.

    python3 code/evaluation/main.py                      # sample cases, prints the report
    python3 code/evaluation/main.py --pred preds.csv --gold dataset/sample_output.csv
    python3 code/evaluation/main.py --custom              # also run the hand-made evaluation cases

Metrics
-------
* case accuracy        share of cases whose deletion_status matches
* exact-case accuracy  share of cases whose entire step list matches
* step precision/recall/F1  steps matched on (phase, system_id, resource_id, operation, state)
* field accuracy       per-field agreement on matched steps (evidence_id included)
* sequence accuracy    share of cases whose predicted step order equals the gold order
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

STEP_KEY = ("phase", "system_id", "resource_id", "operation", "state")


def load(path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = {}
        for r in csv.DictReader(fh):
            try:
                steps = json.loads(r["steps_json"] or "[]")
            except json.JSONDecodeError:
                steps = None
            rows[r["case_id"]] = (r["deletion_status"], steps)
    return rows


def score(pred, gold, verbose=False):
    cases = sorted(gold, key=lambda c: int(c.split("_")[1]) if c.split("_")[1].isdigit() else c)
    n = len(cases)
    status_ok = exact_ok = seq_ok = 0
    tp = fp = fn = 0
    field_hits = Counter()
    field_total = 0
    per_case = []
    confusion = Counter()
    for cid in cases:
        gstatus, gsteps = gold[cid]
        pstatus, psteps = pred.get(cid, ("<missing>", []))
        psteps = psteps or []
        confusion[(gstatus, pstatus)] += 1
        s_ok = gstatus == pstatus
        status_ok += s_ok
        gkeys = [tuple(s[k] for k in STEP_KEY) for s in gsteps]
        pkeys = [tuple(s[k] for k in STEP_KEY) for s in psteps]
        gset, pset = Counter(gkeys), Counter(pkeys)
        matched = sum((gset & pset).values())
        tp += matched
        fp += sum((pset - gset).values())
        fn += sum((gset - pset).values())
        gmap = {k: s for k, s in zip(gkeys, gsteps)}
        pmap = {k: s for k, s in zip(pkeys, psteps)}
        for k in gset & pset:
            field_total += 1
            for f in ("phase", "system_id", "resource_id", "operation", "state", "evidence_id"):
                field_hits[f] += gmap[k].get(f) == pmap[k].get(f)
        exact = s_ok and gkeys == pkeys and all(gmap[k].get("evidence_id") == pmap[k].get("evidence_id") for k in gset)
        exact_ok += exact
        common = [k for k in pkeys if k in gset]
        gcommon = [k for k in gkeys if k in pset]
        seq = (common == gcommon) and (len(gkeys) == len(pkeys))
        seq_ok += seq
        per_case.append((cid, gstatus, pstatus, len(gsteps), len(psteps), matched, exact))
        if verbose and not exact:
            print(f"--- {cid}: gold={gstatus} pred={pstatus}")
            for k in gkeys:
                flag = "ok " if k in pset else "MISSING"
                ev = gmap[k].get("evidence_id")
                pev = pmap.get(k, {}).get("evidence_id")
                evflag = "" if ev == pev else f"  evidence gold={ev} pred={pev}"
                print(f"    {flag} {k}{evflag}")
            for k in pkeys:
                if k not in gset:
                    print(f"    EXTRA   {k} evidence={pmap[k].get('evidence_id')}")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    report = {
        "cases": n,
        "case_status_accuracy": status_ok / n if n else 0.0,
        "exact_case_accuracy": exact_ok / n if n else 0.0,
        "sequence_accuracy": seq_ok / n if n else 0.0,
        "step_precision": prec, "step_recall": rec, "step_f1": f1,
        "steps_gold": tp + fn, "steps_pred": tp + fp, "steps_matched": tp,
        "field_accuracy": {f: (field_hits[f] / field_total if field_total else 0.0)
                           for f in ("phase", "system_id", "resource_id", "operation", "state", "evidence_id")},
        "confusion": {f"{g}->{p}": c for (g, p), c in sorted(confusion.items())},
        "per_case": per_case,
    }
    return report


def format_report(rep) -> str:
    out = [
        f"cases: {rep['cases']}",
        f"case status accuracy : {rep['case_status_accuracy']:.3f}",
        f"exact case accuracy  : {rep['exact_case_accuracy']:.3f}",
        f"sequence accuracy    : {rep['sequence_accuracy']:.3f}",
        f"step precision/recall/F1: {rep['step_precision']:.3f} / {rep['step_recall']:.3f} / {rep['step_f1']:.3f}"
        f"  (gold {rep['steps_gold']}, pred {rep['steps_pred']}, matched {rep['steps_matched']})",
        "field accuracy on matched steps: " + ", ".join(f"{k}={v:.3f}" for k, v in rep["field_accuracy"].items()),
        "status confusion: " + ", ".join(f"{k}:{v}" for k, v in rep["confusion"].items()),
    ]
    return "\n".join(out)


def run_engine(requests_path: str, dataset_dir: str):
    from ghost_audit import loaders
    from ghost_audit.pipeline import AuditEngine

    ds = loaders.load_dataset(dataset_dir)
    ds.cases = loaders.load_cases(requests_path)
    engine = AuditEngine(ds)
    results = engine.run()
    return {r.case.case_id: (r.status, json.loads(r.steps_json())) for r in results}, results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(REPO_ROOT, "dataset"))
    parser.add_argument("--requests", default=None, help="requests CSV to run the engine on (default: sample requests)")
    parser.add_argument("--gold", default=None, help="gold answers CSV (default: dataset/sample_output.csv)")
    parser.add_argument("--pred", default=None, help="score an existing predictions CSV instead of running the engine")
    parser.add_argument("--custom", action="store_true", help="also score the hand-made cases in code/evaluation/custom_cases")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", default=None, help="write the metrics as JSON to this path")
    args = parser.parse_args(argv)

    requests_path = args.requests or os.path.join(args.dataset, "sample_deletion_requests.csv")
    gold_path = args.gold or os.path.join(args.dataset, "sample_output.csv")
    gold = load(gold_path)
    if args.pred:
        pred = load(args.pred)
    else:
        pred, _ = run_engine(requests_path, args.dataset)
    rep = score(pred, gold, verbose=args.verbose)
    print("== sample / reference cases ==")
    print(format_report(rep))
    reports = {"reference": rep}
    if args.custom:
        custom_dir = os.path.join(HERE, "custom_cases")
        creq = os.path.join(custom_dir, "custom_requests.csv")
        cgold = os.path.join(custom_dir, "custom_expected.csv")
        if os.path.exists(creq) and os.path.exists(cgold):
            cpred, _ = run_engine(creq, args.dataset)
            crep = score(cpred, load(cgold), verbose=args.verbose)
            print("\n== custom cases ==")
            print(format_report(crep))
            reports["custom"] = crep
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(reports, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
