#!/usr/bin/env python3
"""Ghost Data Still Exists: deletion-audit agent entry point.

Usage (from the repository root or from ``code/``):

    python3 code/main.py                       # audit dataset/deletion_requests.csv -> output.csv
    python3 code/main.py --requests dataset/sample_deletion_requests.csv --output sample_predictions.csv
    python3 code/main.py --explain case_11     # print the reasoning for one case
    python3 code/main.py --use-llm             # model layer: GEMINI_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY
    python3 code/main.py --use-llm --provider gemini --llm-classify all

The engine is deterministic.  The optional model layer (see ghost_audit/llm.py)
is only used for request classification and screenshot reading when enabled;
every call is counted and written to code/evaluation/usage_run.json, from which
code/evaluation/usage_report.py renders the usage report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ghost_audit import config, loaders  # noqa: E402
from ghost_audit.pipeline import AuditEngine  # noqa: E402


def write_output(results, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["case_id", "deletion_status", "steps_json"])
        for r in results:
            writer.writerow(list(r.row()))


EXPOSURE_COLUMNS = ["case_id", "step_no", "operation", "system_id", "resource_id", "evidence_id", "exposed_fields",
                    "linked_by", "policy_rule"]


def write_exposures(results, path: str) -> int:
    """One row per finding: which identifier fields still expose the person (names only, never values)."""
    rows = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(EXPOSURE_COLUMNS)
        for r in results:
            plans = {(p.state.resource_id, p.finding.operation): p for p in r.plans if p.finding is not None}
            for i, s in enumerate(r.steps, 1):
                if s.phase != "FINDING":
                    continue
                p = plans.get((s.resource_id, s.operation))
                writer.writerow([r.case.case_id, i, s.operation, s.system_id, s.resource_id, s.evidence_id,
                                 "; ".join(p.exposed) if p else "", p.linked_by if p else "", p.rule_id if p else ""])
                rows += 1
    return rows


def explain(result) -> str:
    lines = [f"{result.case.case_id} {result.case.email} -> {result.status}"]
    if result.user:
        lines.append(f"  user={result.user.user_id} name={result.user.name} emails={sorted(result.user.emails)}")
    if result.classification:
        lines.append(f"  request: deletion={result.classification.authorises_deletion} ({result.classification.reason})")
    lines.append(f"  account={result.account_id} via {result.account_method}")
    for p in result.plans:
        st = p.state
        lines.append(f"  - {st.resource_id:18} kind={st.kind:18} present={st.present} identifiable={st.identifiable} "
                     f"enabled={st.enabled} restorable={st.restore_allowed} available={st.available} "
                     f"treatment={p.treatment} compliant={p.compliant} evidence={st.evidence_id} obs_at={st.observed_at}")
        lines.append(f"      {p.rationale}")
        if p.linked_by:
            lines.append(f"      linked by: {p.linked_by}")
        if p.exposed:
            lines.append(f"      still exposes: {', '.join(p.exposed)}")
        if p.retained_missing:
            lines.append("      RETAINED RECORD MISSING: policy requires it, but it is absent or unavailable at audit_at")
    for n in result.notes:
        lines.append(f"  note: {n}")
    for i, s in enumerate(result.steps, 1):
        lines.append(f"  {i:2} {s.phase:12} {s.operation:26} {s.state:10} {s.system_id:20} {s.resource_id:18} {s.evidence_id}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ghost Data Still Exists deletion-audit agent")
    parser.add_argument("--dataset", default=config.DATASET_DIR, help="dataset directory (default: <repo>/dataset)")
    parser.add_argument("--requests", default=None, help="requests CSV (default: <dataset>/deletion_requests.csv)")
    parser.add_argument("--output", default=os.path.join(config.REPO_ROOT, "output.csv"), help="output CSV path")
    parser.add_argument("--explain", action="append", default=[], help="case_id to explain (repeatable)")
    parser.add_argument("--exposures", default=None, metavar="CSV",
                        help="also write one row per finding naming the identifier fields that still expose the person")
    parser.add_argument("--use-llm", action="store_true", help="enable the optional model layer")
    parser.add_argument("--provider", choices=["gemini", "groq", "anthropic"], default=None,
                        help="model provider (default: GHOST_AUDIT_PROVIDER or the first provider with a key)")
    parser.add_argument("--llm-classify", choices=["ambiguous", "all"], default="ambiguous",
                        help="consult the model only for ambiguous request texts (default) or for every request")
    parser.add_argument("--llm-shadow", action="store_true",
                        help="alias for --llm-classify all: classify every request with the model and record agreement")
    parser.add_argument("--no-llm-vision", action="store_true", help="keep local OCR for screenshots even with --use-llm")
    parser.add_argument("--require-llm", action="store_true",
                        help="with --use-llm: stop with an error instead of falling back when the model layer fails")
    parser.add_argument("--doctor", action="store_true",
                        help="check Python, dataset, PDF/OCR tools, .env and API keys/models, then exit")
    parser.add_argument("--cadence", choices=["learned", "default"], default="learned",
                        help="worker schedule used to attribute unlabeled worker rows: learned from the data at run "
                             "time (default) or the values measured on the October 2026 dataset")
    parser.add_argument("--record-usage", action="store_true",
                        help="write code/evaluation/usage_run.json even when --output is not the root output.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    from ghost_audit import envfile

    # .env is read on every run (TESSERACT_CMD lives there too); real environment variables win.
    loaded = envfile.load_dotenv([config.REPO_ROOT, os.getcwd()])
    if loaded:
        print(f"[env] loaded {', '.join(loaded)} from .env (values not shown; keep .env out of code.zip)", file=sys.stderr)
    if args.doctor:
        from ghost_audit.doctor import run_doctor

        providers = [args.provider] if args.provider else ["gemini", "groq", "anthropic"]
        return run_doctor(args.dataset, providers)

    t0 = time.time()
    ds = loaders.load_dataset(args.dataset)
    if args.requests:
        ds.cases = loaders.load_cases(args.requests)
    classifier = None
    image_reader = None
    usage = None
    if args.use_llm:
        from ghost_audit import llm

        mode = "all" if args.llm_shadow else args.llm_classify
        client = llm.ModelLayer.from_env(provider=args.provider, classify_mode=mode)
        if client is None:
            msg = ("no GEMINI_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY found in the environment or in "
                   f"{os.path.join(config.REPO_ROOT, '.env')}")
            if args.require_llm:
                print(f"[llm] ERROR: {msg}", file=sys.stderr)
                return 2
            print(f"[llm] {msg}; running deterministically", file=sys.stderr)
        else:
            print(f"[llm] provider={client.provider}; checking the key and choosing models ...", file=sys.stderr)
            if not client.preflight():
                if args.require_llm:
                    print(f"[llm] ERROR: {client.disabled_reason or 'no model answered'}. "
                          "Run  python code/main.py --doctor  for details.", file=sys.stderr)
                    return 2
                print("[llm] continuing WITHOUT the model (rules and local OCR). The usage report will say so.",
                      file=sys.stderr)
            classifier = client.classify_request
            image_reader = None if args.no_llm_vision else llm.ScreenshotReader(client)
            usage = client
            client.prefetch_classifications([c.request_text for c in ds.cases])
    engine = AuditEngine(ds, verbose=args.verbose, image_reader=image_reader, classifier=classifier,
                         cadence=args.cadence)
    print(f"[schedule] {engine.cadence.source} ({engine.cadence.anchors} pinned requests); "
          f"policy: {len(engine.policy.rules)} rules from deletion_rules.csv; "
          f"system names confirmed by activity for {len(engine.system_map_confirmed)} resource types", file=sys.stderr)
    if args.verbose:
        for line in engine.cadence.describe():
            print(f"[schedule] {line}", file=sys.stderr)
        for kind, treatment in engine.policy.summary().items():
            print(f"[policy] {kind:20} {treatment}", file=sys.stderr)
    results = engine.run()
    for warning in engine.warnings():
        print(f"WARNING: {warning}", file=sys.stderr)
    if usage is not None:
        s = usage.summary()
        print(f"[llm] {s['status']}; {s['calls']} request(s), {s['failures']} failed, {s['total_tokens']:,} tokens, "
              f"text={s['text_model']}, vision={s['vision_model'] or 'local OCR'}, "
              f"agreement={s['classification_agreement']}", file=sys.stderr)
        if args.require_llm and not usage.enabled:
            print("[llm] ERROR: the model layer failed during the run (see messages above); output not written.",
                  file=sys.stderr)
            return 2
    if args.explain:
        wanted = set(args.explain)
        for r in results:
            if r.case.case_id in wanted:
                print(explain(r))
        return 0
    write_output(results, args.output)
    if args.exposures:
        n = write_exposures(results, args.exposures)
        print(f"wrote {n} finding rows to {args.exposures} (identifier field names only, no values)")
    elapsed = time.time() - t0
    statuses = {}
    for r in results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    print(f"wrote {len(results)} rows to {args.output} in {elapsed:.1f}s: {statuses}")
    usage_path = os.path.join(config.CODE_DIR, "evaluation", "usage_run.json")
    final_output = os.path.abspath(args.output) == os.path.abspath(os.path.join(config.REPO_ROOT, "output.csv"))
    if not final_output and not args.record_usage:
        # a run written elsewhere (tests, comparisons) must not replace the record of the final run
        print("(usage_run.json not updated: output is not the root output.csv; add --record-usage to force)")
        return 0
    os.makedirs(os.path.dirname(usage_path), exist_ok=True)
    from ghost_audit.evidence.images import READ_STATS

    with open(usage_path, "w", encoding="utf-8") as fh:
        json.dump({
            "requests_file": args.requests or os.path.join(args.dataset, "deletion_requests.csv"),
            "cases": len(results),
            "elapsed_seconds": round(elapsed, 2),
            "screenshots": {"total": engine.image_count, "unread": len(engine.unread_images),
                            "model": READ_STATS["model_fresh"] + READ_STATS["model_cached"],
                            "ocr": READ_STATS["ocr_fresh"] + READ_STATS["ocr_cached"],
                            "cached": READ_STATS["model_cached"] + READ_STATS["ocr_cached"]},
            "warnings": engine.warnings(),
            "schedule": {"source": engine.cadence.source, "pinned_requests": engine.cadence.anchors,
                         "families": {f: {"first_step_second": engine.cadence.family_base[f],
                                          "step_gap_seconds": list(engine.cadence.gaps[f]),
                                          "first_step_minutes": list(engine.cadence.first_offset[f][:2])}
                                      for f in sorted(engine.cadence.family_base)}},
            "policy": engine.policy.summary(),
            "model_layer": usage.summary() if usage else {"enabled": False, "calls": 0, "input_tokens": 0,
                                                            "output_tokens": 0, "total_tokens": 0,
                                                            "estimated_cost_usd": 0.0, "models": {}},
        }, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
