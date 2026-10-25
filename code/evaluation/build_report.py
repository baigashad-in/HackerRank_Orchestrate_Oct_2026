#!/usr/bin/env python3
"""Build code/evaluation/evaluation_report.md from code.

    python code/evaluation/build_report.py            # compute everything and write the report
    python code/evaluation/build_report.py --check    # rebuild in memory; exit 1 if the report on disk differs

Every number in the report is computed here, at build time:
* reference and custom scores (``main.py`` scorer) with the per-case tables;
* the 2-fold hold-out (``holdout_tests.py``) for the learned and the measured worker schedule;
* the three offline test suites, run as subprocesses (pass counts are parsed from their output);
* a fresh rules-only run of the engine over ``deletion_requests.csv``, compared byte for byte with the
  submitted ``output.csv`` and checked with ``validate_output.py``; statuses, steps, findings, evidence,
  policy, retained records, worker schedule, account linking and exposed identifier fields;
* the final model-layer run, from ``usage_run.json`` (only when it matches ``usage_report.md``);
* the model comparison, from ``model_comparison.json`` (only when it was made by a version that marks
  incomplete providers).

Hand-written text comes only from ``report_template.md``. A ``{{name}}`` placeholder there that this script
does not compute stops the build, so a typed number cannot slip into the report. The output contains no
timestamps, so ``--check`` compares like with like; instead it lists fingerprints of its inputs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import difflib
import hashlib
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)

TEMPLATE = os.path.join(HERE, "report_template.md")
REPORT = os.path.join(HERE, "evaluation_report.md")
DATA = os.path.join(HERE, "report_data.json")
PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")
TEST_SUITES = ("test_engine_rules", "test_llm_adapters", "test_compare_models")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.1%})" if b else "0/0"


def num(n) -> str:
    return f"{n:,}"


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for root, dirs, files in os.walk(os.path.join(CODE_DIR, "ghost_audit")):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith(".py"):
                with open(os.path.join(root, name), "rb") as fh:
                    h.update(name.encode() + fh.read())
    with open(os.path.join(CODE_DIR, "main.py"), "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:12]


def table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def rows_csv(results) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["case_id", "deletion_status", "steps_json"])
    for r in results:
        w.writerow(list(r.row()))
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# computations
# ---------------------------------------------------------------------------
def scores(V: dict, data: dict) -> None:
    import main as scorer

    from ghost_audit import config

    gold = scorer.load(os.path.join(config.DATASET_DIR, "sample_output.csv"))
    pred, _ = scorer.run_engine(os.path.join(config.DATASET_DIR, "sample_deletion_requests.csv"), config.DATASET_DIR)
    rep = scorer.score(pred, gold)
    data["reference"] = {k: v for k, v in rep.items() if k != "per_case"}
    n = rep["cases"]
    status_ok = round(rep["case_status_accuracy"] * n)
    exact_ok = round(rep["exact_case_accuracy"] * n)
    seq_ok = round(rep["sequence_accuracy"] * n)
    confusion = ", ".join(f"{k} {v}" for k, v in rep["confusion"].items())
    V["ref_cases"] = str(n)
    V["ref_summary_table"] = table(["Metric", "Value"], [
        ["Case status accuracy", f"**{f3(rep['case_status_accuracy'])}** ({status_ok}/{n}; {confusion})"],
        ["Exact case accuracy", f"**{f3(rep['exact_case_accuracy'])}** ({exact_ok}/{n})"],
        ["Sequence accuracy", f"**{f3(rep['sequence_accuracy'])}** ({seq_ok}/{n})"],
        ["Step precision", f"**{f3(rep['step_precision'])}** ({rep['steps_pred']} predicted, {rep['steps_matched']} correct)"],
        ["Step recall", f"**{f3(rep['step_recall'])}** ({rep['steps_matched']} of {rep['steps_gold']} reference steps)"],
        ["Step F1", f"**{f3(rep['step_f1'])}**"],
        ["Field accuracy (matched steps)", ", ".join(f"{k} {f3(v)}" for k, v in rep["field_accuracy"].items())],
    ])
    V["ref_per_case_table"] = table(["case", "gold status", "predicted", "gold steps", "predicted steps", "matched", "exact"],
                                    [[c, g, p, gs, ps, m, "yes" if e else "no"] for c, g, p, gs, ps, m, e in rep["per_case"]])
    # reference steps the engine did not produce, and whether their resource ids exist in participant data
    key = ("phase", "system_id", "resource_id", "operation", "state")
    missed = []
    for cid, (_, gsteps) in gold.items():
        psteps = pred.get(cid, ("", []))[1] or []
        pk = collections.Counter(tuple(s[k] for k in key) for s in psteps)
        for s in gsteps:
            t = tuple(s[k] for k in key)
            if pk[t]:
                pk[t] -= 1
            else:
                missed.append((cid, s["resource_id"]))
    V["ref_missed_count"] = str(len(missed))
    if not missed:
        V["ref_missed_paragraph"] = "Every reference step was produced."
    else:
        found = resource_ids_in_participant_data({rid for _, rid in missed})
        listed = ", ".join(f"`{rid}` ({cid})" for cid, rid in missed)
        absent = [rid for _, rid in missed if not found.get(rid)]
        wrong = rep["steps_pred"] - rep["steps_matched"]
        V["ref_missed_paragraph"] = (
            f"The {len(missed)} missed reference steps are {listed}. "
            + (("None of these resource ids occurs" if len(absent) == len(missed) else
                f"{len(absent)} of these resource ids occur nowhere")
               + " in any parsed evidence record, activity row or text file under `dataset/` other than "
                 "`sample_output.csv`, so they cannot be recovered from evidence. " if absent else "")
            + ("Every step the engine produced was correct, including all cited evidence ids."
               if wrong == 0 and rep["field_accuracy"]["evidence_id"] == 1.0 else
               f"{wrong} produced step(s) are not in the reference."))
        data["reference_missed"] = missed

    cdir = os.path.join(HERE, "custom_cases")
    cgold = scorer.load(os.path.join(cdir, "custom_expected.csv"))
    cpred, _ = scorer.run_engine(os.path.join(cdir, "custom_requests.csv"), config.DATASET_DIR)
    crep = scorer.score(cpred, cgold)
    data["custom"] = {k: v for k, v in crep.items() if k != "per_case"}
    for cid, g, p, gs, ps, m, e in crep["per_case"]:
        V[f"custom_result_{cid}"] = "pass" if e else f"**fail** (predicted {p}, {m}/{gs} steps matched)"
    cn = crep["cases"]
    V["custom_summary_table"] = table(["Metric", "Value"], [
        ["Case status accuracy", f"{f3(crep['case_status_accuracy'])} ({round(crep['case_status_accuracy'] * cn)}/{cn})"],
        ["Exact case accuracy", f3(crep["exact_case_accuracy"])],
        ["Sequence accuracy", f3(crep["sequence_accuracy"])],
        ["Step precision / recall / F1", f"{f3(crep['step_precision'])} / {f3(crep['step_recall'])} / {f3(crep['step_f1'])} "
                                         f"({crep['steps_gold']} steps)"],
        ["Field accuracy", ", ".join(f"{k} {f3(v)}" for k, v in crep["field_accuracy"].items())],
    ])


def resource_ids_in_participant_data(rids):
    from ghost_audit import config, loaders
    from ghost_audit.pipeline import AuditEngine

    ds = loaders.load_dataset(config.DATASET_DIR)
    eng = AuditEngine(ds)
    found = {r: False for r in rids}
    for o in eng.observations:
        if o.resource_id in found:
            found[o.resource_id] = True
    for a in ds.activity:
        if a.resource_id in found or any(r in a.details for r in rids):
            for r in rids:
                if r == a.resource_id or r in a.details:
                    found[r] = True
    for root, _, files in os.walk(config.DATASET_DIR):
        for name in files:
            if name in ("sample_output.csv",) or not name.endswith((".csv", ".json", ".log", ".sql", ".txt")):
                continue
            text = open(os.path.join(root, name), encoding="utf-8", errors="ignore").read()
            for r in rids:
                if re.search(rf"\b{re.escape(r)}\b", text):
                    found[r] = True
    return found


def holdout(V: dict, data: dict) -> None:
    import holdout_tests as h

    res = {}
    for kind in ("learned", "default"):
        res[kind] = {"accounts": h.account_test(kind), "rows": {s: h.chain_test(kind, seed=s) for s in (7, 11, 23)}}
    h.set_cadence(h.DEFAULT_CADENCE)
    data["holdout"] = res
    L, D = res["learned"], res["default"]
    V["holdout_pinned"] = str(L["accounts"]["cases"])
    rows = []
    for label, r in (("learned at run time (used for `output.csv`)", L), ("measured defaults (`--cadence default`)", D)):
        a, c = r["accounts"], r["rows"][7]
        rows.append([label, pct(a["recovered"], a["cases"]), a["wrong"], a["unassigned"],
                     f"{c['recovered']}/{c['hidden_rows']}", c["missed"], c["wrong"]])
    V["holdout_table"] = table(["Schedule", "Accounts recovered", "Wrong account", "Unassigned",
                                "Hidden worker rows recovered (seed 7)", "Missed (no step)", "Wrongly attributed"], rows)
    parts = []
    for s in (11, 23):
        lc, dc = L["rows"][s], D["rows"][s]
        parts.append(f"seed {s}: learned {lc['recovered']}/{lc['hidden_rows']} recovered, {lc['wrong']} wrong; "
                     f"defaults {dc['recovered']}/{dc['hidden_rows']}, {dc['wrong']} wrong")
    V["holdout_seeds_sentence"] = ("Worker rows are hidden at random with a fixed seed. Other seeds: " + "; ".join(parts) + ".")
    lw = sum(L["rows"][s]["wrong"] for s in (7, 11, 23))
    dw = sum(D["rows"][s]["wrong"] for s in (7, 11, 23))
    lr = sum(L["rows"][s]["recovered"] for s in (7, 11, 23))
    dr = sum(D["rows"][s]["recovered"] for s in (7, 11, 23))
    V["holdout_compare_sentence"] = (
        f"over three seeds the learned schedule recovered {lr} hidden worker rows with {lw} wrong attributions, "
        f"against {dr} with {dw} for the measured defaults; accounts {pct(L['accounts']['recovered'], L['accounts']['cases'])} "
        f"against {pct(D['accounts']['recovered'], D['accounts']['cases'])}.")
    V["holdout_learned_account_misses"] = str(L["accounts"]["wrong"] + L["accounts"]["unassigned"])


def test_suites(V: dict, data: dict) -> None:
    data["tests"] = {}
    for name in TEST_SUITES:
        proc = subprocess.run([sys.executable, os.path.join(HERE, f"{name}.py")], capture_output=True, text=True,
                              cwd=REPO, timeout=1800)
        lines = proc.stdout.splitlines()
        ok = sum(1 for ln in lines if re.match(r"^\s*ok\b", ln))
        fail = [ln.strip() for ln in lines if re.match(r"^\s*FAIL\b", ln)]
        data["tests"][name] = {"passed": ok, "failed": len(fail), "exit_code": proc.returncode}
        if proc.returncode == 0 and not fail:
            V[f"tests_{name[5:]}"] = f"{ok}/{ok} checks passed"
        else:
            V[f"tests_{name[5:]}"] = (f"**{len(fail)} of {ok + len(fail)} checks FAILED** (exit code {proc.returncode}): "
                                      + "; ".join(fail[:5]))


def full_run(V: dict, data: dict) -> dict:
    from ghost_audit import config, loaders
    from ghost_audit.pipeline import AuditEngine
    from ghost_audit.request_classifier import classify_request

    ds = loaders.load_dataset(config.DATASET_DIR)
    eng = AuditEngine(ds)
    results = eng.run()
    built = rows_csv(results)
    out_path = os.path.join(REPO, "output.csv")
    submitted = open(out_path, "rb").read() if os.path.exists(out_path) else b""
    same = built == submitted
    proc = subprocess.run([sys.executable, os.path.join(CODE_DIR, "validate_output.py")], capture_output=True,
                          text=True, cwd=REPO)
    validator = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
    status = collections.Counter(r.status for r in results)
    steps = [s for r in results for s in r.steps]
    findings = collections.Counter(s.operation for s in steps if s.phase == "FINDING")
    cited = {s.evidence_id for s in steps if s.evidence_id}
    cited_files = {e for e in cited if e in ds.evidence_by_id}
    files_with_obs = {o.file_id for o in eng.observations} | {s.file_id for s in eng.scopes}
    deletion = [r for r in results if r.status != "NOT_APPLICABLE"]
    methods = collections.Counter()
    for r in deletion:
        m = r.account_method or ""
        methods["identifier" if m == "identifier" else "worker-run attribution" if m.startswith("run-assignment")
                else "account seen only in evidence" if m.startswith("identifier-no-activity")
                else "no account in the activity stream"] += 1
    retained = [p for r in deletion for p in r.plans if p.treatment == "RETAIN"]
    retained_kinds = collections.Counter(p.state.kind for p in retained)
    missing = sum(1 for p in retained if p.retained_missing)
    dates = sorted(p.state.retention_until[:4] for p in retained if p.state.retention_until)
    cad = eng.cadence
    fams = sorted(cad.family_base, key=lambda f: cad.gaps[f][0])
    sched = "; ".join(f"one family starts at second {cad.family_base[f]} and steps every {cad.gaps[f][0]} or "
                      f"{cad.gaps[f][1]} s" for f in fams)
    eng_default = AuditEngine(ds, cadence="default")
    same_schedule = [r.row() for r in eng_default.run()] == [r.row() for r in results]

    na = [r for r in results if r.status == "NOT_APPLICABLE"]
    flagged_accounts = set(eng.act.non_deletion_accounts)
    flagged_users = {u for a in flagged_accounts for u in eng.linker.resource_users.get(a, set())}
    na_flagged = sum(1 for r in na if r.user and r.user.user_id in flagged_users)
    rules_unsure = [r.case.case_id for r in results if classify_request(r.case.request_text).confidence < 0.95]
    V["intake_sentence"] = (f"{na_flagged} of the {len(na)} `NOT_APPLICABLE` cases belong to people whose account the "
                            f"privacy intake itself flagged as non-deletion (`CLASSIFY_REQUEST`, {len(flagged_accounts)} "
                            f"accounts flagged).")
    V["background_rows_sentence"] = (
        f"{sum(1 for a in ds.activity if a.resource_id.startswith('email_delivery_'))} e-mail delivery and "
        f"{sum(1 for a in ds.activity if a.resource_id.startswith('cache_'))} cache activity rows exist; "
        f"{sum(1 for r in results for p in r.plans if p.state.kind in ('EMAIL_DELIVERY_LOG', 'CACHE_ENTRY'))} such "
        f"records are linked to any request, so no answer depends on them.")
    V["policy_rule_count"] = str(len(eng.policy.rules))
    V["screenshot_count"] = str(eng.image_count)
    V["system_map_sentence"] = (f"confirmed against the deletion worker's own target systems for "
                                f"{len(eng.system_map_confirmed)} resource types"
                                + (f" ({len(eng.system_map_warnings)} disagreements reported)" if eng.system_map_warnings else "")
                                + f"; the other {len(config.PREFIX_TO_SYSTEM) - len(eng.system_map_confirmed)} come from the "
                                  f"table in `config.py`.")
    V["retention_sentence"] = (f"In this dataset the retention dates of retained records lie in {dates[0]} to {dates[-1]}"
                               f" and {len(eng.policy_fallbacks)} record(s) lost their retention basis."
                               if dates else "No retention dates were found on retained records.")
    V["full_run_table"] = table(["Item", "Value"], [
        ["submitted `output.csv`", ("byte-identical to this build's rules-only run" if same else
                                    "**differs from this build's rules-only run**") + f"; `validate_output.py`: {validator}"],
        ["cases", f"{len(results)}"],
        ["statuses", ", ".join(f"{v} {k}" for k, v in status.most_common())],
        ["non-deletion requests", f"{len(na)} `NOT_APPLICABLE`; {na_flagged} of them belong to people whose account the "
                                  f"intake flagged as non-deletion; texts the rules were unsure about: "
                                  f"{', '.join(rules_unsure) or 'none'}"],
        ["steps", f"{num(len(steps))} (mean {len(steps) / max(1, len(results)):.1f} per case); findings: "
                  + ", ".join(f"{findings[k]} {lbl}" for k, lbl in (("REPORT_REMAINING_RECORD", "remaining records"),
                                                                  ("REPORT_EXPOSED_PII", "exposed identifiers"),
                                                                  ("REPORT_ACTIVE_DATA_FLOW", "active data flows"),
                                                                  ("REPORT_RESTORE_RISK", "restore risks")))],
        ["evidence", f"{num(len(eng.observations))} observations from {len(files_with_obs)} of {len(ds.evidence_files)} "
                     f"files; {len(cited_files)} files cited"
                     + (f", plus {len(cited) - len(cited_files)} activity or audit ids" if len(cited) > len(cited_files) else "")],
        ["policy", f"{len(eng.policy.rules)} rules read from `deletion_rules.csv`, {len(eng.policy.warnings)} warnings; "
                   f"{len(retained)} retained records ({', '.join(f'{v} {k}' for k, v in retained_kinds.most_common())}), "
                   f"{missing} missing; {len(eng.policy_fallbacks)} lost their retention basis"],
        ["worker schedule", f"{cad.source} ({cad.anchors} pinned requests): {sched}; "
                            + ("same answers as the measured defaults" if same_schedule else
                               "**answers differ from the measured defaults**")],
        [f"account linking ({len(deletion)} deletion cases)", ", ".join(f"{v} {k}" for k, v in methods.most_common())],
    ])
    data["full_run"] = {"same_as_submitted": same, "validator": validator, "statuses": dict(status),
                        "steps": len(steps), "findings": dict(findings), "observations": len(eng.observations),
                        "files_with_observations": len(files_with_obs), "files_cited": len(cited_files),
                        "account_linking": dict(methods), "retained": dict(retained_kinds), "retained_missing": missing,
                        "schedule_same_as_default": same_schedule, "rules_unsure": rules_unsure}
    exposures(V, data, results)
    return {"results": results, "same": same, "steps": len(steps), "cases": len(results)}


def exposures(V: dict, data: dict, results) -> None:
    counts = {"REPORT_EXPOSED_PII": collections.Counter(), "REPORT_REMAINING_RECORD": collections.Counter()}
    n = collections.Counter()
    lookup = collections.Counter()
    hist = collections.Counter()
    special = collections.Counter()
    for r in results:
        plans = {(p.state.resource_id, p.finding.operation): p for p in r.plans if p.finding is not None}
        for s in r.steps:
            if s.phase != "FINDING" or s.operation not in counts:
                continue
            p = plans.get((s.resource_id, s.operation))
            fields = p.exposed if p else []
            n[s.operation] += 1
            if any("historical" in f for f in fields):
                hist[s.operation] += 1
            for f in fields:
                base = f.split(" [")[0]
                if "[lookup" in f:
                    lookup[base] += 1
                    continue
                if "not in the identity records" in f:
                    special["unlisted " + base.split(" ")[0]] += 1
                    continue
                if "not shown" in f or "not stated" in f:
                    special["fields not shown"] += 1
                    continue
                counts[s.operation][base] += 1
    label = {"REPORT_EXPOSED_PII": "exposed PII (record kept, identifiers must go)",
             "REPORT_REMAINING_RECORD": "remaining record (must be deleted)"}
    V["exposure_table"] = table(["Finding", "Records", "Fields still exposed (count of findings)"],
                                [[label[op], n[op], ", ".join(f"{k} {v}" for k, v in c.most_common()) or "-"]
                                 for op, c in counts.items()])
    notes = []
    if lookup:
        notes.append(f"{sum(lookup.values())} remaining-record findings are known only by the key a search returned them "
                     f"for: " + ", ".join(f"{k} {v}" for k, v in lookup.most_common()) + ".")
    for k, v in sorted(special.items()):
        if k.startswith("unlisted"):
            field = k.split(" ", 1)[1]
            article = "an" if field[:1] in "aeiou" else "a"
            notes.append(f"{v} finding(s) carry {article} {field} value that is in no identity record; a data-catalog "
                         f"link ties the record to the person.")
        else:
            notes.append(f"{v} finding(s) do not show their identifier fields in the evidence; the report says so "
                         f"instead of guessing.")
    notes.append(f"Historical identifiers still expose the person in {hist['REPORT_REMAINING_RECORD']} remaining-record "
                 f"and {hist['REPORT_EXPOSED_PII']} exposed-PII findings.")
    V["exposure_notes"] = "\n".join(f"- {x}" for x in notes)
    data["exposures"] = {"findings": dict(n), "fields": {k: dict(v) for k, v in counts.items()},
                         "lookup_only": dict(lookup), "special": dict(special), "historical": dict(hist)}


def final_run(V: dict, data: dict, inputs: list) -> None:
    import package_submission as pkg

    path = os.path.join(HERE, "usage_run.json")
    run = None
    if os.path.exists(path) and pkg.usage_record_matches():
        run = json.load(open(path, encoding="utf-8"))
    ml = (run or {}).get("model_layer") or {}
    if not run or not ml.get("calls"):
        reason = ("`usage_run.json` is missing or does not match `usage_report.md`" if not run else
                  "the recorded final run used no model calls")
        V["final_run_sentence"] = ("The final full-dataset run is described in `usage_report.md`; no model-layer run "
                                   f"is recorded in this checkout ({reason}).")
        V["final_run_block"] = (f"**Final run with the model layer:** not recorded in this checkout ({reason}). "
                                "`usage_report.md` describes the final run.")
        V["final_run_elapsed_sentence"] = ""
        data["final_run"] = None
        return
    inputs.append(f"`usage_run.json` {sha(path)}")
    models = ml.get("models", {})
    text_m, vis_m = ml.get("text_model"), ml.get("vision_model")
    agree = ml.get("classification_agreement", {})
    shots = run.get("screenshots", {})
    errors = ml.get("error_counts", {})
    rows = []
    for m, u in models.items():
        role = "request classification" if m == text_m else "screenshot transcription" if m == vis_m else "other"
        rows.append([ml.get("provider", ""), f"`{m}`", role, u["calls"], u["failures"], num(u["input_tokens"]),
                     num(u["output_tokens"]), f"{u['estimated_cost_usd']:.4f}"])
    rows.append(["**overall**", "", "", ml["calls"], ml.get("failures", 0), num(ml["input_tokens"]),
                 num(ml["output_tokens"]), f"{ml['estimated_cost_usd']:.4f}"])
    V["final_run_sentence"] = (f"The final full-dataset run added the {ml.get('provider')} model layer "
                               f"(`{text_m}` for request classification, `{vis_m or 'local OCR'}` for screenshots) and "
                               f"produced the answers described in section 5.")
    V["final_run_block"] = "\n".join([
        f"**Final run with the model layer** (from `usage_run.json`; command "
        f"`python code/main.py --use-llm --llm-classify {ml.get('classify_mode', 'all')} --require-llm`; "
        f"{run.get('cases')} cases, {run.get('elapsed_seconds')} s):", "",
        table(["Provider", "Model", "Role", "Calls", "Failed calls", "Input tokens", "Output tokens", "Est. cost (USD)"], rows),
        "",
        f"- status {ml.get('status')}; screenshots read by the vision model: {shots.get('model', 0)} of "
        f"{shots.get('total', 0)}, by local OCR: {shots.get('ocr', 0)}, unread: {shots.get('unread', 0)}",
        f"- request classification, rules vs model: agree {agree.get('agree', 0)}, disagree {agree.get('disagree', 0)}, "
        f"decided by the model because the rules were unsure {agree.get('model_only', 0)}, "
        f"not classified {agree.get('unclassified', 0)}",
        "- provider errors: " + (", ".join(f"{k} {v}" for k, v in errors.items()) or "none")
        + (" (each was retried; see `usage_report.md` section 7)" if errors else ""),
    ])
    V["final_run_elapsed_sentence"] = f"The final model-layer run took {run.get('elapsed_seconds')} s."
    data["final_run"] = {"provider": ml.get("provider"), "calls": ml["calls"], "tokens": ml["total_tokens"],
                         "cost": ml["estimated_cost_usd"], "agreement": agree, "screenshots": shots}


def comparison(V: dict, data: dict, inputs: list, full: dict) -> None:
    path = os.path.join(HERE, "model_comparison.json")
    try:
        cmp = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        cmp = None
    provs = {k: v for k, v in ((cmp or {}).get("providers") or {}).items() if "skipped" not in v}
    if not provs or any("complete" not in v for v in provs.values()):
        V["comparison_block"] = ("Not available in this checkout: `model_comparison.json` is missing or was made by an "
                                 "earlier version that could not detect a provider falling back to rules or OCR. Run "
                                 "`python code/evaluation/compare_models.py`.")
        V["model_layer_effect_sentence"] = (
            "Whether the model layer changes answers is measured by `compare_models.py`; no valid comparison is recorded "
            "in this checkout, so no claim is made here.")
        data["comparison"] = None
        return
    inputs.append(f"`model_comparison.json` {sha(path)}")
    names = sorted(provs)
    texts = cmp.get("distinct_request_texts", 0)
    shots_total = full.get("screenshots", 11)

    def cell(p, f):
        v = provs[p]
        if f == "coverage":
            return "complete" if v["complete"] else "**incomplete**: " + "; ".join(v.get("problems", []))
        if f == "text":
            return f"`{v.get('text_model')}`"
        if f == "vision":
            src = list((v.get("screenshot_source") or {}).values())
            fresh, cached, ocr = src.count("fresh"), src.count("cached"), src.count("ocr")
            model = f"`{v['vision_model']}`" if v.get("vision_model") else f"{p}-vision cache"
            return f"{model} ({fresh} fresh, {cached} cached, {ocr} OCR fallback)"
        if f == "class":
            c = v.get("classification", {})
            return f"{c.get('classified', 0)}/{texts}, {c.get('agree_with_rules', 0)} agree with the rules"
        if f == "shots":
            return f"{v.get('screenshots_same_as_ocr', 0)}/{v.get('screenshots_compared', 0)} compared"
        if f == "answers":
            a = v.get("answers", {})
            return (f"{full['cases'] - len(a.get('cases_differing', []))}/{full['cases']} cases, "
                    f"{a.get('steps_matched', 0)}/{a.get('steps', 0)} steps")
        if f == "calls":
            return f"{v.get('calls', 0)} ({v.get('failures', 0)})"
        if f == "tokens":
            return f"{num(v.get('input_tokens', 0))} / {num(v.get('output_tokens', 0))}"
        if f == "cost":
            return f"USD {v.get('estimated_cost_usd', 0):.3f}"
        if f == "time":
            return f"{v.get('seconds', 0)} s"
        return ""

    fields = [("coverage", "coverage"), ("text", "text model"), ("vision", "vision model (screenshot sources)"),
              ("class", "request texts classified"), ("shots", "screenshots: same facts as local OCR"),
              ("answers", "answers identical to the rules-only run"), ("calls", "calls (failed: rate limits waited out)"),
              ("tokens", "tokens in / out (incl. reasoning)"), ("cost", "est. cost at paid list prices"),
              ("time", "wall time")]
    rows = [[label] + [cell(p, f) for p in names] for f, label in fields]
    dis = cmp.get("classification_disagreements", [])
    lines = [f"Result (from `model_comparison.json`, written by `compare_models.py` on "
             f"{cmp.get('cases')} requests, {texts} distinct request texts and {shots_total} screenshots):", "",
             table([""] + names, rows), ""]
    complete = [p for p in names if provs[p]["complete"]]
    if len(complete) >= 2:
        lines.append(f"- Texts on which any provider disagrees with the rules or with another provider: {len(dis)}.")
    for p in names:
        limits = provs[p].get("tokens_per_minute_limits") or {}
        if limits:
            lines.append(f"- {p} reported tokens-per-minute limits ("
                         + ", ".join(f"`{m}` {num(n)}" for m, n in limits.items())
                         + f"); later requests were paced under them ({provs[p].get('rate_limit_waits', 0)} pacing waits).")
    incomplete = [p for p in names if not provs[p]["complete"]]
    if incomplete:
        lines.append("- Not counted as evidence about the models, because they did not complete: " + ", ".join(incomplete) + ".")
    lines.append("- This comparison was made by the version that marks incomplete providers (after failure 23).")
    changed = [p for p in complete if provs[p].get("answers", {}).get("cases_differing")]
    if complete and not changed:
        V["model_layer_effect_sentence"] = (
            f"In the comparison no completed provider ({', '.join(complete)}) changed any answer, so the model layer's "
            "value here is corroboration and robustness to OCR failure, not extra accuracy, and it makes a run slower.")
    elif changed:
        V["model_layer_effect_sentence"] = (
            f"In the comparison {', '.join(changed)} changed some answers relative to the rules-only run (see "
            "`model_comparison.md`); the submitted answers come from the run described in `usage_report.md`.")
    else:
        V["model_layer_effect_sentence"] = ("No provider completed the comparison, so no claim is made about whether the "
                                            "model layer changes answers.")
    V["comparison_block"] = "\n".join(lines)
    data["comparison"] = {p: {k: provs[p].get(k) for k in ("complete", "calls", "failures", "seconds")} for p in names}


# ---------------------------------------------------------------------------
def build() -> tuple[str, dict]:
    from ghost_audit import config

    V: dict = {}
    data: dict = {}
    inputs = [f"engine code {code_fingerprint()}", f"`report_template.md` {sha(TEMPLATE)}",
              f"`deletion_requests.csv` {sha(os.path.join(config.DATASET_DIR, 'deletion_requests.csv'))}",
              f"`deletion_rules.csv` {sha(os.path.join(config.DATASET_DIR, 'deletion_rules.csv'))}"]
    out_path = os.path.join(REPO, "output.csv")
    if os.path.exists(out_path):
        inputs.append(f"`output.csv` {sha(out_path)}")
    print("[report] scoring reference and custom cases ...", file=sys.stderr)
    scores(V, data)
    print("[report] full-dataset run ...", file=sys.stderr)
    full = full_run(V, data)
    full["screenshots"] = int(V["screenshot_count"])
    print("[report] hold-out tests ...", file=sys.stderr)
    holdout(V, data)
    print("[report] offline test suites ...", file=sys.stderr)
    test_suites(V, data)
    final_run(V, data, inputs)
    comparison(V, data, inputs, full)
    V["inputs"] = "; ".join(inputs) + " (sha256, first 12 hex digits)."
    template = open(TEMPLATE, encoding="utf-8").read()
    missing = sorted(set(PLACEHOLDER.findall(template)) - set(V))
    if missing:
        raise SystemExit("report_template.md uses values the code does not compute: " + ", ".join(missing))
    text = PLACEHOLDER.sub(lambda m: V[m.group(1)], template)
    return text, data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build the evaluation report from code")
    ap.add_argument("--check", action="store_true", help="exit 1 if evaluation_report.md differs from a fresh build")
    args = ap.parse_args(argv)
    text, data = build()
    if args.check:
        current = open(REPORT, encoding="utf-8").read() if os.path.exists(REPORT) else ""
        if current == text:
            print("evaluation_report.md is up to date with the code")
            return 0
        diff = list(difflib.unified_diff(current.splitlines(), text.splitlines(), "on disk", "fresh build", lineterm="", n=0))
        print("evaluation_report.md differs from a fresh build (stale or edited by hand):")
        print("\n".join(diff[:40]))
        return 1
    if os.path.exists(REPORT):
        previous = open(REPORT, encoding="utf-8").read()
        if previous != text:
            keep = os.path.join(REPO, ".ghost_audit_cache", "reports")
            os.makedirs(keep, exist_ok=True)
            name = f"evaluation_report_{hashlib.sha256(previous.encode()).hexdigest()[:8]}.md"
            with open(os.path.join(keep, name), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(previous)
            print(f"[report] previous report kept as .ghost_audit_cache/reports/{name}", file=sys.stderr)
    with open(REPORT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    with open(DATA, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"wrote {REPORT} (and {os.path.basename(DATA)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
