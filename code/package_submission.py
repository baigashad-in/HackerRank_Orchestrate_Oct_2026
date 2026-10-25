#!/usr/bin/env python3
"""Build the code zip for submission, with a secret check.

    python code/package_submission.py                 # writes code.zip, or code_vN.zip if that name is taken
    python code/package_submission.py --out code_final.zip

First rebuilds code/evaluation/evaluation_report.md with build_report.py (skip with --no-report), so the
packaged report always comes from the code. Contents: README.md, README_SOLUTION.md, .env.example and the
code/ folder. Left out: .env (and any .env.*
except .env.example), dataset/, caches (.ghost_audit_cache, __pycache__, *.pyc), virtual environments,
earlier zips and output CSVs.

Before writing, every packaged file is checked for (1) the actual values of GEMINI_API_KEY, GOOGLE_API_KEY,
GROQ_API_KEY and ANTHROPIC_API_KEY from the environment or .env, and (2) strings shaped like Google, Groq or
Anthropic keys. If anything is found the zip is not written and the file is named (the value never is).
An existing zip is never overwritten.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from ghost_audit import envfile  # noqa: E402

TOP_FILES = ["README.md", "README_SOLUTION.md", ".env.example"]
SKIP_DIRS = {"__pycache__", ".ghost_audit_cache", ".venv", "venv", "env", "node_modules", "dataset", ".git"}
SKIP_SUFFIXES = (".pyc", ".pyo", ".zip")
KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY")
KEY_SHAPES = re.compile(rb"AIza[0-9A-Za-z_\-]{30,}|gsk_[0-9A-Za-z]{30,}|sk-ant-[0-9A-Za-z_\-]{30,}")
ALLOWED_FAKES = (b"AIzaFAKE",)   # the offline tests use an obviously fake key


def files_to_package():
    out = [f for f in TOP_FILES if os.path.exists(os.path.join(REPO, f))]
    for root, dirs, files in os.walk(CODE_DIR):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(SKIP_SUFFIXES) or (name.startswith(".env") and name != ".env.example"):
                continue
            out.append(os.path.relpath(os.path.join(root, name), REPO))
    return out


def usage_record_matches() -> bool:
    """usage_run.json must describe the run in usage_report.md (same model-call total), else it is left out."""
    import json

    run = os.path.join(CODE_DIR, "evaluation", "usage_run.json")
    report = os.path.join(CODE_DIR, "evaluation", "usage_report.md")
    if not (os.path.exists(run) and os.path.exists(report)):
        return False
    try:
        calls = int(json.load(open(run, encoding="utf-8")).get("model_layer", {}).get("calls", 0))
    except (ValueError, OSError):
        return False
    m = re.search(r"\*\*Overall\*\*\s*\|[^|]*\|[^|]*\|\s*\*\*([\d,]+)\*\*", open(report, encoding="utf-8").read())
    reported = int(m.group(1).replace(",", "")) if m else 0
    return calls == reported


def comparison_report_ok() -> bool:
    """model_comparison.md is packaged only when made by a version that marks incomplete providers."""
    import json

    data_path = os.path.join(CODE_DIR, "evaluation", "model_comparison.json")
    try:
        data = json.load(open(data_path, encoding="utf-8"))
    except (OSError, ValueError):
        return False
    provs = [v for v in (data.get("providers") or {}).values() if "skipped" not in v]
    if not provs or any("complete" not in v for v in provs):
        return False
    for name, v in (data.get("providers") or {}).items():
        if "skipped" not in v and not v["complete"]:
            print(f"  note: model_comparison.md marks {name} as incomplete ({'; '.join(v.get('problems', []))})")
    return True


def secret_problems(paths):
    values = [os.environ.get(v, "").strip() for v in KEY_VARS]
    values = [v.encode("utf-8") for v in values if len(v) >= 8]
    problems = []
    for rel in paths:
        with open(os.path.join(REPO, rel), "rb") as fh:
            data = fh.read()
        if any(v in data for v in values):
            problems.append(f"{rel}: contains the value of one of {', '.join(KEY_VARS)}")
        for m in KEY_SHAPES.finditer(data):
            if not m.group(0).startswith(ALLOWED_FAKES):
                problems.append(f"{rel}: contains a string shaped like an API key")
                break
    return problems


def free_name(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}_v{n}{ext}"):
        n += 1
    return f"{stem}_v{n}{ext}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build the submission code zip with a secret check")
    ap.add_argument("--out", default=os.path.join(REPO, "code.zip"))
    ap.add_argument("--no-report", action="store_true",
                    help="do not rebuild code/evaluation/evaluation_report.md from code before packaging")
    args = ap.parse_args(argv)
    envfile.load_dotenv([REPO, os.getcwd()])
    if not args.no_report:
        # the packaged evaluation report is always the one the code produces from this checkout
        import subprocess

        print("rebuilding code/evaluation/evaluation_report.md from code (runs the engine, scorer, hold-out and "
              "test suites; about a minute) ...")
        rc = subprocess.run([sys.executable, os.path.join(CODE_DIR, "evaluation", "build_report.py")], cwd=REPO).returncode
        if rc != 0:
            print("NOT WRITTEN: building the evaluation report failed (see above)", file=sys.stderr)
            return 1
    paths = files_to_package()
    run_rel = os.path.join("code", "evaluation", "usage_run.json")
    if run_rel in paths and not usage_record_matches():
        paths.remove(run_rel)
        print("  note: usage_run.json left out: it does not match usage_report.md (it was overwritten by a later run)")
    cmp_rel = [os.path.join("code", "evaluation", n) for n in ("model_comparison.md", "model_comparison.json")]
    if any(r in paths for r in cmp_rel) and not comparison_report_ok():
        for r in cmp_rel:
            if r in paths:
                paths.remove(r)
        print("  note: model_comparison.md left out: it was made by an earlier version that could not detect a provider "
              "falling back to rules/OCR. Re-run  python code/evaluation/compare_models.py  to include a valid one.")
    problems = secret_problems(paths)
    if problems:
        print("NOT WRITTEN: possible secrets found:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    out = free_name(os.path.abspath(args.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in paths:
            z.write(os.path.join(REPO, rel), rel.replace(os.sep, "/"))
    reports = [p for p in paths if p.replace(os.sep, "/").startswith("code/evaluation/") and p.endswith(".md")]
    print(f"wrote {out}: {len(paths)} files, no secrets found")
    print("  reports included: " + ", ".join(os.path.basename(p) for p in reports))
    missing = [r for r in ("evaluation_report.md", "usage_report.md") if r not in {os.path.basename(p) for p in reports}]
    if missing:
        print("  WARNING: required report missing: " + ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
