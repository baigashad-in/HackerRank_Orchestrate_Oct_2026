"""``python code/main.py --doctor``: check this machine before a real run.

Prints what works and what does not, with the fix for each problem.  Secrets
are never printed (only key names and lengths).  Talking to a model provider
costs at most two small requests per provider (one text, one screenshot).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import List

from . import config, loaders
from .llm import PROVIDER_KEY_ENV, ModelLayer, SCREENSHOT_PROMPT


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _ids_in(text: str) -> int:
    return len(set(config.RESOURCE_ID_RE.findall(text or ""))) + len(set(config.EMAIL_RE.findall(text or "")))


def run_doctor(dataset_dir: str, providers: List[str], vision_probe: bool = True) -> int:
    problems: List[str] = []
    print("Ghost Data Still Exists: environment check\n")

    print("Python")
    ver = sys.version_info
    (_ok if ver >= (3, 10) else _bad)(f"Python {platform.python_version()} on {platform.system()} {platform.release()}")
    if ver < (3, 10):
        problems.append("install Python 3.10 or newer")

    print("\nDataset")
    try:
        ds = loaders.load_dataset(dataset_dir)
        missing = [e.file_path for e in ds.evidence_files if not os.path.exists(e.abs_path)]
        _ok(f"{len(ds.cases)} requests, {len(ds.identities)} identities, {len(ds.activity)} activity rows, "
            f"{len(ds.evidence_files) - len(missing)}/{len(ds.evidence_files)} evidence files present in {dataset_dir}")
        if missing:
            _bad(f"missing evidence files, e.g. {missing[:3]}")
            problems.append("unzip the full repository archive (dataset/evidence must be complete)")
    except Exception as exc:
        _bad(f"cannot load the dataset from {dataset_dir}: {exc}")
        problems.append("run from the repository root, next to AGENTS.md and the dataset/ folder")
        ds = None

    print("\n.env and keys (values are never shown)")
    env_path = os.path.join(config.REPO_ROOT, ".env")
    if os.path.isfile(env_path):
        _ok(f".env found at {env_path}")
    else:
        _warn(f"no .env at {env_path} (fine if the keys are set in the environment)")
        if os.path.isfile(env_path + ".txt"):
            _bad("found .env.txt: Notepad added .txt; rename it to .env")
            problems.append("rename .env.txt to .env")
    for prov, names in PROVIDER_KEY_ENV.items():
        for n in names:
            v = os.environ.get(n, "").strip()
            if v:
                _ok(f"{n} set ({len(v)} characters)")
    tc = os.environ.get("TESSERACT_CMD", "").strip()
    if tc:
        (_ok if os.path.isfile(tc.strip('"')) else _bad)(f"TESSERACT_CMD={tc}")
        if not os.path.isfile(tc.strip('"')):
            problems.append("TESSERACT_CMD points to a file that does not exist")

    print("\nPDF text (4 documents)")
    if ds is not None:
        from .evidence.pdfs import extract_pdf_text
        pdf = next((e for e in ds.evidence_files if e.ext == ".pdf"), None)
        if pdf is not None:
            txt = extract_pdf_text(pdf.abs_path)
            if len(txt.strip()) > 50:
                _ok(f"{os.path.basename(pdf.file_path)}: {len(txt)} characters extracted")
            else:
                _bad("no PDF text extractor works")
                problems.append("python -m pip install -r code/requirements.txt (installs pdfplumber)")

    print("\nScreenshot OCR (11 screenshots; needed unless a vision model reads them)")
    from .evidence import images
    shot = None
    if ds is not None:
        shot = next((e for e in ds.evidence_files if e.file_path.endswith("account_admin_console_20260926.png")), None) \
            or next((e for e in ds.evidence_files if e.ext == ".png"), None)
    import importlib.util

    if importlib.util.find_spec("PIL") and importlib.util.find_spec("pytesseract"):
        _ok("Python packages pillow and pytesseract installed")
    else:
        _bad("pillow / pytesseract not installed")
        problems.append("python -m pip install -r code/requirements.txt")
    cmd = images.find_tesseract()
    if cmd:
        try:
            out = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=30)
            first = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "?"
            _ok(f"tesseract found: {cmd} ({first})")
        except Exception as exc:
            _bad(f"tesseract at {cmd} does not run: {exc}")
            problems.append("reinstall tesseract or fix TESSERACT_CMD")
    else:
        _bad("tesseract program not found")
        if platform.system() == "Windows":
            problems.append("install tesseract (UB Mannheim build: https://github.com/UB-Mannheim/tesseract/wiki), "
                            r"then set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe in .env")
        elif platform.system() == "Darwin":
            problems.append("brew install tesseract")
        else:
            problems.append("sudo apt install tesseract-ocr")
    if shot is not None and cmd:
        text = images.ocr_text(shot.abs_path)
        n = _ids_in(images.normalise_ocr_text(text))
        (_ok if n >= 2 else _bad)(f"OCR of {os.path.basename(shot.file_path)}: {n} identifiers recognised"
                                  + ("" if n >= 2 else f" ({images.OCR_STATUS.get('reason')})"))
        if n < 2:
            problems.append("OCR runs but reads nothing useful; check the tesseract install (English data)")

    print("\nModel providers")
    run_provider = os.environ.get("GHOST_AUDIT_PROVIDER", "").strip().lower() or next(
        (p for p in ("gemini", "groq", "anthropic") if any(os.environ.get(k, "").strip() for k in PROVIDER_KEY_ENV[p])), "")
    working: List[str] = []
    optional_problems: List[str] = []
    any_key = False
    for prov in providers:
        if not any(os.environ.get(k, "").strip() for k in PROVIDER_KEY_ENV[prov]):
            continue
        any_key = True
        layer = ModelLayer.from_env(provider=prov, classify_mode="ambiguous", verbose=False)
        if layer is None:
            continue
        ok = layer.preflight()
        if ok:
            working.append(prov)
            _ok(f"{prov}: key works; text model {layer.text_model}"
                + (f" ({len(layer.available_models)} models visible)" if layer.available_models else ""))
        else:
            _bad(f"{prov}: {layer.disabled_reason or 'no text model answered'}")
            seen = set()
            for e in layer.error_log:
                key = (e["model"], e["kind"], e["status"], e["message"][:160])
                if key in seen:
                    continue
                seen.add(key)
                print(f"           {e['model']}: {e['kind']} HTTP {e['status']}: {e['message'][:160]}")
                if len(seen) >= 6:
                    break
            optional_problems.append(f"{prov} does not work (see above); "
                                     f"{'it is the provider your runs use' if prov == run_provider else 'not needed while ' + (run_provider or 'another provider') + ' works'}")
        if ok and vision_probe and shot is not None:
            import base64
            with open(shot.abs_path, "rb") as fh:
                payload = base64.b64encode(fh.read()).decode("ascii")
            resp, model = layer._run_with_failover("vision", SCREENSHOT_PROMPT, ("image/png", payload), 8192, False,
                                                   "doctor screenshot")
            if resp is not None:
                n = _ids_in(resp.text)
                (_ok if n >= 2 else _warn)(f"{prov}: vision model {model} read the test screenshot ({n} identifiers)")
            else:
                _warn(f"{prov}: no vision model worked; screenshots will use local OCR")
                for e in layer.error_log[-3:]:
                    print(f"           {e['model']}: {e['kind']} HTTP {e['status']}: {e['message'][:160]}")
        s = layer.summary()
        print(f"           ({s['calls']} request(s) used by this check)")
    if not any_key:
        _warn("no API key found: --use-llm will fall back to the deterministic run")
    elif run_provider and run_provider not in working:
        problems.extend(optional_problems or [f"{run_provider} does not work; fix it or set GHOST_AUDIT_PROVIDER "
                                              "to a provider that works"])
    elif optional_problems:
        print(f"  note: --use-llm will use {run_provider}; " + "; ".join(optional_problems))

    print("\nVerdict")
    if problems:
        for p in dict.fromkeys(problems):
            print(f"  - {p}")
        print("\nFix the items above, then run:  python code/main.py --doctor  again.")
        return 1
    print("  everything needed for a full run is in place.")
    return 0
