#!/usr/bin/env python3
"""Offline check of compare_models.py with fake Gemini and Groq providers (no keys, no network).

    python code/evaluation/test_compare_models.py

The fake providers classify like the rules and return one fixed transcript for every screenshot. Two
scenarios: Groq answers its first request with the per-minute 429 seen on a real run (with an upgrade/billing
hint), which must be waited out; and Groq stops halfway (no credit), which must be reported as incomplete
instead of as agreement. A provider without a key is skipped, and output.csv and usage_run.json are never
touched.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import test_llm_adapters as fakes  # noqa: E402
from ghost_audit import config, llm  # noqa: E402

MODELS = {"gemini": ["gemini-3.5-flash-lite", "gemini-3.8-flash"], "groq": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]}


class FailsOnNthPost(fakes.FakeProvider):
    """Answers normally, then raises ``error`` on the n-th generate request (the provider stopping mid-run)."""

    def __init__(self, provider, models, n, error):
        super().__init__(provider, models)
        self.n, self.error, self.posts = n, error, 0

    def __call__(self, url, headers, body, method="POST"):
        if method == "POST":
            self.posts += 1
            if self.posts >= self.n:
                self.requests.append((method, url, headers, None))
                raise self.error
        return super().__call__(url, headers, body, method)


def run(scenario, out_dir):
    import compare_models

    def fake_from_env(provider=None, classify_mode="all", verbose=True):
        if provider not in MODELS:
            return None
        if provider == "groq" and scenario == "groq_tpm":
            fake = fakes.FakeProvider("groq", MODELS["groq"], script={"openai/gpt-oss-120b": [fakes.groq_error(429, fakes.GROQ_TPM)]})
        elif provider == "groq" and scenario == "groq_stops":
            fake = FailsOnNthPost("groq", MODELS["groq"], 3, llm.ProviderHTTPError(
                429, json.dumps({"error": {"message": "Your credit balance is too low (organization org_01testorganisation0000)."}})))
        else:
            fake = fakes.FakeProvider(provider, MODELS[provider])
        return fakes.make_layer(provider, fake, mode=classify_mode)

    llm.ModelLayer.from_env = staticmethod(fake_from_env)
    out = os.path.join(out_dir, f"model_comparison_{scenario}.md")
    rc = compare_models.main(["--providers", "gemini", "groq", "anthropic", "--fresh-vision", "--out", out])
    data = json.load(open(os.path.splitext(out)[0] + ".json", encoding="utf-8"))
    text = open(out, encoding="utf-8").read()
    return rc, data, text


def main() -> int:
    original = llm.ModelLayer.from_env
    protected = [os.path.join(config.REPO_ROOT, "output.csv"), os.path.join(HERE, "usage_run.json")]
    stamps = {p: os.path.getmtime(p) for p in protected if os.path.exists(p)}
    cache_before = set(glob.glob(os.path.join(config.REPO_ROOT, ".ghost_audit_cache", "ocr", "*")))
    checks = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rc, data, text = run("groq_tpm", tmp)
            g, q = data["providers"]["gemini"], data["providers"]["groq"]
            checks += [
                ("runs and exits 0", rc == 0),
                ("provider without a key is skipped, not fatal", "skipped" in data["providers"]["anthropic"]),
                ("Groq per-minute 429 with a billing hint is waited out: Groq completes",
                 q["complete"] and q["classification"]["classified"] == data["distinct_request_texts"]),
                ("the tokens-per-minute limit is reported", q["tokens_per_minute_limits"].get("openai/gpt-oss-120b") == 8000),
                ("both providers classified every request text and agree with the rules",
                 all(data["providers"][p]["classification"]["disagree_with_rules"] == 0 for p in MODELS)),
                ("every screenshot was read by each model (fresh)",
                 all(set(data["providers"][p]["screenshot_source"].values()) == {"fresh"} for p in MODELS)),
                ("a fixed fake transcript is reported as different screenshot facts",
                 all(data["providers"][p]["screenshots_same_as_ocr"] < data["providers"][p]["screenshots_compared"] for p in MODELS)),
                ("calls, tokens and cost are reported", g["calls"] > 0 and q["calls"] > 0),
                ("report has all five sections", all(f"## {i}." in text for i in range(1, 6))),
                ("no e-mail address in the report", "@" not in text),
            ]
            rc, data, text = run("groq_stops", tmp)
            q = data["providers"]["groq"]
            checks += [
                ("a provider that stops mid-run is marked incomplete", rc == 0 and not q["complete"] and q["problems"]),
                ("its screenshots are not claimed as compared", q.get("screenshots_compared", 0) == 0
                 and set(q["screenshot_source"].values()) == {"ocr"}),
                ("the conclusion leaves it out and says why", "Not counted in this conclusion" in text and "groq (" in text),
                ("its organization id is masked", "org_01testorganisation0000" not in text + json.dumps(data)),
                ("the provider that completed is still reported", data["providers"]["gemini"]["complete"]),
            ]
        checks += [
            ("output.csv and usage_run.json untouched", all(os.path.getmtime(p) == t for p, t in stamps.items())),
            ("one-off screenshot transcripts removed from the cache",
             not [f for f in set(glob.glob(os.path.join(config.REPO_ROOT, ".ghost_audit_cache", "ocr", "*"))) - cache_before
                  if "compare" in f]),
        ]
    finally:
        llm.ModelLayer.from_env = original
        for f in set(glob.glob(os.path.join(config.REPO_ROOT, ".ghost_audit_cache", "ocr", "*"))) - cache_before:
            os.remove(f)
    ok = True
    for name, cond in checks:
        print(("  ok   " if cond else "  FAIL ") + name)
        ok = ok and bool(cond)
    print(f"all {len(checks)} model-comparison checks passed" if ok else "model-comparison checks FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
