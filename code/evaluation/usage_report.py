#!/usr/bin/env python3
"""Render code/evaluation/usage_report.md from the usage_run.json written by the last `python code/main.py`.

    python code/evaluation/usage_report.py            # print the report
    python code/evaluation/usage_report.py --write    # also write code/evaluation/usage_report.md

Run this right after the full-dataset run that produced the submitted output.csv,
so the report describes that run: provider, models, calls, failures, tokens, cost,
where the screenshots were read, and any problem the run reported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def render(run: dict) -> str:
    ml = run.get("model_layer") or {}
    cases = int(run.get("cases") or 0)
    enabled = bool(ml.get("enabled"))
    calls = int(ml.get("calls", 0))
    failures = int(ml.get("failures", sum(m.get("failures", 0) for m in (ml.get("models") or {}).values())))
    tin = int(ml.get("input_tokens", 0))
    tout = int(ml.get("output_tokens", 0))
    total = int(ml.get("total_tokens", tin + tout))
    cost = float(ml.get("estimated_cost_usd", 0.0))
    status = str(ml.get("status", "ok" if enabled else "not used"))
    shots = run.get("screenshots") or {}

    lines = ["# Usage report: final full-dataset run", ""]
    lines += [f"Source: `code/evaluation/usage_run.json` written by `python code/main.py` "
              f"(requests file `{os.path.basename(str(run.get('requests_file', '')))}`, {cases} cases, "
              f"{run.get('elapsed_seconds', 0)} s wall-clock).", ""]
    if enabled and (status != "ok" or (calls and not total)):
        lines += ["> **Warning: the model layer did not work in this run.** "
                  f"Status: {status}. The answers come from the deterministic engine (rules and local OCR); "
                  "the calls below are failed attempts. Fix the cause (`python code/main.py --doctor`) and re-run "
                  "before using this report.", ""]

    lines += ["## 1. Model providers, models and calls", ""]
    lines += ["| Provider | Model | Role | Calls | Failures | Input tokens | Output tokens | Total tokens | Est. cost (USD) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    if enabled and ml.get("models"):
        for name, m in ml["models"].items():
            role = []
            if name == ml.get("text_model"):
                role.append("request classification")
            if name == ml.get("vision_model"):
                role.append("screenshot transcription")
            if not role:
                role.append("tried, not used" if not m.get("input_tokens") else "partly used")
            lines.append(f"| {m.get('provider', ml.get('provider'))} | {name} | {', '.join(role)} | {m['calls']} | "
                         f"{m.get('failures', 0)} | {m['input_tokens']:,} | {m['output_tokens']:,} | "
                         f"{m.get('total_tokens', m['input_tokens'] + m['output_tokens']):,} | {m.get('estimated_cost_usd', 0):.4f} |")
    else:
        lines.append("| none (deterministic engine) | none | all stages | 0 | 0 | 0 | 0 | 0 | 0.0000 |")
    lines.append(f"| **Overall** | | | **{calls}** | **{failures}** | **{tin:,}** | **{tout:,}** | **{total:,}** | **{cost:.4f}** |")
    lines += ["", "Output tokens include reasoning (\"thinking\") tokens where the provider reports them, because they "
              "are billed as output. Costs are estimates at paid list prices; free-tier usage costs nothing.", ""]

    per = (lambda v: v / cases if cases else 0.0)
    lines += ["## 2. Per request", "",
              f"* model calls per request: {per(calls):.3f}",
              f"* input tokens per request: {per(tin):.1f}; output tokens per request: {per(tout):.1f}; "
              f"total tokens per request: {per(total):.1f}",
              f"* estimated cost per request: USD {per(cost):.6f}", ""]

    lines += ["## 3. Where the evidence text came from", ""]
    if shots:
        lines.append(f"* screenshots: {shots.get('total', 0)} files; read by the vision model: {shots.get('model', 0)}; "
                     f"by local OCR: {shots.get('ocr', 0)}; cached from an earlier run: {shots.get('cached', 0)}; "
                     f"with no usable evidence: {shots.get('unread', 0)}")
    lines.append("* PDFs: local text extraction (pdfplumber / pypdf / pdftotext); logs, JSON and SQL: parsed directly")
    for w in run.get("warnings") or []:
        lines.append(f"* run warning: {w}")
    lines.append("")

    if enabled:
        lines += ["## 4. Configuration of this run", "",
                  f"* provider: `{ml.get('provider')}`; status: {status}",
                  f"* text model: `{ml.get('text_model')}`; vision model: `{ml.get('vision_model')}`",
                  f"* classification mode: `{ml.get('classify_mode')}` (ambiguous = model consulted only when the rules are "
                  "unsure; all = every request text is classified by both the rules and the model), "
                  f"batch size {ml.get('batch_size')}, pacing {ml.get('min_interval_s')} s between requests",
                  "* prices used for the estimate (USD per million tokens): " +
                  (", ".join(f"{n}: in {m.get('price_in_per_mtok')}, out {m.get('price_out_per_mtok')}"
                             for n, m in (ml.get("models") or {}).items()) or "n/a"), ""]
        sel = ml.get("model_selection") or []
        dead = ml.get("unusable_models") or {}
        if sel or dead:
            lines += ["## 5. Model selection", ""]
            for s in sel:
                lines.append(f"* {s.get('purpose')}: `{s.get('model')}` {s.get('outcome')}"
                             + (f" (HTTP {s.get('status')}: {s.get('message', '')[:140]})" if s.get("status") else ""))
            for name, why in dead.items():
                if not any(s.get("model") == name for s in sel):
                    lines.append(f"* `{name}` unusable: {why[:160]}")
            lines.append("")
        agr = ml.get("classification_agreement") or {}
        lines += ["## 6. Rules vs model agreement on request classification", "",
                  f"* agree: {agr.get('agree', 0)}; disagree (rules kept, logged): {agr.get('disagree', 0)}; "
                  f"decided by rules alone: {agr.get('rules_only', 0)}; decided by the model (rules unsure): "
                  f"{agr.get('model_only', 0)}; not classified by the model: {agr.get('unclassified', 0)}", ""]
        for d in ml.get("disagreements") or []:
            lines.append(f"  * rules={d.get('rules')} model={d.get('model')} ({d.get('model_reason', '')}): {d.get('text')}")
        if ml.get("disagreements"):
            lines.append("")
        errs = ml.get("error_counts") or {}
        if errs:
            lines += ["## 7. Errors reported by the provider", "",
                      "* counts: " + ", ".join(f"{k}: {v}" for k, v in errs.items())]
            seen = set()
            for e in ml.get("errors") or []:
                key = (e.get("model"), e.get("kind"), e.get("status"))
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"* `{e.get('model')}` ({e.get('purpose')}): {e.get('kind')}, HTTP {e.get('status')}: "
                             f"{str(e.get('message', ''))[:200]}")
            lines.append("")
    else:
        lines += ["## 4. Notes", "",
                  "The run was fully deterministic: request classification by multilingual rules (checked against the "
                  "privacy intake's own `CLASSIFY_REQUEST` rows), evidence extraction with the standard library, PDFs with "
                  "pdfplumber and screenshots with local tesseract OCR. No API tokens were consumed. The optional model "
                  "layer (`--use-llm`, Gemini / Groq / Anthropic) is documented in `code/README.md`.", "",
                  "Expected volume with the model layer on this dataset (`--use-llm --llm-classify all`): 1 preflight "
                  "request, 15 batched classification requests (375 distinct request texts, 25 per request) and 11 "
                  "screenshot requests, about 27 requests and roughly 90k tokens including reasoning tokens. At the "
                  "estimated list prices of Gemini Flash-Lite (text) and Flash (screenshots) that is about USD 0.10-0.15 "
                  "for the dataset, under USD 0.001 per request, and nothing on the free tier. Run "
                  "`python code/evaluation/usage_report.py --write` after such a run to replace this report with the "
                  "measured numbers.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(HERE, "usage_run.json"))
    ap.add_argument("--write", action="store_true", help="write code/evaluation/usage_report.md")
    args = ap.parse_args(argv)
    with open(args.run, encoding="utf-8") as fh:
        run = json.load(fh)
    text = render(run)
    print(text)
    if args.write:
        out = os.path.join(HERE, "usage_report.md")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"\nwritten to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
