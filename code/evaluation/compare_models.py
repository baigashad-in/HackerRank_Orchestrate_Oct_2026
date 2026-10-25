#!/usr/bin/env python3
"""Model comparison: rules-only engine vs every model provider that has a key (Gemini, Groq, Anthropic).

    python code/evaluation/compare_models.py                       # every provider with a key in .env / env
    python code/evaluation/compare_models.py --providers gemini groq
    python code/evaluation/compare_models.py --fresh-vision        # re-read screenshots instead of the cache

For each provider it classifies every request text (batched, ``--llm-classify all`` behaviour), reads the
eleven screenshots with the provider's vision model, runs the full audit with both, and records calls,
tokens, estimated cost and wall time. It then compares, against the rules-only run and between providers:

* request classification: agreement with the multilingual rules and between providers;
* screenshot facts: for each screenshot, whether the records and states extracted from the model's
  transcript equal those extracted from local OCR;
* final answers: cases and steps that differ from the rules-only output.

Writes ``code/evaluation/model_comparison.md`` (and ``model_comparison.json``). It never writes
``output.csv`` or ``usage_run.json``, so the final run's usage report is not touched. Keys are read from the
environment or the uncommitted ``.env``; they are never printed or written. Request texts in the report are
shortened and e-mail addresses masked.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ghost_audit import config, envfile, loaders  # noqa: E402
from ghost_audit.evidence import images  # noqa: E402
from ghost_audit.pipeline import AuditEngine  # noqa: E402
from ghost_audit.request_classifier import classify_request  # noqa: E402

PROVIDERS = ("gemini", "groq", "anthropic")
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def short(text: str, n: int = 80) -> str:
    t = EMAIL.sub("<email>", " ".join((text or "").split()))
    return t if len(t) <= n else t[: n - 3] + "..."


def image_facts(engine) -> dict:
    """file_id -> sorted list of (resource, present, identifiable, enabled, restorable, available)."""
    image_ids = {e.file_id for e in engine.ds.evidence_files if e.ext in (".png", ".jpg", ".jpeg", ".webp")}
    facts = {fid: set() for fid in image_ids}
    for o in engine.observations:
        if o.file_id in facts:
            facts[o.file_id].add((o.resource_id, o.present, o.identifiable, o.enabled, o.restore_allowed, o.available))
    return {fid: sorted(v, key=str) for fid, v in sorted(facts.items(), key=lambda x: config.case_sort_key(x[0]))}


def rows_of(results) -> dict:
    return {r.case.case_id: (r.status, r.steps_json()) for r in results}


def diff_rows(a: dict, b: dict):
    cases = [cid for cid in a if a[cid] != b.get(cid)]
    steps_a = sum(len(json.loads(v[1])) for v in a.values())
    matched = 0
    for cid in a:
        sa = {tuple(sorted(s.items())) for s in json.loads(a[cid][1])}
        sb = {tuple(sorted(s.items())) for s in json.loads(b.get(cid, ("", "[]"))[1])}
        matched += len(sa & sb)
    status = [cid for cid in a if a[cid][0] != b.get(cid, ("",))[0]]
    return {"cases_differing": cases, "status_differing": status, "steps": steps_a, "steps_matched": matched}


def snapshot_read_stats() -> dict:
    return dict(images.READ_STATS)


def delta_stats(before: dict, after: dict) -> dict:
    return {k: after[k] - before.get(k, 0) for k in after}


def run_rules(ds):
    t0 = time.time()
    engine = AuditEngine(ds)
    results = engine.run()
    return {"engine": engine, "results": results, "seconds": round(time.time() - t0, 1)}


class TrackingReader:
    """Wraps the provider's screenshot reader and records, per file, whether the model produced the text.

    A cached transcript never reaches the reader (it is served from the cache, which only holds model text),
    so a file the reader was not asked about counts as read by the model from the cache.
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.cache_tag = inner.cache_tag
        self.last_from_model = False
        self.from_model = {}

    def __call__(self, path: str) -> str:
        text = self.inner(path)
        self.last_from_model = self.inner.last_from_model
        self.from_model[os.path.abspath(path)] = self.last_from_model
        return text


def run_provider(ds, provider: str, *, vision: bool, fresh_vision: bool):
    from ghost_audit import llm

    client = llm.ModelLayer.from_env(provider=provider, classify_mode="all")
    if client is None:
        return {"provider": provider, "skipped": "no API key in the environment or .env"}
    t0 = time.time()
    print(f"[compare] {provider}: checking the key and choosing models ...", file=sys.stderr)
    if not client.preflight():
        return {"provider": provider, "skipped": client.disabled_reason or "preflight failed", "usage": client.summary()}
    texts = [c.request_text for c in ds.cases]
    client.prefetch_classifications(texts)
    reader = None
    if vision:
        inner = llm.ScreenshotReader(client)
        if fresh_vision:
            inner.cache_tag = f"{provider}-vision-compare-{int(time.time())}"
        reader = TrackingReader(inner)
    before = snapshot_read_stats()
    engine = AuditEngine(ds, image_reader=reader, classifier=client.classify_request)
    results = engine.run()
    reads = delta_stats(before, snapshot_read_stats())
    if reader is not None and fresh_vision:
        # transcripts made under the one-off tag are never reused; do not leave them in the cache
        import glob

        for f in glob.glob(os.path.join(config.REPO_ROOT, ".ghost_audit_cache", "ocr", f"*{reader.cache_tag}*")):
            os.remove(f)
    verdicts = {t: v[0] for t, v in client.batch_results.items()}
    # where each screenshot's text came from: "fresh" (model now), "cached" (model, earlier run) or "ocr" (fallback)
    shot_source = {}
    for e in ds.evidence_files:
        if e.ext not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        if reader is None:
            shot_source[e.file_id] = "ocr"
        else:
            seen = reader.from_model.get(os.path.abspath(e.abs_path))
            shot_source[e.file_id] = "cached" if seen is None else ("fresh" if seen else "ocr")
    distinct = {c.request_text for c in ds.cases}
    classified = sum(1 for t in distinct if t in verdicts)
    by_model = sum(1 for v in shot_source.values() if v != "ocr")
    complete = client.enabled and classified == len(distinct) and (not vision or by_model == len(shot_source))
    problems = []
    if classified < len(distinct):
        problems.append(f"classified {classified}/{len(distinct)} request texts")
    if vision and by_model < len(shot_source):
        problems.append(f"{len(shot_source) - by_model}/{len(shot_source)} screenshots fell back to local OCR")
    if not client.enabled:
        problems.append(f"model layer stopped: {client.disabled_reason}")
    return {"provider": provider, "engine": engine, "results": results, "seconds": round(time.time() - t0, 1),
            "usage": client.summary(), "verdicts": verdicts, "reads": reads, "fresh_vision": fresh_vision,
            "vision": vision, "shot_source": shot_source, "complete": complete, "problems": problems}


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.1%})" if b else "0/0"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="compare the rules-only engine with each model provider")
    ap.add_argument("--providers", nargs="*", default=None, help=f"subset of {', '.join(PROVIDERS)} (default: all with a key)")
    ap.add_argument("--no-vision", action="store_true", help="classification only; screenshots stay on local OCR")
    ap.add_argument("--fresh-vision", action="store_true", help="ignore cached model transcripts of the screenshots")
    ap.add_argument("--out", default=os.path.join(HERE, "model_comparison.md"))
    args = ap.parse_args(argv)

    loaded = envfile.load_dotenv([config.REPO_ROOT, os.getcwd()])
    if loaded:
        print(f"[env] loaded {', '.join(loaded)} from .env (values not shown)", file=sys.stderr)
    ds = loaders.load_dataset(config.DATASET_DIR)
    texts = sorted({c.request_text for c in ds.cases})
    case_of_text = {}
    for c in ds.cases:
        case_of_text.setdefault(c.request_text, c.case_id)

    print("[compare] rules-only run (local OCR) ...", file=sys.stderr)
    base = run_rules(ds)
    base_rows = rows_of(base["results"])
    base_facts = image_facts(base["engine"])
    rules = {t: classify_request(t) for t in texts}

    wanted = args.providers or [p for p in PROVIDERS]
    runs = []
    for p in wanted:
        r = run_provider(ds, p, vision=not args.no_vision, fresh_vision=args.fresh_vision)
        if "skipped" in r:
            print(f"[compare] {p}: skipped ({r['skipped']})", file=sys.stderr)
        runs.append(r)
    done = [r for r in runs if "skipped" not in r]

    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "cases": len(ds.cases),
           "distinct_request_texts": len(texts), "rules_seconds": base["seconds"], "providers": {}}
    lines = ["# Model comparison", "",
             f"Generated {out['generated_at']} by `python code/evaluation/compare_models.py` on {len(ds.cases)} requests "
             f"({len(texts)} distinct request texts) and {len(base_facts)} screenshots. Baseline: the rules-only "
             f"engine with local OCR ({base['seconds']} s). Every provider classified every request text "
             f"(`--llm-classify all`) and, unless noted, read every screenshot with its vision model.", ""]

    # 1. usage
    lines += ["## 1. Calls, tokens, cost and time", "",
              "| Provider | Coverage | Text model | Vision model | Calls | Failed calls | Input tokens | Output tokens | Est. cost (USD) | Wall time (s) | Screenshots: fresh / cached / OCR fallback |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        if "skipped" in r:
            lines.append(f"| {r['provider']} | skipped: {r['skipped']} | | | | | | | | | |")
            out["providers"][r["provider"]] = {"skipped": r["skipped"]}
            continue
        u = r["usage"]
        src = list(r["shot_source"].values())
        fresh, cached, ocr = src.count("fresh"), src.count("cached"), src.count("ocr")
        vision_label = u["vision_model"] or (f"cached transcripts ({r['provider']}-vision)" if cached else "not used")
        coverage = "complete" if r["complete"] else "**incomplete**"
        lines.append(f"| {r['provider']} | {coverage} | {u['text_model']} | {vision_label} | {u['calls']} | {u['failures']} | "
                     f"{u['input_tokens']:,} | {u['output_tokens']:,} | {u['estimated_cost_usd']:.4f} | {r['seconds']} | "
                     f"{fresh} / {cached} / {ocr} |")
        out["providers"][r["provider"]] = {k: u[k] for k in ("text_model", "vision_model", "calls", "failures", "input_tokens",
                                                            "output_tokens", "estimated_cost_usd", "error_counts",
                                                            "rate_limit_waits", "tokens_per_minute_limits")}
        out["providers"][r["provider"]].update({"seconds": r["seconds"], "screenshot_source": r["shot_source"],
                                                "complete": r["complete"], "problems": r["problems"]})
    lines += ["", "Costs are estimates at paid list prices (free-tier usage costs nothing). Output tokens include "
              "reasoning tokens where the provider reports them. Failed calls include rate-limit answers (HTTP 429) "
              "that were waited out and retried. Cached screenshots were read by the same provider in an earlier run "
              "and cost nothing here.", ""]
    for r in done:
        if not r["complete"]:
            lines.append(f"* **{r['provider']} is incomplete**: " + "; ".join(r["problems"])
                         + ". Where it fell back, the rules or local OCR stood in for the model, so those parts are "
                           "not an independent check and are marked as such below.")
        waits = r["usage"].get("rate_limit_waits", 0)
        limits = r["usage"].get("tokens_per_minute_limits") or {}
        if limits:
            lines.append(f"* {r['provider']}: tokens-per-minute limit reported by the provider "
                         + ", ".join(f"{m}: {n:,}" for m, n in limits.items())
                         + f"; requests were paced under it ({waits} pacing waits).")
    lines.append("")

    # 2. classification
    lines += ["## 2. Request classification", "",
              "| Provider | Texts classified | Agree with rules | Disagree with rules | Rules unsure (model decided) |",
              "| --- | --- | --- | --- | --- |"]
    for r in done:
        v = r["verdicts"]
        cl = [t for t in texts if t in v]
        agree = sum(1 for t in cl if v[t] == rules[t].authorises_deletion)
        unsure = sum(1 for t in cl if rules[t].confidence < 0.95)
        lines.append(f"| {r['provider']} | {pct(len(cl), len(texts))} | {pct(agree, len(cl))} | {len(cl) - agree} | {unsure} |")
        out["providers"][r["provider"]]["classification"] = {"classified": len(cl), "agree_with_rules": agree,
                                                             "disagree_with_rules": len(cl) - agree}
    if len(done) >= 2:
        lines += ["", "Between providers:", ""]
        for i in range(len(done)):
            for j in range(i + 1, len(done)):
                a, b = done[i], done[j]
                both = [t for t in texts if t in a["verdicts"] and t in b["verdicts"]]
                same = sum(1 for t in both if a["verdicts"][t] == b["verdicts"][t])
                lines.append(f"* {a['provider']} vs {b['provider']}: agree on {pct(same, len(both))} texts")
    dis = []
    for t in texts:
        row = {"case": case_of_text[t], "rules": rules[t].authorises_deletion, "rules_conf": rules[t].confidence}
        differs = False
        for r in done:
            if t in r["verdicts"]:
                row[r["provider"]] = r["verdicts"][t]
                differs = differs or r["verdicts"][t] != rules[t].authorises_deletion
        if differs or (len({row.get(r["provider"]) for r in done if r["provider"] in row}) > 1):
            row["text"] = short(t)
            dis.append(row)
    lines += ["", f"Texts where any provider disagrees with the rules or with another provider: {len(dis)}", ""]
    if dis:
        hdr = "| First case | Rules (confidence) | " + " | ".join(r["provider"] for r in done) + " | Request text (shortened) |"
        lines += [hdr, "| --- | --- | " + " | ".join("---" for _ in done) + " | --- |"]
        for row in dis[:40]:
            lines.append(f"| {row['case']} | {'deletion' if row['rules'] else 'not deletion'} ({row['rules_conf']}) | "
                         + " | ".join(("deletion" if row.get(r['provider']) else "not deletion") if r["provider"] in row else "-"
                                      for r in done)
                         + f" | {row['text']} |")
        lines += ["", "When the rules are confident and a model disagrees, the engine keeps the rules' verdict "
                  "(they agree with the privacy intake's own CLASSIFY_REQUEST rows); the table shows what each model said.", ""]
    out["classification_disagreements"] = dis

    # 3. screenshots
    lines += ["## 3. Screenshot facts (vision model vs local OCR)", ""]
    if not any(r["vision"] for r in done):
        lines += ["Not compared (`--no-vision`).", ""]
    else:
        hdr = "| Screenshot | OCR records | " + " | ".join(f"{r['provider']}" for r in done if r["vision"]) + " |"
        lines += [hdr, "| --- | --- | " + " | ".join("---" for r in done if r["vision"]) + " |"]
        names = {e.file_id: os.path.basename(e.file_path) for e in ds.evidence_files}
        per = {}
        for fid, facts in base_facts.items():
            cells = []
            for r in done:
                if not r["vision"]:
                    continue
                per.setdefault(r["provider"], [0, 0])
                source = r["shot_source"].get(fid, "ocr")
                if source == "ocr":
                    cells.append("not compared (fell back to OCR)")
                    continue
                theirs = image_facts(r["engine"]).get(fid, [])
                same = theirs == facts
                per[r["provider"]][0] += int(same)
                per[r["provider"]][1] += 1
                cells.append(("same facts" if same else f"differs ({len(theirs)} records)")
                             + (" (cached)" if source == "cached" else ""))
            lines.append(f"| {fid} {names.get(fid, '')} | {len(facts)} | " + " | ".join(cells) + " |")
        lines += ["", "\"Same facts\" means the records, presence and identifier exposure extracted from the "
                  "model's transcript equal those from local OCR. \"(cached)\" marks a transcript the same provider "
                  "made in an earlier run.", ""]
        for p, (n, compared) in per.items():
            lines.append(f"* {p}: {n}/{compared} screenshots read by the model give the same facts as local OCR"
                         + (f" ({len(base_facts) - compared} not read by the model)" if compared < len(base_facts) else ""))
            out["providers"][p]["screenshots_same_as_ocr"] = n
            out["providers"][p]["screenshots_compared"] = compared
        lines.append("")

    # 4. final answers
    lines += ["## 4. Final answers (`output.csv` rows) vs the rules-only run", "",
              "| Provider | Cases identical | Status changes | Steps matched |", "| --- | --- | --- | --- |"]
    for r in done:
        d = diff_rows(base_rows, rows_of(r["results"]))
        lines.append(f"| {r['provider']} | {pct(len(base_rows) - len(d['cases_differing']), len(base_rows))} | "
                     f"{len(d['status_differing'])} | {pct(d['steps_matched'], d['steps'])} |")
        out["providers"][r["provider"]]["answers"] = {"cases_differing": d["cases_differing"],
                                                      "status_differing": d["status_differing"],
                                                      "steps_matched": d["steps_matched"], "steps": d["steps"]}
        if d["cases_differing"]:
            lines.append(f"|  | differing cases: {', '.join(d['cases_differing'][:15])}"
                         + (" ..." if len(d["cases_differing"]) > 15 else "") + " | | |")
    for r in done:
        if not r["complete"]:
            lines.append(f"\n{r['provider']}: its answers used the rules or local OCR wherever it fell back "
                         f"({'; '.join(r['problems'])}), so agreement here is not evidence about the model.")
    lines.append("")

    # 5. reading
    lines += ["## 5. Reading the comparison", ""]
    full = [r for r in done if r["complete"]]
    partial = [r for r in done if not r["complete"]]
    same_all = all(not out["providers"][r["provider"]]["answers"]["cases_differing"] for r in full)
    if not done:
        lines.append("No provider could be run; see the skipped reasons in section 1.")
    elif not full:
        lines.append("No provider completed the comparison (see section 1), so no conclusion about the models is drawn. "
                     "Re-run after the limits reset, or with `--providers <name>` for one provider at a time.")
    elif same_all:
        names = ", ".join(r["provider"] for r in full)
        cheapest = min(full, key=lambda r: (r["usage"]["estimated_cost_usd"], r["seconds"]))
        fastest = min(full, key=lambda r: r["seconds"])
        lines.append(f"Every provider that completed ({names}) reproduces the rules-only answers on all {len(base_rows)} "
                     f"cases, so the models act as an independent check rather than a source of extra accuracy on this "
                     f"dataset. Lowest estimated cost: {cheapest['provider']}; fastest: {fastest['provider']}. A "
                     f"disagreement table above (if any) shows where the models and the rules read a request differently.")
    else:
        lines.append("At least one provider changes some answers relative to the rules-only run. Inspect the differing "
                     "cases with `python code/main.py --use-llm --provider <name> --explain <case>` and compare them "
                     "with the run that produced the submitted output.csv before trusting either version.")
    if partial and done:
        lines.append("")
        lines.append("Not counted in this conclusion because they did not complete: "
                     + "; ".join(f"{r['provider']} ({'; '.join(r['problems'])})" for r in partial) + ".")
    lines.append("")
    # keep the previous report (outside the package) instead of silently replacing it
    for path in (args.out, os.path.splitext(args.out)[0] + ".json"):
        if os.path.exists(path):
            keep = os.path.join(config.REPO_ROOT, ".ghost_audit_cache", "reports")
            os.makedirs(keep, exist_ok=True)
            stamp = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            stem, ext = os.path.splitext(os.path.basename(path))
            os.replace(path, os.path.join(keep, f"{stem}_{stamp}{ext}"))
            print(f"[compare] previous {os.path.basename(path)} kept as .ghost_audit_cache/reports/{stem}_{stamp}{ext}",
                  file=sys.stderr)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\n".join(lines))
    print(f"\nwritten to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
