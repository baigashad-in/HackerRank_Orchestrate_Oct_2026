#!/usr/bin/env python3
"""Offline tests for the model layer (no network, no keys).

    python3 code/evaluation/test_llm_adapters.py

A fake transport plays the provider: it answers like Gemini, Groq or Anthropic,
or fails the way real providers fail (retired model 404, invalid key, per-minute
and per-day 429, network down).  Checked: request shapes and auth headers,
model discovery and failover, retry/backoff, stopping after fatal errors,
batched classification, token and cost accounting, the rules-vs-model merge,
the OCR fallback, and an end-to-end engine run that must reproduce the
deterministic answers.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CODE_DIR)

from ghost_audit import llm  # noqa: E402
from ghost_audit.request_classifier import classify_request  # noqa: E402

FAKE_KEY = "AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKE00000"
PASSED = 0


def check(cond, msg):
    global PASSED
    if not cond:
        raise AssertionError(msg)
    PASSED += 1
    print("  ok ", msg)


def gemini_error(status, message, status_name, retry=None):
    err = {"error": {"code": status, "message": message, "status": status_name}}
    if retry:
        err["error"]["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry}]
    return llm.ProviderHTTPError(status, json.dumps(err))


class FakeProvider:
    """Answers generateContent / chat.completions / messages; behaviour per model is scripted."""

    def __init__(self, provider, models, script=None, list_error=None, network_down=False):
        self.provider = provider
        self.models = models                    # ids the listing returns
        self.script = script or {}              # model -> list of exceptions to raise first, or "dead"
        self.list_error = list_error
        self.network_down = network_down
        self.requests = []                      # (method, url, headers, body)

    def _model_of(self, url, body):
        if self.provider == "gemini":
            return url.split("/models/")[1].split(":")[0]
        return body["model"]

    def __call__(self, url, headers, body, method="POST"):
        payload = json.loads(body.decode("utf-8")) if body else None
        self.requests.append((method, url, headers, payload))
        if self.network_down:
            raise llm.ProviderNetworkError("URLError: [Errno 11001] getaddrinfo failed")
        if method == "GET":
            if self.list_error:
                raise self.list_error
            if self.provider == "gemini":
                return {"models": [{"name": f"models/{m}", "supportedGenerationMethods": ["generateContent"]}
                                   for m in self.models]}
            return {"data": [{"id": m, "active": True} for m in self.models]}
        model = self._model_of(url, payload)
        plan = self.script.get(model)
        if plan == "dead":
            raise gemini_error(404, f"This model models/{model} is no longer available. Please update your code "
                                    "to use a newer model.", "NOT_FOUND")
        if isinstance(plan, list) and plan:
            raise plan.pop(0)
        reply = self._reply(payload)
        if self.provider == "gemini":
            return {"candidates": [{"content": {"parts": [{"text": reply}]}}],
                    "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 100, "thoughtsTokenCount": 50}}
        if self.provider == "groq":
            return {"choices": [{"message": {"content": "<think>hmm</think>" + reply}}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 150}}
        return {"content": [{"type": "text", "text": reply}], "usage": {"input_tokens": 1000, "output_tokens": 150}}

    def _reply(self, payload):
        if self.provider == "gemini":
            parts = payload["contents"][0]["parts"]
            text = parts[-1]["text"]
            has_image = any("inline_data" in p for p in parts)
        elif self.provider == "groq":
            content = payload["messages"][0]["content"]
            has_image = isinstance(content, list)
            text = content[0]["text"] if has_image else content
        else:
            content = payload["messages"][0]["content"]
            has_image = isinstance(content, list)
            text = content[-1]["text"] if has_image else content
        if has_image:
            return "Customer Operations\nHaruka Sato\nuser_11 · account_5762\nharuka.s1838@zoho.com\n+81 90 7610 2507"
        if "Messages (JSON):" in text:
            items = json.loads(text.split("Messages (JSON):", 1)[1])
            return json.dumps([{"id": it["id"], "authorises_deletion": classify_request(it["text"]).authorises_deletion,
                                "reason": "fake"} for it in items])
        return "OK"


def make_layer(provider, fake, mode="all"):
    layer = llm.ModelLayer(provider=provider, api_key=FAKE_KEY,
                           text_candidates=list(llm.PROVIDER_CANDIDATES[provider]["text"]),
                           vision_candidates=list(llm.PROVIDER_CANDIDATES[provider]["vision"]),
                           classify_mode=mode, transport=fake, min_interval=0.0, verbose=False)
    layer.sleeps = []
    layer.sleep = layer.sleeps.append
    return layer


def generate_calls(fake):
    return [r for r in fake.requests if r[0] == "POST"]


# ---------------------------------------------------------------------------
def test_retired_model_failover():
    # the situation on the user's PC: the first candidate is retired (404), discovery + failover find a live one
    fake = FakeProvider("gemini", ["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"],
                        script={"gemini-3.5-flash-lite": "dead", "gemini-3.1-flash-lite": "dead",
                                "gemini-flash-lite-latest": "dead"})
    layer = make_layer("gemini", fake)
    check(layer.preflight(), "preflight succeeds after skipping retired models")
    check(layer.text_model == "gemini-3.8-flash", f"text model fails over to a live model ({layer.text_model})")
    check("gemini-3.5-flash-lite" in layer.dead_models, "retired model recorded with its error")
    check(fake.requests[0][0] == "GET" and fake.requests[0][2]["x-goog-api-key"] == FAKE_KEY, "models are listed first")
    s = layer.summary()
    check(s["status"] == "ok" and s["error_counts"].get("model_unavailable", 0) >= 1, "summary reports the 404s")
    check(FAKE_KEY not in json.dumps(s), "the API key never appears in the summary")


def test_invalid_key_stops_immediately():
    fake = FakeProvider("gemini", [], list_error=gemini_error(400, "API key not valid. Please pass a valid API key.",
                                                              "INVALID_ARGUMENT"))
    layer = make_layer("gemini", fake)
    check(not layer.preflight(), "invalid key: preflight fails")
    check(len(generate_calls(fake)) == 0, "invalid key: no generate requests are wasted")
    check("API key not valid" in (layer.disabled_reason or ""), "invalid key: provider message is surfaced")
    res = layer.classify_request("Please permanently delete my account.")
    check(res.authorises_deletion and res.method == "rules", "disabled layer falls back to the rules")


def test_rate_limit_retry_and_daily_quota():
    fake = FakeProvider("gemini", ["gemini-3.5-flash-lite", "gemini-3.8-flash"], script={
        "gemini-3.5-flash-lite": [gemini_error(429, "Resource has been exhausted (e.g. check quota).",
                                               "RESOURCE_EXHAUSTED", retry="12s")]})
    layer = make_layer("gemini", fake)
    check(layer.preflight() and layer.text_model == "gemini-3.5-flash-lite", "per-minute 429: same model after waiting")
    check(layer.sleeps and abs(layer.sleeps[0] - 12.0) < 1e-6, f"waits the provider's retryDelay ({layer.sleeps[:1]})")
    fake2 = FakeProvider("gemini", ["gemini-3.5-flash-lite", "gemini-3.8-flash"], script={
        "gemini-3.5-flash-lite": [gemini_error(429, "Quota exceeded for metric: generate_content_free_tier_requests, "
                                                    "limit: 20, model: gemini-3.5-flash-lite. GenerateRequestsPerDay",
                                               "RESOURCE_EXHAUSTED")]})
    layer2 = make_layer("gemini", fake2)
    check(layer2.preflight() and layer2.text_model == "gemini-3.8-flash", "daily quota: moves to the next model")


def groq_error(status, message, code="rate_limit_exceeded"):
    return llm.ProviderHTTPError(status, json.dumps({"error": {"message": message, "type": "tokens", "code": code}}))


GROQ_TPM = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01testorganisation0000` service "
            "tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 6389, Requested 2108. Please try again in "
            "3.7275s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing")


def test_rate_limit_messages_with_billing_hints():
    """Seen on the user's PC: Groq's per-minute 429 carries an upgrade/billing hint and was taken as fatal."""
    k = llm.classify_http_error(groq_error(429, GROQ_TPM))
    check(k.kind == llm.RATE and abs((k.retry_after or 0) - 3.7275) < 1e-6,
          f"Groq tokens-per-minute 429 with a billing hint is a rate limit ({k.kind}, retry {k.retry_after})")
    gem = gemini_error(429, "You exceeded your current quota, please check your plan and billing details. Quota "
                            "exceeded for metric: generate_content_free_tier_requests, limit: 10. "
                            "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "RESOURCE_EXHAUSTED", retry="33s")
    check(llm.classify_http_error(gem).kind == llm.RATE, "Gemini per-minute 429 saying 'billing details' is a rate limit")
    tpd = groq_error(429, "Rate limit reached for model `x` in organization `org_01testorganisation0000` on tokens per "
                          "day (TPD): Limit 200000, Used 199500. Please try again in 7m12.5s.")
    k = llm.classify_http_error(tpd)
    check(k.kind == llm.QUOTA_DAY and abs((k.retry_after or 0) - 432.5) < 1e-6, "tokens-per-day 429 moves to the next model")
    nof = llm.ProviderHTTPError(429, json.dumps({"error": {"message": "Your credit balance is too low to access the API."}}))
    check(llm.classify_http_error(nof).kind == llm.FATAL, "a real no-credit 429 still stops the provider")
    k = llm.classify_http_error(groq_error(429, "Too many requests. Please try again in 850ms."))
    check(k.kind == llm.RATE and abs((k.retry_after or 0) - 0.85) < 1e-6, "millisecond retry delays are parsed")

    # end to end: the 429 is waited out, the limit is learned and later requests are paced under it
    fake = FakeProvider("groq", ["openai/gpt-oss-120b"], script={"openai/gpt-oss-120b": [groq_error(429, GROQ_TPM)]})
    layer = make_layer("groq", fake)
    now = [1000.0]
    layer.clock = lambda: now[0]

    def fake_sleep(d):
        layer.sleeps.append(d)
        now[0] += d
    layer.sleep = fake_sleep
    texts = [f"Please permanently delete account number {i}." for i in range(200)]
    layer.batch_size = 25
    layer.prefetch_classifications(texts)
    check(layer.enabled and len(layer.batch_results) == 200, f"all 200 texts classified after the 429 ({len(layer.batch_results)})")
    check(layer.tpm_limit.get("openai/gpt-oss-120b") == 8000, "tokens-per-minute limit learned from the message")
    check(layer.rate_waits >= 1, f"later requests are paced under the limit ({layer.rate_waits} waits)")
    blob = json.dumps(layer.summary())
    check("org_01testorganisation0000" not in blob and "org_[REDACTED]" in blob, "organization id is masked in the summary")


def test_network_down():
    fake = FakeProvider("groq", [], network_down=True)
    layer = make_layer("groq", fake)
    check(not layer.preflight(), "network down: preflight fails")
    check("cannot reach" in (layer.disabled_reason or ""), "network down: clear reason")
    check(len(fake.requests) <= 4, f"network down: gives up after a few attempts ({len(fake.requests)} requests)")


def test_batched_classification_and_accounting():
    texts = [f"Please permanently delete account number {i}." for i in range(60)] + \
            ["Please stop the promotional emails. I still need the account and my order history, "
             "so I am not asking for deletion."]
    fake = FakeProvider("gemini", ["gemini-3.5-flash-lite"])
    layer = make_layer("gemini", fake)
    layer.batch_size = 25
    layer.prefetch_classifications(texts)
    posts = generate_calls(fake)
    check(len(posts) == 3, f"61 texts classified in 3 batched requests ({len(posts)})")
    check(posts[0][3]["generationConfig"].get("responseMimeType") == "application/json", "JSON mode requested")
    verdicts = [layer.classify_request(t) for t in texts]
    check(all(v.authorises_deletion for v in verdicts[:60]) and not verdicts[60].authorises_deletion,
          "batched verdicts are applied to the right texts")
    check(layer.agreement["agree"] == 61 and len(generate_calls(fake)) == 3, "no extra calls after prefetch")
    s = layer.summary()
    m = s["models"]["gemini-3.5-flash-lite"]
    check(m["input_tokens"] == 3000 and m["output_tokens"] == 450, "thinking tokens are counted as output")
    check(abs(m["estimated_cost_usd"] - (3000 * 0.30 + 450 * 2.50) / 1e6) < 1e-9, "cost uses the model's price")


def test_disagreement_and_ambiguous_mode():
    class Contrarian(FakeProvider):
        def _reply(self, payload):
            items = json.loads(payload["contents"][0]["parts"][-1]["text"].split("Messages (JSON):", 1)[1])
            return json.dumps([{"id": it["id"], "authorises_deletion": True, "reason": "x"} for it in items])

    fake = Contrarian("gemini", ["gemini-3.5-flash-lite"])
    layer = make_layer("gemini", fake)
    res = layer.classify_request("Please stop the promotional emails. I still need the account and my order "
                                 "history, so I am not asking for deletion.")
    check(res.authorises_deletion is False and layer.agreement["disagree"] == 1, "confident rules win a disagreement")
    fake2 = FakeProvider("gemini", ["gemini-3.5-flash-lite"])
    layer2 = make_layer("gemini", fake2, mode="ambiguous")
    layer2.prefetch_classifications(["Please permanently delete my account."])
    check(len(generate_calls(fake2)) == 0, "ambiguous mode: confident texts need no model call")


def test_provider_shapes():
    for provider, models in (("groq", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]),
                             ("anthropic", ["claude-haiku-5-5", "claude-sonnet-5-5"])):
        fake = FakeProvider(provider, models)
        layer = make_layer(provider, fake)
        check(layer.preflight(), f"{provider}: preflight")
        method, url, headers, body = generate_calls(fake)[-1]
        if provider == "groq":
            check(url.endswith("/openai/v1/chat/completions") and headers["authorization"] == f"Bearer {FAKE_KEY}",
                  "groq: endpoint and bearer header")
            check(layer.text_model == "openai/gpt-oss-120b", "groq: current text model chosen")
        else:
            check(url.endswith("/v1/messages") and headers["x-api-key"] == FAKE_KEY, "anthropic: endpoint and key header")
        layer.prefetch_classifications(["Please permanently delete my account."])
        check(layer.classify_request("Please permanently delete my account.").method == "rules+model",
              f"{provider}: reasoning text stripped and JSON parsed")


def test_screenshot_reader(png):
    fake = FakeProvider("gemini", ["gemini-3.8-flash"], script={"gemini-3.8-flash": []})
    layer = make_layer("gemini", fake)
    text = llm.ScreenshotReader(layer)(png)
    _, _, _, body = generate_calls(fake)[-1]
    check(body["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "image/png", "gemini: inline image part")
    check("account_5762" in text and layer.vision_model == "gemini-3.8-flash", "vision transcript returned")
    fake2 = FakeProvider("gemini", ["gemini-3.8-flash"], script={m: "dead" for m in llm.PROVIDER_CANDIDATES["gemini"]["vision"]})
    layer2 = make_layer("gemini", fake2)
    out = llm.ScreenshotReader(layer2)(png)
    check(isinstance(out, str), "no vision model: falls back to local OCR without raising")


def test_end_to_end_with_fake_provider():
    """Whole engine with a fake Gemini that classifies like the rules and transcribes a screenshot."""
    import glob

    from ghost_audit import loaders
    from ghost_audit.pipeline import AuditEngine

    repo = os.path.dirname(CODE_DIR)
    ds = loaders.load_dataset(os.path.join(repo, "dataset"))
    ds.cases = loaders.load_cases(os.path.join(repo, "dataset", "sample_deletion_requests.csv"))
    fake = FakeProvider("gemini", ["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"],
                        script={"gemini-2.5-flash": "dead"})
    layer = make_layer("gemini", fake)
    cache_dir = os.path.join(repo, ".ghost_audit_cache", "ocr")
    before = set(glob.glob(os.path.join(cache_dir, "*.gemini-vision.txt")))
    try:
        check(layer.preflight(), "end-to-end: preflight")
        layer.prefetch_classifications([c.request_text for c in ds.cases])
        engine = AuditEngine(ds, image_reader=llm.ScreenshotReader(layer), classifier=layer.classify_request)
        results = engine.run()
        baseline = AuditEngine(ds).run()
        check(all(a.row() == b.row() for a, b in zip(results, baseline)),
              "end-to-end: same answers as the deterministic run on the 32 sample cases")
        s = layer.summary()
        check(s["classification_agreement"]["agree"] == len(ds.cases), "end-to-end: model agreed on every request")
        check(s["calls"] <= 1 + 2 + 11, f"end-to-end: {s['calls']} requests (1 preflight, 2 batches, 11 screenshots max)")
    finally:
        for f in set(glob.glob(os.path.join(cache_dir, "*.gemini-vision.txt"))) - before:
            os.remove(f)


def test_cloudflare_block_and_user_agent():
    # the Groq answer seen on a Windows PC: HTTP 403 with Cloudflare's plain-text "error code: 1010"
    fake = FakeProvider("groq", [], list_error=llm.ProviderHTTPError(403, "error code: 1010\n\n"))
    layer = make_layer("groq", fake)
    check(not layer.preflight(), "cloudflare 1010: preflight fails without wasting generate requests")
    reason = layer.disabled_reason or ""
    check("Cloudflare" in reason and "API key" not in reason.split("not the API key")[0],
          "cloudflare 1010: reported as a Cloudflare block, not as a bad key")
    check("\n" not in reason, "error messages are collapsed to one line")
    # the real transport must send an explicit User-Agent (urllib's default is what Cloudflare blocks)
    import urllib.request
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"data": []}'

    original = urllib.request.urlopen
    urllib.request.urlopen = lambda req, timeout=0: (captured.update(headers=dict(req.header_items())), _Resp())[1]
    try:
        llm.http_transport("https://api.groq.com/openai/v1/models", {"authorization": "Bearer x"}, None, method="GET")
    finally:
        urllib.request.urlopen = original
    ua = {k.lower(): v for k, v in captured["headers"].items()}.get("user-agent", "")
    check(ua.startswith("ghost-audit/") and "urllib" not in ua.lower(), f"explicit User-Agent sent ({ua})")


def main() -> int:
    print("== retired model / failover");                 test_retired_model_failover()
    print("== invalid key");                               test_invalid_key_stops_immediately()
    print("== rate limits and daily quota");               test_rate_limit_retry_and_daily_quota()
    print("== rate-limit messages with billing hints");   test_rate_limit_messages_with_billing_hints()
    print("== network down");                              test_network_down()
    print("== cloudflare block and user agent");           test_cloudflare_block_and_user_agent()
    print("== batched classification and accounting");    test_batched_classification_and_accounting()
    print("== rules / model merge");                       test_disagreement_and_ambiguous_mode()
    print("== groq and anthropic shapes");                 test_provider_shapes()
    png = os.path.join(CODE_DIR, "..", "dataset", "evidence", "images", "account_admin_console_20260926.png")
    if os.path.exists(png):
        print("== screenshot reader");                     test_screenshot_reader(png)
    print("== end-to-end with a fake provider");          test_end_to_end_with_fake_provider()
    print(f"all {PASSED} model-layer checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
