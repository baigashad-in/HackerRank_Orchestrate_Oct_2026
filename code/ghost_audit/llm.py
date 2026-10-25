"""Optional model layer (Gemini, Groq or Anthropic).

The audit engine is deterministic and evidence-driven.  A language model is
useful for two sub-tasks where natural language or pixels have to be read:

* deciding whether free-text (possibly non-English) request text authorises
  deletion, and
* transcribing application screenshots (vision models read them more reliably
  than local OCR).

Both have deterministic fallbacks (multilingual rules and local OCR), so the
layer is strictly optional.  It is enabled with ``--use-llm`` and configured
through environment variables or an uncommitted ``.env`` file:

    GHOST_AUDIT_PROVIDER      gemini | groq | anthropic (default: first provider with a key)
    GEMINI_API_KEY            (or GOOGLE_API_KEY)    enables Gemini
    GROQ_API_KEY                                     enables Groq
    ANTHROPIC_API_KEY                                enables Anthropic
    GHOST_AUDIT_MODEL         text model id (put first in the candidate list)
    GHOST_AUDIT_VISION_MODEL  vision model id (put first in the candidate list)
    GHOST_AUDIT_MIN_INTERVAL  minimum seconds between requests (free-tier pacing)
    GHOST_AUDIT_BATCH_SIZE    request texts per classification call (default 25)
    GHOST_AUDIT_PRICE_IN / GHOST_AUDIT_PRICE_OUT   USD per million tokens for the cost estimate

Robustness, because hosted model ids are retired often (Gemini 2.5 Flash now
returns 404 to keys that never used it; Groq retired Llama 3.3 70B and Llama 4
Scout in 2026):

* the layer lists the models the key can use and tries an ordered list of
  candidates, moving to the next one on "model not found / retired" or on an
  exhausted daily quota;
* per-minute rate limits are honoured (provider retry delay, then backoff);
* authentication, permission and network failures stop the layer after the
  first error instead of failing every request, and the provider's own message
  (with the key redacted) is printed and written to usage_run.json;
* request texts are classified in batches, so the whole dataset needs about
  15 classification calls plus one call per screenshot.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .request_classifier import Classification, classify_request

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
BATCH_CLASSIFY_PROMPT = """You are a privacy-operations analyst. For each customer message below, decide whether it
authorises deletion / erasure / closure of the customer's account and personal data.

Rules:
- Requests for access, an export or copy of data, correction, marketing opt-out, login help, or
  explicit statements that the account must be kept are NOT deletion requests.
- Messages that ask to verify, finish, redo or audit an earlier deletion ARE deletion requests.
- Messages may be in any language.

Return ONLY a JSON array with exactly one object per message, using the same ids:
[{"id": 1, "authorises_deletion": true, "reason": "at most 12 words"}, ...]

Messages (JSON):
"""

SCREENSHOT_PROMPT = ("Transcribe every piece of text visible in this screenshot of an internal admin console, "
                     "line by line, preserving identifiers (user_*, account_*, search_*, analytics_*, support_*, "
                     "export_*, data_flow_*, backup_*, audit_record_*, marketing_*, tombstone_*), e-mail addresses, "
                     "phone numbers, file names and status words exactly as shown. Output plain text only.")

PING_PROMPT = "Reply with the single word OK."

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------
# Ordered candidates. The first one the key can actually use wins; the rest are failovers.
# Flash-Lite first for text (much higher free-tier daily quota), Flash first for screenshots.
PROVIDER_CANDIDATES: Dict[str, Dict[str, List[str]]] = {
    "gemini": {
        "text": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest",
                 "gemini-3.8-flash", "gemini-3.6-flash", "gemini-flash-latest", "gemini-2.5-flash-lite",
                 "gemini-2.5-flash"],
        "vision": ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-flash-latest", "gemini-3.5-flash-lite",
                   "gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-2.5-flash"],
    },
    "groq": {
        "text": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "qwen/qwen3.6-27b",
                 "llama-3.3-70b-versatile"],
        "vision": ["qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "meta-llama/llama-4-scout-17b-16e-instruct",
                   "meta-llama/llama-4-maverick-17b-128e-instruct"],
    },
    "anthropic": {
        "text": ["claude-haiku-5-5", "claude-sonnet-5-5"],
        "vision": ["claude-sonnet-5-5", "claude-haiku-5-5"],
    },
}

# Estimated list prices, USD per million tokens (input, output), matched by model-id prefix.
# Free tiers cost nothing; these are only for the "estimated cost" column. Override with env vars.
PRICE_TABLE: List[Tuple[str, Tuple[float, float]]] = [
    ("gemini-3.8-flash", (0.75, 3.75)), ("gemini-3.6-flash", (0.75, 3.75)), ("gemini-flash-latest", (0.75, 3.75)),
    ("gemini-3.5-flash-lite", (0.30, 2.50)), ("gemini-3.1-flash-lite", (0.25, 1.50)),
    ("gemini-flash-lite-latest", (0.30, 2.50)), ("gemini-2.5-flash-lite", (0.10, 0.40)),
    ("gemini-2.5-flash", (0.30, 2.50)), ("gemini", (0.75, 3.75)),
    ("openai/gpt-oss-120b", (0.15, 0.75)), ("openai/gpt-oss-20b", (0.10, 0.50)), ("qwen/", (0.29, 0.59)),
    ("meta-llama/", (0.11, 0.34)), ("llama", (0.59, 0.79)),
    ("claude-haiku", (1.0, 5.0)), ("claude-sonnet", (3.0, 15.0)), ("claude", (3.0, 15.0)),
]

PROVIDER_KEY_ENV = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}

# Default pacing between requests (seconds): keeps free tiers under their per-minute limits.
DEFAULT_MIN_INTERVAL = {"gemini": 4.0, "groq": 2.5, "anthropic": 0.0}

KEY_PATTERNS = [re.compile(p) for p in (r"AIza[0-9A-Za-z_\-]{20,}", r"gsk_[0-9A-Za-z]{20,}", r"sk-ant-[0-9A-Za-z_\-]{20,}")]
# account identifiers that providers put in error messages (for example Groq's organization id)
ACCOUNT_ID_PATTERNS = [(re.compile(r"\borg_[0-9A-Za-z]{8,}\b"), "org_[REDACTED]"),
                       (re.compile(r"\bproj(ect)?_[0-9A-Za-z]{8,}\b"), "project_[REDACTED]"),
                       (re.compile(r"(projects/)\d{6,}"), r"\1[REDACTED]")]
# 429 wording that really means "no money / account blocked" (stop the provider). Rate-limit messages often
# carry an upgrade or billing hint as well ("Need more tokens? Upgrade ... /settings/billing", "check your plan
# and billing details"), so "billing" alone must not stop anything.
NO_FUNDS_WORDS = ("insufficient", "credit balance", "spend limit", "spending limit", "billing hard limit",
                  "payment required", "account is not active", "account has been suspended", "prepaid", "prepayment")
TPM_RE = re.compile(r"tokens per minute.*?limit\s*(\d+)", re.I)


def price_for(model: str) -> Tuple[float, float]:
    env_in, env_out = os.environ.get("GHOST_AUDIT_PRICE_IN"), os.environ.get("GHOST_AUDIT_PRICE_OUT")
    base = next((p for prefix, p in PRICE_TABLE if model.startswith(prefix)), (0.0, 0.0))
    return (float(env_in) if env_in else base[0], float(env_out) if env_out else base[1])


# ---------------------------------------------------------------------------
# Transport and errors
# ---------------------------------------------------------------------------
class ProviderHTTPError(Exception):
    def __init__(self, status: int, body: str, headers: Optional[Dict[str, str]] = None) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body or ""
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}


class ProviderNetworkError(Exception):
    pass


Transport = Callable[..., Dict]


USER_AGENT = "ghost-audit/1.1 (deletion-audit agent; Python)"


def http_transport(url: str, headers: Dict[str, str], body: Optional[bytes], method: str = "POST") -> Dict:
    """Default transport: urllib, JSON in and out. Raises ProviderHTTPError / ProviderNetworkError.

    An explicit User-Agent is required: Cloudflare in front of some APIs (Groq) answers urllib's default
    "Python-urllib/3.x" with HTTP 403 "error code: 1010".
    """
    headers = dict(headers)
    headers.setdefault("user-agent", USER_AGENT)
    headers.setdefault("accept", "application/json")
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            text = ""
        raise ProviderHTTPError(exc.code, text, dict(exc.headers or {})) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ProviderNetworkError(f"{exc.__class__.__name__}: {reason}") from None


# Error kinds
FATAL = "fatal"                 # auth / permission / billing: stop the provider
MODEL = "model_unavailable"     # this model id cannot be used: try the next candidate
QUOTA_DAY = "quota_exhausted"   # daily quota for this model: try the next candidate
RATE = "rate_limited"           # per-minute limit: wait and retry the same model
TRANSIENT = "transient"         # 5xx: retry a few times
NETWORK = "network"             # host unreachable
BAD_RESPONSE = "bad_response"   # 200 but no usable text (safety block, truncation)


@dataclass
class CallError:
    kind: str
    status: Optional[int]
    message: str
    retry_after: Optional[float] = None


@dataclass
class ModelUsage:
    provider: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    price_in: float = 0.0
    price_out: float = 0.0

    def cost(self) -> float:
        return self.input_tokens / 1e6 * self.price_in + self.output_tokens / 1e6 * self.price_out


@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int


def _parse_retry_after(payload: dict, headers: Dict[str, str], message: str) -> Optional[float]:
    for detail in (payload.get("error") or {}).get("details") or []:
        delay = detail.get("retryDelay") if isinstance(detail, dict) else None
        if isinstance(delay, str):
            m = re.match(r"([\d.]+)s", delay)
            if m:
                return float(m.group(1))
    if "retry-after" in headers:
        try:
            return float(headers["retry-after"])
        except ValueError:
            pass
    m = re.search(r"(?:try again|retry) in (?:(\d+)m(?!s))?([\d.]+)(ms|s)\b", message)
    if m:
        value = float(m.group(2)) / (1000.0 if m.group(3) == "ms" else 1.0)
        return (int(m.group(1) or 0) * 60) + value
    return None


def classify_http_error(exc: ProviderHTTPError) -> CallError:
    try:
        payload = json.loads(exc.body) if exc.body.strip().startswith("{") else {}
    except json.JSONDecodeError:
        payload = {}
    err = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    message = re.sub(r"\s+", " ", str(err.get("message") or exc.body or "")).strip()[:400]
    cf = re.search(r"error code:\s*(10\d\d)", exc.body or "")
    if cf:
        return CallError(FATAL, exc.status, f"blocked by Cloudflare in front of the API (error {cf.group(1)}: "
                                            "request signature or network rejected, not the API key)")
    code = str(err.get("code") or err.get("type") or err.get("status") or "")
    low = (message + " " + code + " " + str(err.get("status") or "")).lower()
    status = exc.status
    retry = _parse_retry_after(payload, exc.headers, message)
    model_words = ("model", "decommission", "deprecat", "no longer available", "not supported", "does not exist",
                   "not found", "unsupported", "image input", "vision", "multimodal")
    if status == 429:
        if re.search(r"per[ _-]?minute|\btpm\b|\brpm\b|per min\b", low):
            return CallError(RATE, status, message, retry)
        if any(w in low for w in ("per day", "perday", "per_day", "daily", "requests per day", "rpd", "tpd")):
            return CallError(QUOTA_DAY, status, message, retry)
        if any(w in low for w in NO_FUNDS_WORDS):
            return CallError(FATAL, status, message, retry)
        return CallError(RATE, status, message, retry)
    if status in (401,):
        return CallError(FATAL, status, message)
    if status == 400 and ("api key" in low or "api_key" in low or "invalid_api_key" in low):
        return CallError(FATAL, status, message)
    if status == 403:
        if any(w in low for w in ("model", "access to")) and "api key" not in low and "has not been used" not in low:
            return CallError(MODEL, status, message)
        return CallError(FATAL, status, message)
    if status in (404, 410):
        return CallError(MODEL, status, message)
    if status in (400, 422) and any(w in low for w in model_words):
        return CallError(MODEL, status, message)
    if status >= 500:
        return CallError(TRANSIENT, status, message, retry)
    return CallError(MODEL, status, message)   # other 4xx: this model/request combination does not work


def _strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.strip()


def _extract_json_array(text: str):
    text = _strip_reasoning(text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Model layer
# ---------------------------------------------------------------------------
@dataclass
class ModelLayer:
    provider: str
    api_key: str
    text_candidates: List[str]
    vision_candidates: List[str]
    classify_mode: str = "ambiguous"          # "ambiguous": model only when rules are unsure; "all": every text
    transport: Transport = http_transport     # swapped for a fake in offline tests
    sleep: Callable[[float], None] = time.sleep
    min_interval: float = 0.0
    batch_size: int = 25
    verbose: bool = True
    usage: Dict[str, ModelUsage] = field(default_factory=dict)
    batch_results: Dict[str, Tuple[bool, str, str]] = field(default_factory=dict)   # text -> (verdict, reason, model)
    agreement: Dict[str, int] = field(default_factory=lambda: {"agree": 0, "disagree": 0, "rules_only": 0,
                                                              "model_only": 0, "unclassified": 0})
    disagreements: List[dict] = field(default_factory=list)
    errors: Counter = field(default_factory=Counter)
    error_log: List[dict] = field(default_factory=list)
    attempts: List[dict] = field(default_factory=list)
    dead_models: Dict[str, str] = field(default_factory=dict)
    text_model: Optional[str] = None
    vision_model: Optional[str] = None
    disabled_reason: Optional[str] = None
    available_models: Optional[List[str]] = None
    _last_call: float = 0.0
    _printed: set = field(default_factory=set)
    clock: Callable[[], float] = time.monotonic
    tpm_limit: Dict[str, int] = field(default_factory=dict)          # model -> tokens-per-minute limit it reported
    recent_tokens: Dict[str, List[Tuple[float, int]]] = field(default_factory=dict)   # model -> (time, tokens)
    rate_waits: int = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, provider: Optional[str] = None, classify_mode: str = "ambiguous",
                 verbose: bool = True) -> Optional["ModelLayer"]:
        provider = (provider or os.environ.get("GHOST_AUDIT_PROVIDER", "")).strip().lower()
        if not provider:
            for name in ("gemini", "groq", "anthropic"):
                if any(os.environ.get(k, "").strip() for k in PROVIDER_KEY_ENV[name]):
                    provider = name
                    break
        if provider not in PROVIDER_CANDIDATES:
            return None
        key = next((os.environ.get(k, "").strip() for k in PROVIDER_KEY_ENV[provider] if os.environ.get(k, "").strip()), "")
        if not key:
            return None
        text = list(PROVIDER_CANDIDATES[provider]["text"])
        vision = list(PROVIDER_CANDIDATES[provider]["vision"])
        if os.environ.get("GHOST_AUDIT_MODEL", "").strip():
            text.insert(0, os.environ["GHOST_AUDIT_MODEL"].strip())
        if os.environ.get("GHOST_AUDIT_VISION_MODEL", "").strip():
            vision.insert(0, os.environ["GHOST_AUDIT_VISION_MODEL"].strip())
        interval = os.environ.get("GHOST_AUDIT_MIN_INTERVAL", "").strip()
        batch = os.environ.get("GHOST_AUDIT_BATCH_SIZE", "").strip()
        return cls(provider=provider, api_key=key, text_candidates=_dedupe(text), vision_candidates=_dedupe(vision),
                   classify_mode=classify_mode,
                   min_interval=float(interval) if interval else DEFAULT_MIN_INTERVAL[provider],
                   batch_size=max(1, int(batch)) if batch else 25, verbose=verbose)

    @property
    def cache_tag(self) -> str:
        return f"{self.provider}-vision"

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None

    # ------------------------------------------------------------------
    def _log(self, msg: str, once_key: Optional[str] = None) -> None:
        if once_key is not None:
            if once_key in self._printed:
                return
            self._printed.add(once_key)
        if self.verbose:
            print(f"[llm] {msg}", file=sys.stderr, flush=True)

    def redact(self, text: str) -> str:
        text = str(text or "")
        if self.api_key:
            text = text.replace(self.api_key, "[REDACTED]")
        for pat in KEY_PATTERNS:
            text = pat.sub("[REDACTED]", text)
        for pat, repl in ACCOUNT_ID_PATTERNS:
            text = pat.sub(repl, text)
        return text

    def _stats(self, model: str) -> ModelUsage:
        if model not in self.usage:
            pin, pout = price_for(model)
            self.usage[model] = ModelUsage(self.provider, model, price_in=pin, price_out=pout)
        return self.usage[model]

    def _pace(self, model: Optional[str] = None) -> None:
        if self.min_interval > 0 and self._last_call:
            wait = self.min_interval - (self.clock() - self._last_call)
            if wait > 0:
                self.sleep(wait)
        if model and model in self.tpm_limit:
            # the provider told us its tokens-per-minute limit: wait until the next request fits in the window
            now = self.clock()
            window = [(t, n) for t, n in self.recent_tokens.get(model, []) if now - t < 60.0]
            self.recent_tokens[model] = window
            if window:
                estimate = max(n for _, n in window)
                used = sum(n for _, n in window)
                if used + estimate > 0.9 * self.tpm_limit[model]:
                    wait = min(60.0, 60.0 - (now - window[0][0]) + 0.5)
                    if wait > 0:
                        self.rate_waits += 1
                        self.sleep(wait)
        self._last_call = self.clock()

    def _disable(self, reason: str) -> None:
        if self.disabled_reason is None:
            self.disabled_reason = self.redact(reason)
            self._log(f"model layer disabled: {self.disabled_reason}. Falling back to rules and local OCR.")

    def _note_error(self, model: str, purpose: str, err: CallError) -> None:
        self.errors[err.kind] += 1
        entry = {"model": model, "purpose": purpose, "kind": err.kind, "status": err.status,
                 "message": self.redact(err.message)[:300]}
        if len(self.error_log) < 50:
            self.error_log.append(entry)
        self._log(f"{model} ({purpose}): {err.kind} HTTP {err.status}: {entry['message'][:220]}",
                  once_key=f"{model}|{err.kind}|{err.status}")

    # ------------------------------------------------------------------
    # One HTTP request (with retries for rate limits / transient errors)
    # ------------------------------------------------------------------
    def _request(self, model: str, prompt: str, image: Optional[Tuple[str, str]], max_tokens: int,
                 json_mode: bool, purpose: str) -> Tuple[Optional[ModelResponse], Optional[CallError]]:
        rate_retries = transient_retries = network_retries = 0
        while True:
            if not self.enabled:
                return None, CallError(FATAL, None, self.disabled_reason or "disabled")
            self._pace(model)
            stats = self._stats(model)
            stats.calls += 1
            try:
                if self.provider == "gemini":
                    resp = self._call_gemini(model, prompt, image, max_tokens, json_mode)
                elif self.provider == "groq":
                    resp = self._call_openai_compatible("https://api.groq.com/openai/v1/chat/completions",
                                                        model, prompt, image, max_tokens)
                else:
                    resp = self._call_anthropic(model, prompt, image, max_tokens)
                if not resp.text.strip() and purpose != "preflight":   # preflight only needs HTTP 200
                    stats.failures += 1
                    err = CallError(BAD_RESPONSE, 200, "empty response (blocked or truncated)")
                    self._note_error(model, purpose, err)
                    return None, err
                return resp, None
            except ProviderHTTPError as exc:
                stats.failures += 1
                err = classify_http_error(exc)
            except ProviderNetworkError as exc:
                stats.failures += 1
                err = CallError(NETWORK, None, str(exc))
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                stats.failures += 1
                err = CallError(BAD_RESPONSE, 200, f"unexpected response shape: {exc!r}")
            self._note_error(model, purpose, err)
            tpm = TPM_RE.search(err.message or "")
            if err.kind == RATE and tpm:
                self.tpm_limit[model] = int(tpm.group(1))
            if err.kind == RATE and rate_retries < 5:
                rate_retries += 1
                delay = min(max(err.retry_after or 0.0, 5.0 * (2 ** (rate_retries - 1))), 90.0)
                self._log(f"rate limited on {model}; waiting {delay:.0f}s (retry {rate_retries}/5)")
                self.sleep(delay)
                continue
            if err.kind == TRANSIENT and transient_retries < 3:
                transient_retries += 1
                self.sleep(min(err.retry_after or 3.0 * transient_retries, 30.0))
                continue
            if err.kind == NETWORK and network_retries < 2:
                network_retries += 1
                self.sleep(3.0 * network_retries)
                continue
            if err.kind == NETWORK:
                self._disable(f"cannot reach the {self.provider} API ({err.message}). Check the internet "
                              "connection, proxy or firewall")
            elif err.kind == FATAL:
                self._disable(f"{self.provider} rejected the request (HTTP {err.status}): {err.message}")
            elif err.kind == RATE:
                err = CallError(QUOTA_DAY, err.status, err.message)   # persistent limiting: treat as exhausted
            return None, err

    def _call_gemini(self, model: str, text: str, image, max_tokens: int, json_mode: bool) -> ModelResponse:
        parts: List[dict] = []
        if image is not None:
            parts.append({"inline_data": {"mime_type": image[0], "data": image[1]}})
        parts.append({"text": text})
        gen: Dict[str, object] = {"maxOutputTokens": max_tokens}
        if not re.match(r"gemini-(?:[3-9]|\d{2})", model) and "latest" not in model:
            gen["temperature"] = 0      # Gemini 3+ models are meant to run at their default temperature
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = json.dumps({"contents": [{"role": "user", "parts": parts}], "generationConfig": gen}).encode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        data = self.transport(url, {"content-type": "application/json", "x-goog-api-key": self.api_key}, body)
        meta = data.get("usageMetadata", {}) or {}
        out_tokens = int(meta.get("candidatesTokenCount", 0)) + int(meta.get("thoughtsTokenCount", 0))
        self._record(model, int(meta.get("promptTokenCount", 0)), out_tokens)
        cands = data.get("candidates") or []
        out = "".join(p.get("text", "") for p in (cands[0].get("content", {}) or {}).get("parts", [])
                      if not p.get("thought")) if cands else ""
        return ModelResponse(out, int(meta.get("promptTokenCount", 0)), out_tokens)

    def _call_openai_compatible(self, url: str, model: str, text: str, image, max_tokens: int) -> ModelResponse:
        if image is not None:
            content: object = [{"type": "text", "text": text},
                               {"type": "image_url", "image_url": {"url": f"data:{image[0]};base64,{image[1]}"}}]
        else:
            content = text
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                           "max_tokens": max_tokens, "temperature": 0}).encode("utf-8")
        data = self.transport(url, {"content-type": "application/json", "authorization": f"Bearer {self.api_key}"}, body)
        usage = data.get("usage", {}) or {}
        self._record(model, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
        out = _strip_reasoning(data["choices"][0]["message"].get("content") or "")
        return ModelResponse(out, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))

    def _call_anthropic(self, model: str, text: str, image, max_tokens: int) -> ModelResponse:
        if image is not None:
            content: object = [{"type": "image", "source": {"type": "base64", "media_type": image[0], "data": image[1]}},
                               {"type": "text", "text": text}]
        else:
            content = text
        body = json.dumps({"model": model, "max_tokens": max_tokens,
                           "messages": [{"role": "user", "content": content}]}).encode("utf-8")
        base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        data = self.transport(f"{base}/v1/messages", {"content-type": "application/json", "x-api-key": self.api_key,
                                                      "anthropic-version": "2023-06-01"}, body)
        usage = data.get("usage", {}) or {}
        self._record(model, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))
        out = "".join(c.get("text", "") for c in data.get("content", []) if c.get("type") == "text")
        return ModelResponse(out, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))

    def _record(self, model: str, tin: int, tout: int) -> None:
        stats = self._stats(model)
        stats.input_tokens += tin
        stats.output_tokens += tout
        self.recent_tokens.setdefault(model, []).append((self.clock(), tin + tout))

    # ------------------------------------------------------------------
    # Model discovery and failover
    # ------------------------------------------------------------------
    def list_models(self) -> Optional[List[str]]:
        """Model ids the key can see (None when the provider does not answer)."""
        try:
            if self.provider == "gemini":
                data = self.transport("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
                                      {"x-goog-api-key": self.api_key}, None, method="GET")
                names = []
                for m in data.get("models", []) or []:
                    methods = m.get("supportedGenerationMethods") or []
                    if "generateContent" in methods or not methods:
                        names.append(str(m.get("name", "")).replace("models/", ""))
                return names
            if self.provider == "groq":
                data = self.transport("https://api.groq.com/openai/v1/models",
                                      {"authorization": f"Bearer {self.api_key}"}, None, method="GET")
                return [str(m.get("id")) for m in data.get("data", []) or [] if m.get("active", True)]
            base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
            data = self.transport(f"{base}/v1/models?limit=1000", {"x-api-key": self.api_key,
                                                                   "anthropic-version": "2023-06-01"}, None, method="GET")
            return [str(m.get("id")) for m in data.get("data", []) or []]
        except ProviderHTTPError as exc:
            err = classify_http_error(exc)
            self._note_error("(list models)", "discovery", err)
            if err.kind == FATAL:
                self._disable(f"{self.provider} refused the request (HTTP {err.status}): {err.message}")
            return None
        except ProviderNetworkError as exc:
            self._note_error("(list models)", "discovery", CallError(NETWORK, None, str(exc)))
            return None
        except Exception:
            return None

    def _ordered_candidates(self, purpose: str) -> List[str]:
        base = self.text_candidates if purpose == "text" else self.vision_candidates
        if not self.available_models:
            return [m for m in base if m not in self.dead_models]
        avail = set(self.available_models)
        listed = [m for m in base if m in avail]
        # newest discovered models of the right family as extra fallbacks
        extra: List[str] = []
        if self.provider == "gemini":
            def ver(name: str):
                m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
                return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)
            fams = [m for m in self.available_models if "flash" in m and not any(
                w in m for w in ("image", "tts", "live", "audio", "embedding", "preview", "exp", "thinking"))]
            if purpose == "text":
                fams.sort(key=lambda n: ("lite" not in n, tuple(-x for x in ver(n))))
            else:
                fams.sort(key=lambda n: ("lite" in n, tuple(-x for x in ver(n))))
            extra = fams
        elif self.provider == "groq":
            if purpose == "vision":
                extra = [m for m in self.available_models if any(w in m.lower() for w in ("vision", "scout", "maverick", "qwen3"))]
            else:
                extra = [m for m in self.available_models if any(w in m.lower() for w in ("gpt-oss", "qwen", "llama"))
                         and not any(w in m.lower() for w in ("guard", "whisper", "tts", "orpheus", "safeguard"))]
        unlisted = [m for m in base if m not in avail]   # listings can omit aliases; keep them last
        return [m for m in _dedupe(listed + extra + unlisted) if m not in self.dead_models]

    def _run_with_failover(self, purpose: str, prompt: str, image, max_tokens: int, json_mode: bool,
                           label: str) -> Tuple[Optional[ModelResponse], Optional[str]]:
        current = self.text_model if purpose == "text" else self.vision_model
        order = ([current] if current and current not in self.dead_models else []) + \
            [m for m in self._ordered_candidates(purpose) if m != current]
        for model in order:
            if not self.enabled:
                return None, None
            resp, err = self._request(model, prompt, image, max_tokens, json_mode, label)
            if resp is not None:
                if purpose == "text" and self.text_model != model:
                    self.text_model = model
                    self.attempts.append({"purpose": purpose, "model": model, "outcome": "selected"})
                    self._log(f"text model: {model}")
                if purpose == "vision" and self.vision_model != model:
                    self.vision_model = model
                    self.attempts.append({"purpose": purpose, "model": model, "outcome": "selected"})
                    self._log(f"vision model: {model}")
                return resp, model
            if err is None:
                return None, None
            if err.kind in (MODEL, QUOTA_DAY):
                self.dead_models[model] = f"{err.kind}: {self.redact(err.message)[:160]}"
                self.attempts.append({"purpose": purpose, "model": model, "outcome": err.kind,
                                      "status": err.status, "message": self.redact(err.message)[:200]})
                if self.text_model == model:
                    self.text_model = None
                if self.vision_model == model:
                    self.vision_model = None
                continue
            return None, None          # bad response for this item, or provider disabled
        if self.enabled and purpose == "text":
            self._disable(f"no usable {purpose} model among {', '.join(order) or 'the candidates'}")
        return None, None

    def preflight(self) -> bool:
        """List models, then make one tiny text request so problems surface before the run."""
        self.available_models = self.list_models()
        if not self.enabled:
            return False
        if self.available_models is not None:
            self._log(f"{self.provider}: key accepted, {len(self.available_models)} models visible")
        resp, model = self._run_with_failover("text", PING_PROMPT, None, 2048, False, "preflight")
        return resp is not None and self.enabled

    # ------------------------------------------------------------------
    # Request classification (batched)
    # ------------------------------------------------------------------
    def prefetch_classifications(self, texts: Sequence[str]) -> None:
        """Classify every text that needs the model, in batches, before the engine asks for them one by one."""
        if not self.enabled:
            return
        wanted = []
        for t in _dedupe(list(texts)):
            if t in self.batch_results:
                continue
            if self.classify_mode != "all" and classify_request(t).confidence >= 0.95:
                continue
            wanted.append(t)
        if not wanted:
            return
        n_batches = (len(wanted) + self.batch_size - 1) // self.batch_size
        self._log(f"classifying {len(wanted)} request texts in {n_batches} batch(es)")
        for b in range(n_batches):
            if not self.enabled:
                break
            chunk = wanted[b * self.batch_size:(b + 1) * self.batch_size]
            items = [{"id": i + 1, "text": t} for i, t in enumerate(chunk)]
            prompt = BATCH_CLASSIFY_PROMPT + json.dumps(items, ensure_ascii=False)
            resp, model = self._run_with_failover("text", prompt, None, 8192, True, f"classify batch {b + 1}/{n_batches}")
            if resp is None:
                continue
            arr = _extract_json_array(resp.text)
            if not isinstance(arr, list):
                self._note_error(model or "?", "classify", CallError(BAD_RESPONSE, 200, "batch answer was not a JSON array"))
                continue
            for obj in arr:
                if not isinstance(obj, dict):
                    continue
                try:
                    idx = int(obj.get("id")) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(chunk) and isinstance(obj.get("authorises_deletion"), bool):
                    self.batch_results[chunk[idx]] = (obj["authorises_deletion"], str(obj.get("reason", ""))[:160],
                                                      model or "")

    def classify_request(self, text: str) -> Classification:
        """Rules first; the model decides ambiguous texts (mode 'ambiguous') or is consulted on every text (mode 'all').

        When both are confident and disagree, the rules win (they were validated against the privacy intake's own
        classifications) and the disagreement is recorded for the reports.
        """
        rules = classify_request(text)
        if self.classify_mode != "all" and rules.confidence >= 0.95:
            self.agreement["rules_only"] += 1
            return rules
        if text not in self.batch_results and self.enabled:
            self.prefetch_classifications([text])
        if text not in self.batch_results:
            self.agreement["unclassified"] += 1
            return rules
        verdict, why, model = self.batch_results[text]
        reason = f"model({model}): {why}"
        if rules.confidence >= 0.95:
            if verdict == rules.authorises_deletion:
                self.agreement["agree"] += 1
                return Classification(rules.authorises_deletion, 0.99, rules.reason + "; " + reason, method="rules+model")
            self.agreement["disagree"] += 1
            if len(self.disagreements) < 50:
                self.disagreements.append({"text": text[:200], "rules": rules.authorises_deletion, "model": verdict,
                                           "model_reason": why})
            return rules
        self.agreement["model_only"] += 1
        return Classification(verdict, 0.9, reason, method="model")

    # ------------------------------------------------------------------
    def read_screenshot_from_model(self, path: str) -> Optional[str]:
        """Vision-model transcript, or None when no model could read it."""
        if not self.enabled:
            return None
        with open(path, "rb") as fh:
            payload = base64.b64encode(fh.read()).decode("ascii")
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        resp, _ = self._run_with_failover("vision", SCREENSHOT_PROMPT, (mime, payload), 8192, False,
                                          f"screenshot {os.path.basename(path)}")
        if resp is not None and resp.text.strip():
            return resp.text
        return None

    def read_screenshot(self, path: str) -> str:
        text = self.read_screenshot_from_model(path)
        if text is not None:
            return text
        from .evidence.images import cached_ocr
        return cached_ocr(path)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        models = {}
        total_calls = total_in = total_out = total_fail = 0
        total_cost = 0.0
        for name, u in self.usage.items():
            models[name] = {"provider": self.provider, "calls": u.calls, "failures": u.failures,
                            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                            "total_tokens": u.input_tokens + u.output_tokens,
                            "price_in_per_mtok": u.price_in, "price_out_per_mtok": u.price_out,
                            "estimated_cost_usd": round(u.cost(), 6)}
            total_calls += u.calls
            total_fail += u.failures
            total_in += u.input_tokens
            total_out += u.output_tokens
            total_cost += u.cost()
        return {"enabled": True, "status": "ok" if self.enabled else f"disabled: {self.disabled_reason}",
                "provider": self.provider, "text_model": self.text_model, "vision_model": self.vision_model,
                "classify_mode": self.classify_mode, "batch_size": self.batch_size, "min_interval_s": self.min_interval,
                "calls": total_calls, "failures": total_fail, "input_tokens": total_in, "output_tokens": total_out,
                "total_tokens": total_in + total_out, "estimated_cost_usd": round(total_cost, 6),
                "classification_agreement": dict(self.agreement), "disagreements": self.disagreements[:20],
                "model_selection": self.attempts[:40], "unusable_models": dict(self.dead_models),
                "error_counts": dict(self.errors), "errors": self.error_log[:20], "models": models,
                "rate_limit_waits": self.rate_waits, "tokens_per_minute_limits": dict(self.tpm_limit)}


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


class ScreenshotReader:
    """Callable wrapper so the screenshot cache can key on the provider."""

    def __init__(self, layer: ModelLayer) -> None:
        self.layer = layer
        self.cache_tag = layer.cache_tag
        self.last_from_model = False
        self.model_reads = 0
        self.ocr_fallbacks = 0

    def __call__(self, path: str) -> str:
        text = self.layer.read_screenshot_from_model(path)
        if text is not None:
            self.last_from_model = True
            self.model_reads += 1
            return text
        self.last_from_model = False           # do not cache OCR text under the model's tag
        self.ocr_fallbacks += 1
        from .evidence.images import cached_ocr
        return cached_ocr(path)
