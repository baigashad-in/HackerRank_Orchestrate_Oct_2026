# Usage report: final full-dataset run

Source: `code/evaluation/usage_run.json` written by `python code/main.py` (requests file `deletion_requests.csv`, 518 cases, 903.68 s wall-clock).

## 1. Model providers, models and calls

| Provider | Model | Role | Calls | Failures | Input tokens | Output tokens | Total tokens | Est. cost (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini | gemini-3.5-flash-lite | request classification | 16 | 0 | 19,699 | 14,585 | 34,284 | 0.0424 |
| gemini | gemini-3.8-flash | screenshot transcription | 12 | 1 | 12,808 | 13,450 | 26,258 | 0.0600 |
| **Overall** | | | **28** | **1** | **32,507** | **28,035** | **60,542** | **0.1024** |

Output tokens include reasoning ("thinking") tokens where the provider reports them, because they are billed as output. Costs are estimates at paid list prices; free-tier usage costs nothing.

## 2. Per request

* model calls per request: 0.054
* input tokens per request: 62.8; output tokens per request: 54.1; total tokens per request: 116.9
* estimated cost per request: USD 0.000198

## 3. Where the evidence text came from

* screenshots: 11 files; read by the vision model: 11; by local OCR: 0; cached from an earlier run: 0; with no usable evidence: 0
* PDFs: local text extraction (pdfplumber / pypdf / pdftotext); logs, JSON and SQL: parsed directly

## 4. Configuration of this run

* provider: `gemini`; status: ok
* text model: `gemini-3.5-flash-lite`; vision model: `gemini-3.8-flash`
* classification mode: `all` (ambiguous = model consulted only when the rules are unsure; all = every request text is classified by both the rules and the model), batch size 25, pacing 4.0 s between requests
* prices used for the estimate (USD per million tokens): gemini-3.5-flash-lite: in 0.3, out 2.5, gemini-3.8-flash: in 0.75, out 3.75

## 5. Model selection

* text: `gemini-3.5-flash-lite` selected
* vision: `gemini-3.8-flash` selected

## 6. Rules vs model agreement on request classification

* agree: 517; disagree (rules kept, logged): 0; decided by rules alone: 0; decided by the model (rules unsure): 1; not classified by the model: 0

## 7. Errors reported by the provider

* counts: transient: 1
* `gemini-3.8-flash` (screenshot backup_restore_console_20260429.png): transient, HTTP 503: This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.

