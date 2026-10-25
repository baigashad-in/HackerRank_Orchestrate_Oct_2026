# Model comparison

Generated 2026-10-25T10:59:03Z by `python code/evaluation/compare_models.py` on 518 requests (375 distinct request texts) and 11 screenshots. Baseline: the rules-only engine with local OCR (4.1 s). Every provider classified every request text (`--llm-classify all`) and, unless noted, read every screenshot with its vision model.

## 1. Calls, tokens, cost and time

| Provider | Coverage | Text model | Vision model | Calls | Failed calls | Input tokens | Output tokens | Est. cost (USD) | Wall time (s) | Screenshots: fresh / cached / OCR fallback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini | complete | gemini-3.5-flash-lite | cached transcripts (gemini-vision) | 16 | 0 | 19,699 | 14,688 | 0.0426 | 66.5 | 0 / 11 / 0 |
| groq | complete | openai/gpt-oss-120b | qwen/qwen3.8-27b | 40 | 13 | 41,487 | 20,234 | 0.0235 | 1045.3 | 11 / 0 / 0 |
| anthropic | skipped: no API key in the environment or .env | | | | | | | | | |

Costs are estimates at paid list prices (free-tier usage costs nothing). Output tokens include reasoning tokens where the provider reports them. Failed calls include rate-limit answers (HTTP 429) that were waited out and retried. Cached screenshots were read by the same provider in an earlier run and cost nothing here.

* groq: tokens-per-minute limit reported by the provider openai/gpt-oss-120b: 8,000, qwen/qwen3.8-27b: 1,000; requests were paced under it (22 pacing waits).

## 2. Request classification

| Provider | Texts classified | Agree with rules | Disagree with rules | Rules unsure (model decided) |
| --- | --- | --- | --- | --- |
| gemini | 375/375 (100.0%) | 375/375 (100.0%) | 0 | 1 |
| groq | 375/375 (100.0%) | 375/375 (100.0%) | 0 | 1 |

Between providers:

* gemini vs groq: agree on 375/375 (100.0%) texts

Texts where any provider disagrees with the rules or with another provider: 0

## 3. Screenshot facts (vision model vs local OCR)

| Screenshot | OCR records | gemini | groq |
| --- | --- | --- | --- |
| evidence_04 account_admin_console_20260926.png | 1 | same facts (cached) | same facts |
| evidence_09 search_index_inspector_20260901.png | 1 | same facts (cached) | same facts |
| evidence_14 connector_management_console_20260807.png | 1 | same facts (cached) | same facts |
| evidence_19 analytics_profile_view_20260713.png | 1 | same facts (cached) | same facts |
| evidence_23 customer_export_console_20260618.png | 1 | same facts (cached) | same facts |
| evidence_28 support_ticket_console_20260524.png | 1 | same facts (cached) | same facts |
| evidence_32 backup_restore_console_20260429.png | 1 | same facts (cached) | same facts |
| evidence_37 security_audit_explorer_20260905.png | 1 | same facts (cached) | same facts |
| evidence_42 deleted_account_reference_20260310.png | 1 | same facts (cached) | same facts |
| evidence_46 object_storage_browser_20260213.png | 1 | same facts (cached) | same facts |
| evidence_60 marketing_profile_console.png | 1 | same facts (cached) | same facts |

"Same facts" means the records, presence and identifier exposure extracted from the model's transcript equal those from local OCR. "(cached)" marks a transcript the same provider made in an earlier run.

* gemini: 11/11 screenshots read by the model give the same facts as local OCR
* groq: 11/11 screenshots read by the model give the same facts as local OCR

## 4. Final answers (`output.csv` rows) vs the rules-only run

| Provider | Cases identical | Status changes | Steps matched |
| --- | --- | --- | --- |
| gemini | 518/518 (100.0%) | 0 | 3251/3251 (100.0%) |
| groq | 518/518 (100.0%) | 0 | 3251/3251 (100.0%) |

## 5. Reading the comparison

Every provider that completed (gemini, groq) reproduces the rules-only answers on all 518 cases, so the models act as an independent check rather than a source of extra accuracy on this dataset. Lowest estimated cost: groq; fastest: gemini. A disagreement table above (if any) shows where the models and the rules read a request differently.
