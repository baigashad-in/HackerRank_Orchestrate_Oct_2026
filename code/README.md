# Ghost Data Still Exists: deletion-audit agent

Solution for HackerRank Orchestrate (October 2026). For every request in
`dataset/deletion_requests.csv` the agent decides whether deletion was actually
requested, reconstructs the person's current and historical identifiers, reads
all 94 evidence files (logs in six dialects, JSON exports and snapshot bundles,
SQLite SQL snapshots with replicas, a text handover, PDFs and screenshots),
determines the state of every relevant record at `audit_at`, applies
`deletion_rules.csv`, and writes the ordered findings, remediation and
verification steps to `output.csv`.

## Setup

* Python 3.10+ (tested on 3.13). No third-party package is required for the
  core engine (CSV, JSON, SQLite, regex are all standard library).
* Optional, for the four PDF documents: `pdfplumber` or `pypdf`, or the
  `pdftotext` CLI (poppler). Any one of them is enough.
* Optional, for the eleven screenshots: `tesseract` with `pytesseract` and
  `Pillow`. Without OCR the screenshots are skipped and the engine still runs.
* Optional model layer: put `GEMINI_API_KEY` (or `GROQ_API_KEY` /
  `ANTHROPIC_API_KEY`) in the environment or in an uncommitted `.env` file
  (copy `.env.example`) and run with `--use-llm`. See "Model layer" below.

```bash
pip install -r code/requirements.txt        # optional helpers: pdfplumber, pytesseract, pillow
```

On Windows, tesseract is usually not on `PATH`: install it (for example the UB
Mannheim build). The engine looks in the usual install folders by itself; if it
still cannot find it, put `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`
in `.env`. Without OCR (and without a vision model) the eleven screenshots
cannot be read: the run still completes and validates, but it prints a warning
and the cases that depend on screenshots are marked INCOMPLETE with the checks
that must pass, so do not submit such a run. `python code/main.py --doctor`
checks all of this in one go.

## Run

From the repository root:

```bash
python3 code/main.py                 # reads dataset/, writes output.csv at the repository root
python3 code/validate_output.py      # checks output.csv against the submission contract
python3 code/evaluation/main.py --custom   # scores the 32 sample cases and the custom cases
python3 code/main.py --explain case_11     # prints the evidence trail, policy rule and exposed fields for one case
python3 code/main.py --exposures exposures.csv   # also lists, per finding, which identifier fields still expose the person
```

Checks and tests (no keys needed):

```bash
python3 code/evaluation/test_engine_rules.py      # 46 checks: policy file, retention and holds, retained records, exposures, schedule
python3 code/evaluation/test_llm_adapters.py      # 51 checks of the model layer with fake providers
python3 code/evaluation/test_compare_models.py    # 17 checks of the model comparison with fake providers
python3 code/evaluation/holdout_tests.py          # 2-fold hold-out: account and worker-row attribution
python3 code/evaluation/build_report.py           # writes evaluation_report.md: every number computed by code
python3 code/evaluation/build_report.py --check   # fails if evaluation_report.md is stale or was edited by hand
```

All three reports come from code:
- `evaluation_report.md` from `build_report.py`, with hand-written text only from `report_template.md`;
- `usage_report.md` from `usage_report.py` (from `usage_run.json` of the final run);
- `model_comparison.md` from `compare_models.py`.

`package_submission.py` rebuilds the evaluation report before it builds the zip.

`main.py` also accepts `--requests <csv>` and `--output <csv>` to audit another
request file (this is how the sample and custom evaluations run). Only a run
that writes the root `output.csv` records `code/evaluation/usage_run.json` (the
source of the usage report), so test runs cannot overwrite the final run's
record. `--cadence default` uses the measured worker schedule instead of the
one learned at run time (the answers are the same on this dataset). Screenshot
OCR is cached under `<repo>/.ghost_audit_cache/`; the first run takes about
25 s, later runs under one second.

## How it works

```
dataset/*.csv  ──▶ loaders ──▶ IdentityIndex (current + historical emails, phones, sha256 hashes)
evidence/*     ──▶ evidence parsers ──▶ Observations (file, time, system, resource, presence, identifiers, flags)
system_activity.csv ──▶ ActivityIndex (worker runs, catalog links, intake classifications)
                                   │
request_text ──▶ request_classifier (multilingual rules, optional model) ──▶ NOT_APPLICABLE or continue
                                   │
            Linker: resource ↔ person (identifiers, catalog links, worker-run cadence, elimination)
                                   │
            state.reconstruct: latest eligible observation per resource at audit_at (+ closed-world snapshots)
                                   │
            policy: deletion_rules.csv read at run time (category ⇒ record kinds, conditions checked)
                                   │
            planner: treatment ⇒ findings / remediation / verification, ordered; exposure: fields still exposed
                                   │
                              output.csv
```

Package layout (`code/ghost_audit/`):

| Module | Role |
| --- | --- |
| `loaders.py` | dataclasses and CSV readers for the participant-facing files |
| `identity.py` | identity index: current snapshot plus `user_audit_log.csv`; sha256 of every known e-mail |
| `request_classifier.py` | deletion vs non-deletion rules for en/es/pt/fr/de/it/nl/ja/ko/zh; explicit "keep my account" statements win |
| `evidence/logs.py` | logfmt, JSON lines, bracketed, pipe-separated, syslog and cloud-runtime dialects; trace-id joins of `*_started` / `*_completed` lines |
| `evidence/jsonrecords.py` | service resource exports, records-assurance snapshot bundles, record-lookup exports, collection dumps |
| `evidence/sqlsnap.py` | executes each SQL file in SQLite; primary and replica tables are compared (a copy in any table means the record exists) |
| `evidence/textrecords.py`, `pdfs.py`, `images.py`, `docfacts.py` | handover notes, PDF text, screenshot OCR; documents are reduced to the record they are about plus the identifiers that belong to it |
| `activity.py` | indexes `system_activity.csv`: worker rows, `LINK_RESOURCE_IDENTITY` catalog links and aliases, `CLASSIFY_REQUEST` rows |
| `linking.py` | resource ↔ person linking, including the deletion worker's run cadence |
| `cadence.py` | learns the worker's schedule (step families, gaps, first-step seconds and delays) from requests pinned by identifiers |
| `policy.py` | reads `deletion_rules.csv`: data category ⇒ record kinds, action ⇒ treatment, conditions (retention period, active legal hold) checked against the evidence |
| `exposure.py` | names the identifier fields a record still exposes (current or historical), never their values |
| `state.py` | state at `audit_at` and the evidence to cite |
| `planner.py` | step generation and ordering; a retained record that is missing blocks COMPLETE |
| `pipeline.py` | orchestration, resource universe per case, chain attribution |
| `llm.py` | optional model layer with call and token accounting |

### Decisions worth knowing

* **Claims vs evidence.** `reported_status` and `reported_result` are never used
  as proof of state. They are used only as *linking* signals (which worker run
  belongs to which request) and the `CHECK_*` rows as proof that a flow or
  restore path is still open when no file says otherwise.
* **Time.** A record-level timestamp (log line time, inner snapshot
  `captured_at`, row `updated_at`) decides whether an observation may be used;
  the latest usable observation wins. For citations the engine prefers files
  whose manifest `captured_at` is on or before `audit_at`. Evidence older than
  `requested_at` describes the pre-deletion world and is only a last resort.
* **Closed-world snapshots.** A table or collection dump that does not list a
  record proves the record is absent at that time (this is how the sample
  answers cite `customer_accounts_failover_replica_20260902.sql` for accounts
  that are not in it).
* **Treatment.** Every treatment comes from a row of `deletion_rules.csv`; the
  engine carries no copy of the policy. The record kind comes from the evidence;
  where the evidence cannot tell sub-types apart, the workflow's own action does
  (`DELETE_RECORD` on a support record marks an attachment, `ANONYMIZE_RECORD`
  an analytics profile), and the rules file then decides that sub-type's
  treatment. Tombstones keep the stable `user_*` reference; any name, e-mail,
  phone or hash on them is exposed PII. Audit records and legal holds are
  retained and verified present, unless their retention period ended or the
  hold was released; a retained record that is missing makes the case
  INCOMPLETE with the presence check and a warning. Backups are retained but
  their restore path must be blocked. A changed or broken rules file (unknown
  category, action or condition) is reported, never ignored.
* **Linking without identifiers.** Anonymised analytics profiles and 404
  account lookups carry no identifier. The worker processes a request as one
  run on a regular schedule. The schedule is learned on every run from the 252
  requests whose account an identifier pins down (no answers used). On this
  dataset it comes out as: the account step, then policy actions every 6:30 or
  7:29 starting N min 41 s later, and results and checks every 18:30 or 19:29
  starting N min 52 s later. The engine assigns unlabeled rows to the run whose
  schedule they fit, after eliminating resources already pinned to other
  people, and never attributes a row that fits two runs equally well. A 2-fold
  hold-out (`holdout_tests.py`) recovers 96.8% of hidden accounts.
* **Determinism.** No randomness anywhere; the same input always yields the same
  `output.csv`.

### Model layer (Gemini, Groq or Anthropic)

The engine is deterministic by design. `--use-llm` adds a model for the two
sub-tasks where language or pixels have to be read:

* **request classification**: the multilingual rules run first; in the default
  mode (`--llm-classify ambiguous`) the model is consulted only when the rules
  are unsure, in `--llm-classify all` (alias `--llm-shadow`) every request is
  classified by both and the agreement is recorded. When both are confident and
  disagree, the rules win (they were validated against the privacy intake's own
  `CLASSIFY_REQUEST` rows) and the disagreement is logged for review;
* **screenshot transcription** with a vision model instead of local OCR
  (`--no-llm-vision` keeps OCR). Transcripts are cached per provider and model.

Keys come from the environment, or from an **uncommitted `.env` file** in the
repository root (next to `AGENTS.md`; copy `.env.example`). Real environment
variables always win over `.env`; values are never printed, logged or packaged,
and `.gitignore` plus the packaging command below exclude the file:

| Provider | Key variable | Text model candidates (first usable wins) | Vision model candidates |
| --- | --- | --- | --- |
| `gemini` (recommended) | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-flash-lite-latest`, then Flash models | `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-flash-latest`, then Flash-Lite |
| `groq` | `GROQ_API_KEY` | `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` | `qwen/qwen3.8-27b` (else local OCR) |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5`, `claude-sonnet-5-5` | `claude-sonnet-5-5`, `claude-haiku-5-5` |

Hosted model ids are retired often (in 2026 Gemini 2.5 Flash started returning
404 to keys that had never used it, and Groq retired Llama 3.3 70B and Llama 4
Scout), so the layer does not trust a fixed id:

* **preflight**: it lists the models the key can see and sends one tiny request
  before the audit; an invalid key, missing permission or unreachable host stops
  the layer after that first error, with the provider's message (key redacted),
  instead of failing every request;
* **failover**: a model that answers "not found / retired" or whose daily quota
  is exhausted is skipped for the next candidate (newest listed Flash models are
  added automatically for Gemini);
* **rate limits**: per-minute 429s wait for the provider's retry delay and retry;
  requests are paced (`GHOST_AUDIT_MIN_INTERVAL`, default 4 s for Gemini, 2.5 s
  for Groq) to stay under free-tier limits;
* **batching**: request texts are classified 25 per request
  (`GHOST_AUDIT_BATCH_SIZE`), so the full dataset needs about 27 requests
  (1 preflight, 15 classification batches, 11 screenshots). This matters on the
  Gemini free tier, where Flash models are reported to allow only about 20
  requests per day and Flash-Lite about 500;
* **fallback**: if the layer stops, the run continues with rules and local OCR
  and the usage report says so in a warning at the top; `--require-llm` makes
  the run stop with exit code 2 instead.

The measured numbers are not repeated here, because they must come from code.
The final full-dataset run (calls, tokens, cost, errors) is in
`code/evaluation/usage_report.md`. The Gemini vs Groq comparison is in
`code/evaluation/model_comparison.md`. Both are summarised in section 5 of
`evaluation_report.md`.

`GHOST_AUDIT_PROVIDER` picks the provider when several keys are present
(`--provider` wins); `GHOST_AUDIT_MODEL` / `GHOST_AUDIT_VISION_MODEL` put a
specific id first in the candidate list; `GHOST_AUDIT_PRICE_IN` /
`GHOST_AUDIT_PRICE_OUT` (USD per million tokens) override the price estimates.

```bash
cp .env.example .env                              # then fill in GEMINI_API_KEY (Windows: copy .env.example .env)
python code/main.py --doctor                      # checks Python, dataset, PDF/OCR, keys and models; costs <= 2 requests
python code/main.py --use-llm --llm-classify all --require-llm
python code/validate_output.py
python code/evaluation/usage_report.py --write    # usage_report.md from this run's usage_run.json
python code/evaluation/test_llm_adapters.py       # 51 offline checks of the model layer (no keys needed)
python code/evaluation/compare_models.py          # Gemini vs Groq (every provider with a key): writes model_comparison.md
python code/package_submission.py                 # builds code.zip (or code_vN.zip); refuses if a key would be packaged
```

`compare_models.py` classifies every request and reads every screenshot with
each provider, runs the full audit with each, and compares them with the
rules-only run and with each other. It reports calls, tokens, cost and time,
classification agreement, screenshot facts against OCR, and the final answers.
Cached screenshot transcripts are reused (add `--fresh-vision` to re-read). It
never writes `output.csv` or `usage_run.json`.

```bash
# packaging by hand, if you prefer (.env must stay out):
zip -r code.zip README.md README_SOLUTION.md .env.example code -x "*/__pycache__/*" ".env" "*/.env"
```

Every request is counted per model and written to
`code/evaluation/usage_run.json`: calls, failures, input/output tokens
(reasoning tokens included, since they are billed), estimated cost, which
models were selected or skipped and why, rules-vs-model agreement, and where
each screenshot's text came from (vision model, OCR or cache).

### Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `N of 11 screenshots produced no usable evidence (... tesseract ...)` and cases marked INCOMPLETE "with the checks that must pass" | tesseract not installed or not found | install tesseract; on Windows add `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` to `.env`; run `--doctor` |
| usage report shows calls but 0 tokens | the model layer failed; the reason is printed and listed in section 7 of the report | `python code/main.py --doctor` |
| Groq `HTTP 429 ... tokens per minute (TPM): Limit 8000` | the free tier allows 8,000 tokens per minute for `openai/gpt-oss-120b`; the layer waits the stated delay, learns the limit and paces later requests, so the run only gets slower | nothing; if it persists for minutes, `GHOST_AUDIT_BATCH_SIZE=10` in `.env` makes each request smaller |
| `model_unavailable HTTP 404 ... no longer available` | retired model id | nothing to do: the next candidate is used; set `GHOST_AUDIT_MODEL` to force one |
| `quota_exhausted HTTP 429` | free-tier daily quota | wait for the daily reset, use `--llm-classify ambiguous`, or another provider |
| `fatal HTTP 400 API key not valid` | wrong or revoked key | create a new key; check `.env` is not saved as `.env.txt` |
| `cannot reach the ... API` | no internet, proxy or firewall | check the connection; behind a proxy set `HTTPS_PROXY` |
| `blocked by Cloudflare ... error 1010` (seen with Groq) | Cloudflare rejected the request signature; older versions sent urllib's default User-Agent | fixed (explicit User-Agent); if it persists, a VPN or network filter is the likely cause |

The engine never caches a failed OCR or model read, so after fixing a problem a
plain re-run picks it up.

## Evaluation

`code/evaluation/main.py` scores predictions against reference answers: case
status accuracy, exact-case accuracy, sequence accuracy, step precision /
recall / F1 and per-field accuracy. `code/evaluation/custom_cases/` holds five
additional hand-verified cases. Results and the failure log are in
`code/evaluation/evaluation_report.md`.
