# Evaluation report: Ghost Data Still Exists

> **How this report is made.** `python code/evaluation/build_report.py` writes
> this file.
> - **Computed by code:** every number about the current engine (scores,
>   per-case tables, hold-out results, test-suite results, full-run
>   statistics, exposed fields, the final model run and the model comparison),
>   at build time.
> - **Hand-written:** the text around the numbers, in
>   `code/evaluation/report_template.md`. That covers metric definitions, what
>   each test does, the failures history (section 6, quoted from the session
>   log; numbers there describe earlier versions) and the limitations.
> - **Guard:** a number the code cannot compute stops the build.
> - **Check:** `python code/evaluation/build_report.py --check` rebuilds the
>   report and fails if this file differs.
>
> Inputs of this build: engine code e5e33606d30a; `report_template.md` 13122ff0e5ac; `deletion_requests.csv` 78464850941b; `deletion_rules.csv` e12d47ca4cbe; `output.csv` 61e1e9811f24; `usage_run.json` e63e4e920f83; `model_comparison.json` 74c77b87f67d (sha256, first 12 hex digits).

Engine: `code/main.py`. Decisions come from deterministic rules and evidence
parsers. The deletion policy is read from `dataset/deletion_rules.csv` at run
time (15 rules), and the deletion worker's timing is learned
from the participant data at run time (no answer labels). The final full-dataset run added the gemini model layer (`gemini-3.5-flash-lite` for request classification, `gemini-3.8-flash` for screenshots) and produced the answers described in section 5.

## 1. Metrics

Steps are matched on the tuple `(phase, system_id, resource_id, operation, state)`.

| Metric | Definition |
| --- | --- |
| Case status accuracy | share of cases whose `deletion_status` equals the reference |
| Exact case accuracy | share of cases whose status, full ordered step list and every `evidence_id` equal the reference |
| Sequence accuracy | share of cases where the predicted step order equals the reference order and the lists have the same length |
| Step precision / recall / F1 | predicted steps that exist in the reference / reference steps that were predicted |
| Field accuracy | on matched steps, agreement per field (`evidence_id` included) |

Scorer: `code/evaluation/main.py` (`python code/evaluation/main.py --custom`
prints the same metrics).

## 2. Reference cases (`dataset/sample_deletion_requests.csv`, 32 cases)

| Metric | Value |
| --- | --- |
| Case status accuracy | **1.000** (32/32; COMPLETE->COMPLETE 10, INCOMPLETE->INCOMPLETE 22) |
| Exact case accuracy | **0.906** (29/32) |
| Sequence accuracy | **0.906** (29/32) |
| Step precision | **1.000** (190 predicted, 190 correct) |
| Step recall | **0.984** (190 of 193 reference steps) |
| Step F1 | **0.992** |
| Field accuracy (matched steps) | phase 1.000, system_id 1.000, resource_id 1.000, operation 1.000, state 1.000, evidence_id 1.000 |

Per case:

| case | gold status | predicted | gold steps | predicted steps | matched | exact |
| --- | --- | --- | --- | --- | --- | --- |
| case_01 | COMPLETE | COMPLETE | 1 | 1 | 1 | yes |
| case_02 | COMPLETE | COMPLETE | 1 | 1 | 1 | yes |
| case_03 | COMPLETE | COMPLETE | 3 | 3 | 3 | yes |
| case_04 | COMPLETE | COMPLETE | 2 | 1 | 1 | no |
| case_05 | INCOMPLETE | INCOMPLETE | 3 | 3 | 3 | yes |
| case_06 | INCOMPLETE | INCOMPLETE | 4 | 4 | 4 | yes |
| case_07 | INCOMPLETE | INCOMPLETE | 4 | 4 | 4 | yes |
| case_08 | INCOMPLETE | INCOMPLETE | 4 | 4 | 4 | yes |
| case_09 | INCOMPLETE | INCOMPLETE | 5 | 5 | 5 | yes |
| case_10 | INCOMPLETE | INCOMPLETE | 4 | 4 | 4 | yes |
| case_11 | INCOMPLETE | INCOMPLETE | 14 | 14 | 14 | yes |
| case_12 | INCOMPLETE | INCOMPLETE | 5 | 5 | 5 | yes |
| case_13 | INCOMPLETE | INCOMPLETE | 8 | 8 | 8 | yes |
| case_14 | INCOMPLETE | INCOMPLETE | 5 | 5 | 5 | yes |
| case_15 | INCOMPLETE | INCOMPLETE | 5 | 5 | 5 | yes |
| case_16 | INCOMPLETE | INCOMPLETE | 7 | 7 | 7 | yes |
| case_17 | INCOMPLETE | INCOMPLETE | 5 | 5 | 5 | yes |
| case_18 | INCOMPLETE | INCOMPLETE | 14 | 13 | 13 | no |
| case_19 | INCOMPLETE | INCOMPLETE | 8 | 8 | 8 | yes |
| case_20 | INCOMPLETE | INCOMPLETE | 8 | 7 | 7 | no |
| case_21 | COMPLETE | COMPLETE | 3 | 3 | 3 | yes |
| case_22 | INCOMPLETE | INCOMPLETE | 8 | 8 | 8 | yes |
| case_23 | INCOMPLETE | INCOMPLETE | 8 | 8 | 8 | yes |
| case_24 | INCOMPLETE | INCOMPLETE | 7 | 7 | 7 | yes |
| case_25 | COMPLETE | COMPLETE | 3 | 3 | 3 | yes |
| case_26 | COMPLETE | COMPLETE | 6 | 6 | 6 | yes |
| case_27 | COMPLETE | COMPLETE | 6 | 6 | 6 | yes |
| case_28 | COMPLETE | COMPLETE | 6 | 6 | 6 | yes |
| case_29 | INCOMPLETE | INCOMPLETE | 14 | 14 | 14 | yes |
| case_30 | INCOMPLETE | INCOMPLETE | 14 | 14 | 14 | yes |
| case_31 | COMPLETE | COMPLETE | 3 | 3 | 3 | yes |
| case_32 | INCOMPLETE | INCOMPLETE | 5 | 5 | 5 | yes |

The 3 missed reference steps are `account_004` (case_04), `audit_record_6645` (case_18), `audit_record_5569` (case_20). None of these resource ids occurs in any parsed evidence record, activity row or text file under `dataset/` other than `sample_output.csv`, so they cannot be recovered from evidence. Every step the engine produced was correct, including all cited evidence ids.

## 3. Custom evaluation cases (`code/evaluation/custom_cases/`)

Five additional cases were written against the real dataset so that the expected
answer can be verified by hand in the evidence files. The descriptions are
hand-written; the result column is computed.

| case | what it tests | expected | result |
| --- | --- | --- | --- |
| custom_01 | Spanish access request that explicitly keeps the account ("no quiero cerrar la cuenta ni borrar nada") | `NOT_APPLICABLE`, `[]` | pass |
| custom_02 | request sent from a **historical, upper-cased** e-mail (`MINJUN.KIM90@GMAIL.COM`, replaced in `user_audit_log.csv`); the only remaining record is a contact entry keyed by the sha256 hash of that old address | INCOMPLETE; `analytics_9914` remaining (evidence_01), delete, verify account/contact absent and audit record present | pass |
| custom_03 | **temporal cutoff in the other direction**: same person as sample case_27 but audited on 2026-09-18, after the 2026-09-17 snapshot in evidence_02 that shows `account_3866` active again | INCOMPLETE with the account finding cited to evidence_02 and six verification checks | pass |
| custom_04 | **replication drift**: `account_994` exists only in the read-replica table of evidence_05, not in the primary; search entry also present | INCOMPLETE; both records reported remaining, audit record verified present | pass |
| custom_05 | Portuguese request; **tombstone with exposed PII** (`tombstone_8095` still carries a phone number and e-mail hash in evidence_52) | INCOMPLETE; REPORT_EXPOSED_PII + REDACT_RECORD on the tombstone; retained audit record and absent account verified | pass |

| Metric | Value |
| --- | --- |
| Case status accuracy | 1.000 (5/5) |
| Exact case accuracy | 1.000 |
| Sequence accuracy | 1.000 |
| Step precision / recall / F1 | 1.000 / 1.000 / 1.000 (25 steps) |
| Field accuracy | phase 1.000, system_id 1.000, resource_id 1.000, operation 1.000, state 1.000, evidence_id 1.000 |

## 4. Component tests on the full dataset (hold-out style)

The reference answers cover only a few cases. To estimate behaviour on the
rest, `code/evaluation/holdout_tests.py` builds its own ground truth from the
participant data: a request whose account (or worker row) is pinned to the
person by an identifier has a known answer. The test hides those links and
checks whether the run-timing logic recovers them.

It is a **2-fold cross-validation**. The 252 pinned requests are
split by case number. The worker schedule is learned on one half only, and the
other half's links are hidden and recovered with it. The measured default
schedule is graded on exactly the same hidden halves.

| Schedule | Accounts recovered | Wrong account | Unassigned | Hidden worker rows recovered (seed 7) | Missed (no step) | Wrongly attributed |
| --- | --- | --- | --- | --- | --- | --- |
| learned at run time (used for `output.csv`) | 244/252 (96.8%) | 8 | 0 | 164/203 | 37 | 2 |
| measured defaults (`--cadence default`) | 244/252 (96.8%) | 8 | 0 | 165/203 | 34 | 4 |

Worker rows are hidden at random with a fixed seed. Other seeds: seed 11: learned 149/184 recovered, 2 wrong; defaults 150/184, 2 wrong; seed 23: learned 147/188 recovered, 0 wrong; defaults 149/188, 0 wrong.

How the learned schedule's delay window was chosen (history, hand-written):
- Six candidate rules were run through the same 2-fold test.
- They were ranked first by wrongly attributed worker rows, because in an
  audit, reporting another person's record is worse than missing one. Ties were
  broken by accounts and rows recovered.
- The chosen rule is the 5th to 95th percentile of the observed first-step
  delays, widened by two step gaps.
- Disclosure: before the final choice I had also seen that the full-range rule
  changed 2 of the 518 answers. Each of those attributions relied on a
  95 to 102 minute first-step delay, outside the bulk of the observed delays.

Offline test suites (descriptions hand-written, results computed at build time):

**Policy, retained records, exposures and schedule.**
`code/evaluation/test_engine_rules.py`: 46/46 checks passed. It checks:
- the treatments derived from `deletion_rules.csv` equal the table the engine
  used to hardcode;
- editing the rules file changes the plans (analytics -> DELETE turns every
  anonymisation step into a deletion);
- an audit log past its retention period, or a released or expired legal hold,
  loses RETAIN;
- an unrecognised condition, an unsupported action, an unknown category and a
  missing rules file are all reported;
- a wrongly deleted audit record turns a COMPLETE case into INCOMPLETE, with
  the presence check and a warning;
- exposed fields tell current values from historical ones, never list another
  person's identifier, and never write values;
- the learned schedule equals the measured one and gives the same answers;
- system names match the deletion worker's own target systems.

**Model layer.** `code/evaluation/test_llm_adapters.py`: 51/51 checks passed.
It plays the providers with a fake transport, including the failures seen in
real runs:
- a retired model answering 404 (failover to a live model);
- a Cloudflare block (HTTP 403 "error code: 1010", reported as such and not as
  a bad key);
- an invalid key (stops before any generate request);
- per-minute 429s with a retry delay, including Groq's message with an
  upgrade/billing hint and Gemini's "check your plan and billing details";
- tokens-per-day limits (next model), and a real no-credit answer (stop);
- network down;
- batched classification, and token and cost accounting with reasoning tokens;
- the rules-vs-model merge, request shapes for Gemini, Groq and Anthropic, the
  vision path and its OCR fallback;
- tokens-per-minute pacing and masking of organization ids;
- an end-to-end engine run with a fake provider that reproduces the
  deterministic answers on the sample cases.

**Model comparison.** `code/evaluation/test_compare_models.py`:
17/17 checks passed. It runs the comparison against fake Gemini and Groq
providers:
- Groq first answers with the per-minute 429 seen on the real run, and must
  complete.
- Groq stops halfway. It must be reported as incomplete, its screenshots must
  not be claimed as compared, and the conclusion must leave it out.

## 5. Full-dataset run (`output.csv`)

The engine is re-run (rules only, local OCR) at build time and its answers are
compared with the submitted file.

| Item | Value |
| --- | --- |
| submitted `output.csv` | byte-identical to this build's rules-only run; `validate_output.py`: OK: 518 rows, header and ordering valid, all steps well-formed, evidence ids resolve and respect audit_at (1 record-level citations noted) |
| cases | 518 |
| statuses | 356 INCOMPLETE, 152 COMPLETE, 10 NOT_APPLICABLE |
| non-deletion requests | 10 `NOT_APPLICABLE`; 10 of them belong to people whose account the intake flagged as non-deletion; texts the rules were unsure about: case_538 |
| steps | 3,251 (mean 6.3 per case); findings: 384 remaining records, 199 exposed identifiers, 113 active data flows, 26 restore risks |
| evidence | 2,912 observations from 94 of 94 files; 66 files cited |
| policy | 15 rules read from `deletion_rules.csv`, 0 warnings; 423 retained records (416 AUDIT_RECORD, 7 LEGAL_HOLD), 0 missing; 0 lost their retention basis |
| worker schedule | learned from the participant activity (252 pinned requests): one family starts at second 41 and steps every 390 or 449 s; one family starts at second 52 and steps every 1110 or 1169 s; same answers as the measured defaults |
| account linking (508 deletion cases) | 258 worker-run attribution, 238 identifier, 10 no account in the activity stream, 2 account seen only in evidence |

**Final run with the model layer** (from `usage_run.json`; command `python code/main.py --use-llm --llm-classify all --require-llm`; 518 cases, 903.68 s):

| Provider | Model | Role | Calls | Failed calls | Input tokens | Output tokens | Est. cost (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini | `gemini-3.5-flash-lite` | request classification | 16 | 0 | 19,699 | 14,585 | 0.0424 |
| gemini | `gemini-3.8-flash` | screenshot transcription | 12 | 1 | 12,808 | 13,450 | 0.0600 |
| **overall** |  |  | 28 | 1 | 32,507 | 28,035 | 0.1024 |

- status ok; screenshots read by the vision model: 11 of 11, by local OCR: 0, unread: 0
- request classification, rules vs model: agree 517, disagree 0, decided by the model because the rules were unsure 1, not classified 0
- provider errors: transient 1 (each was retried; see `usage_report.md` section 7)

**Which identifiers still expose the person** (`python code/main.py
--exposures exposures.csv`). There is one row per finding with field names
only, never values.

| Finding | Records | Fields still exposed (count of findings) |
| --- | --- | --- |
| exposed PII (record kept, identifiers must go) | 199 | email (current) 166, email (historical) 32, email_hash (current) 31, phone (current) 29, name (current) 4 |
| remaining record (must be deleted) | 384 | email (current) 230, user_id (stable reference) 124, name (current) 121, email (historical) 76, phone (current) 24 |

- 76 remaining-record findings are known only by the key a search returned them for: email (current) 62, email (historical) 14.
- 1 finding(s) do not show their identifier fields in the evidence; the report says so instead of guessing.
- 1 finding(s) carry an email value that is in no identity record; a data-catalog link ties the record to the person.
- 1 finding(s) carry a phone value that is in no identity record; a data-catalog link ties the record to the person.
- Historical identifiers still expose the person in 90 remaining-record and 32 exposed-PII findings.

**Model comparison (Gemini vs Groq).** `python
code/evaluation/compare_models.py` runs every provider with a key over all
requests and screenshots. It writes `model_comparison.md` (and `.json`, which
this report reads), covering:
- calls, tokens, cost and time;
- classification agreement with the rules and between providers;
- screenshot facts against local OCR;
- final answers against the rules-only run.

It needs the API keys, so it runs on the machine that holds them.

Result (from `model_comparison.json`, written by `compare_models.py` on 518 requests, 375 distinct request texts and 11 screenshots):

|  | gemini | groq |
| --- | --- | --- |
| coverage | complete | complete |
| text model | `gemini-3.5-flash-lite` | `openai/gpt-oss-120b` |
| vision model (screenshot sources) | gemini-vision cache (0 fresh, 11 cached, 0 OCR fallback) | `qwen/qwen3.8-27b` (11 fresh, 0 cached, 0 OCR fallback) |
| request texts classified | 375/375, 375 agree with the rules | 375/375, 375 agree with the rules |
| screenshots: same facts as local OCR | 11/11 compared | 11/11 compared |
| answers identical to the rules-only run | 518/518 cases, 3251/3251 steps | 518/518 cases, 3251/3251 steps |
| calls (failed: rate limits waited out) | 16 (0) | 40 (13) |
| tokens in / out (incl. reasoning) | 19,699 / 14,688 | 41,487 / 20,234 |
| est. cost at paid list prices | USD 0.043 | USD 0.024 |
| wall time | 66.5 s | 1045.3 s |

- Texts on which any provider disagrees with the rules or with another provider: 0.
- groq reported tokens-per-minute limits (`openai/gpt-oss-120b` 8,000, `qwen/qwen3.8-27b` 1,000); later requests were paced under them (22 pacing waits).
- This comparison was made by the version that marks incomplete providers (after failure 23).

Hand-written note on the comparison: the screenshot request allows up to
8,192 output tokens. A provider with a small per-minute output limit (Groq's
free tier for its vision model) answers 429 until the window allows it. The
engine waits and paces, so such a run is slower, not wrong.

What this shows, and what it does not: when the vision transcripts and local
OCR lead to the same facts, and independent models agree with the rules, that
is a consistency check, not ground truth. The models and the rules could share
a blind spot, and the answers for the evaluation requests are unknown to us.

## 6. Failures found and improvements made

History, hand-written from the session log. Numbers in this section describe
the engine at the time of each fix; current numbers are in sections 2 to 5.

1. **Pipe-separated and cloud-runtime log dialects parsed to nothing** (two
   files, 76 observations lost). Fixed by normalising the leading `| LEVEL |`
   form and unescaping the quoted `message="k=\"v\""` payload.
2. **OCR lost underscores in identifiers** (`user 17`, `search 1874`), so the
   tombstone screenshot produced no record. Added identifier repair before
   extraction and dark-theme inversion plus 2x upscaling for OCR; all eleven
   screenshots now yield the record they show. Slightly garbled e-mail addresses
   are snapped to known identities within edit distance 2.
3. **Request classifier missed follow-up deletion wording** ("my old profile
   still appears", "please remove records linked to both addresses", "the
   account associated with this address to be erased"). Patterns were
   broadened, and the genuine non-deletion requests are still rejected on
   their explicit "keep / do not delete" statements. Currently:
   10 of the 10 `NOT_APPLICABLE` cases belong to people whose account the privacy intake itself flagged as non-deletion (`CLASSIFY_REQUEST`, 10 accounts flagged).
4. **Accounts with no identifier anywhere** (sample cases 21, 23, 25, 31, 32
   and 258 evaluation cases). First version used time proximity to
   `reported_at`, which fails when several requests share the same
   `reported_at`. Replaced by run assignment: status claim vs worker result,
   elimination of accounts pinned to other people, and the worker's step
   cadence. Hold-out accuracy (then: all anchors hidden at once, no
   cross-validation) went 84.5% -> 90.1% (sequential chain fit) -> 94.4%
   (exact-seconds preference and plausible first-step delay).
5. **Unlabeled worker rows attributed to the wrong request** (11 of 192 in the
   hold-out test of that time). Fixed by requiring a candidate row to sit on
   the cadence between its pinned neighbours in the chain and never claiming a
   row that two runs fit equally.
6. **Evidence ids lost for COMPLETE cases** when the only file showing the
   compliant state was exported after `audit_at` although its entry was earlier
   (the 2026-09-16 family). Added citation fallback: an earlier citable file
   that agrees, then a closed-world snapshot proving absence, then the
   record-level file. Sample case_09 kept its reference citation (evidence_93).
7. **Stale evidence pulled in old resources** (a tombstone screenshot from
   March for a September request, a legal register from May). Resources are
   only included when an observation lies inside `[requested_at, audit_at]`;
   evidence older than the request is a last resort for state only.
8. **A later re-registration leaked into an earlier request** (user_17 has
   `account_6523` in March and `account_4092` in September; the March case
   received both). Evidence-known accounts are now restricted to those observed
   on or before `audit_at`.
9. **Object-storage screenshot produced no record** because the console shows
   file names, not resource ids. Catalog aliases from `LINK_RESOURCE_IDENTITY`
   (`object_19 (delivery_confirmation_19.pdf)`, `export_1842 (NST-2026-0718-1842)`,
   `tombstone_17 (user_17)`) are now resolved when parsing documents.
10. **OCR cache was written inside `dataset/`**; moved to
    `<repo>/.ghost_audit_cache/` so the dataset stays untouched and the cache is
    not packaged.
11. **First run on a Windows PC: every screenshot unreadable.**
    - tesseract was not usable there, so the 11 screenshots yielded nothing.
      8 deletion cases were written as `COMPLETE` with an empty step list
      (rejected by `validate_output.py`), and 3 more lost steps.
    - Worse, the empty OCR result was cached, so installing tesseract
      afterwards would not have helped.

    Fixes:
    - tesseract is searched in the usual install folders and `TESSERACT_CMD`
      (read from `.env` on every run);
    - failed reads are never cached, and empty cache entries are retried;
    - the run prints which screenshots produced no evidence and why;
    - a deletion case with no evidence-backed check is now `INCOMPLETE` with
      the checks that must pass, instead of an empty `COMPLETE`;
    - `python code/main.py --doctor` checks the whole toolchain.
12. **First live model run: 386 requests, 386 failures, 0 tokens, no
    explanation.**
    - The default ids had been retired by the providers: `gemini-2.5-flash`
      answers 404 to keys that never used it, and Groq retired
      `llama-3.3-70b-versatile` and `llama-4-scout`.
    - Every request text was sent separately (375 requests).
    - Errors were swallowed.

    Fixes:
    - model discovery plus ordered candidates with failover;
    - preflight that stops on fatal errors after one request;
    - retry with the provider's delay on per-minute limits;
    - batched classification;
    - provider error messages in the console, in `usage_run.json` and in the
      usage report (key redacted);
    - `--require-llm` to refuse a silent fallback.
13. **Model fallback text cached under the model's name.** When the model
    failed, the OCR text was cached as if the model had produced it, so a later
    run with a working key would never have asked the model. OCR text is now
    cached only under the OCR entry.
14. **Groq rejected a valid key with HTTP 403 "error code: 1010".** This is
    Cloudflare in front of the API blocking Python urllib's default
    User-Agent, not a key problem, but it was reported as "rejected the API
    key". Fixes: an explicit User-Agent on every request, the 10xx codes
    classified and explained as a network-side block, and `--doctor` treating
    only the provider chosen for the run as required.
15. **Stale figures in the hand-typed report.** A recount against the final
    `output.csv` found a wrong step total and an outdated account-linking
    breakdown. This is why the report is now generated by code.
16. **The policy was a copy in code.** `deletion_rules.csv` was loaded but
    never used. The treatments came from a table in `planner.py` that only
    happened to match the rules, so a changed policy would have been ignored.
    Fixes:
    - `ghost_audit/policy.py` reads the rules file of the dataset being audited
      and maps each data category to record kinds (vocabulary only);
    - it checks the conditions the evidence can decide (retention period,
      active legal hold);
    - it reports unknown categories, actions and conditions.
17. **The worker's own action overrode the policy.** A test that edited the
    rules (analytics -> DELETE) found that nothing changed: the worker's
    ANONYMIZE claim won. The worker's action now only tells record sub-types
    apart (attachment or message, contact record or analytics profile), and the
    rules file decides the treatment for that sub-type.
18. **A missing retained record went unnoticed.** If an audit or legal-hold
    record had been wrongly deleted, the engine emitted nothing and could still
    answer COMPLETE. Now the case is INCOMPLETE, keeps the presence check that
    must pass (the contract has no operation for "restore a retained record"),
    and the engine warns about it.
19. **E-mail delivery logs (rule 11) and cache entries (rule 4) were skipped**
    as "unknown kind". They now follow their rules. Currently:
    368 e-mail delivery and 284 cache activity rows exist; 0 such records are linked to any request, so no answer depends on them.
20. **The exposed identifier fields were not named.** `ghost_audit/exposure.py`
    names, for each finding, which of the person's identifiers the record still
    carries and whether each is the current or a historical value (`--explain`
    and `--exposures`).
21. **The worker timing was a set of constants measured on this dataset.**
    `ghost_audit/cadence.py` now learns it on every run from the pinned
    requests. The cross-validation found three learner bugs, all fixed:
    - It picked second 21 instead of 52 on one fold. A step adds 29 to 30 s, so
      the pattern repeats half a minute later. The learner now decides between
      the two candidates by which comes earlier in minutes.
    - A tie-break counted templates instead of rows.
    - The first window rule let one outlier stretch the plausible delay.

    Current result: over three seeds the learned schedule recovered 460 hidden worker rows with 4 wrong attributions, against 464 with 6 for the measured defaults; accounts 244/252 (96.8%) against 244/252 (96.8%).
22. **Any run could overwrite the record of the final run.**
    - `usage_run.json` was rewritten by every `main.py` run. It is now written
      only when the run writes the root `output.csv`.
    - `code/package_submission.py` leaves the file out if it no longer matches
      `usage_report.md`.
    - It refuses to build the zip if a key value or a key-shaped string is
      found in any packaged file.
23. **The first real comparison run stopped Groq at its per-minute limit, and
    the report then over-claimed.**
    - Groq's 429 for "tokens per minute (TPM)" also says "Upgrade to Dev Tier
      ... /settings/billing". The classifier treated any 429 that mentions
      billing as fatal, so Groq was switched off after 3 of 15 batches.
    - The same rule would have stopped Gemini at a per-minute limit, because
      Gemini's message says "check your plan and billing details".
    - The comparison then counted Groq's rules-and-OCR fallback as agreement.
    - The message also contained the account's organization id.

    Fixes:
    - per-minute limits are retried after the provider's delay;
    - a stated tokens-per-minute limit paces later requests;
    - per-day limits move to the next model, and only a real no-credit answer
      is fatal;
    - organization and project ids are masked;
    - the comparison tracks per screenshot whether the model or OCR produced
      the text, marks incomplete providers, leaves them out of its conclusion,
      and keeps the previous report outside the package.
24. **The evaluation report itself was typed by the agent.** The numbers came
    from code runs but were copied by hand, which is how the stale figures of
    failure 15 got in. The report is now rendered by
    `code/evaluation/build_report.py`. Hand-written text lives in
    `report_template.md`. `code/package_submission.py` rebuilds the report
    before packaging, and `--check` detects a hand-edited or stale report.

## 7. Known limitations

* Resources that exist only in the hidden truth (no activity row, no evidence)
  cannot be produced: 3 such steps exist in the reference set
  (section 2).
* Within one worker run, two requests with the same status claim and no
  cadence information remain a coin flip for account attribution. In the
  hold-out, 8 of 252 hidden
  accounts were wrong or left unassigned.
* The closed-world reading of snapshot files (a dump that does not list a
  record proves absence) follows the reference answers, but it is applied only
  when no direct observation of the record exists.
* Screenshot reading depends on local OCR quality (or on a vision model when a
  key is available). Without either, the run still validates, but the
  screenshot-dependent cases cannot be decided and are flagged. There are only
  11 screenshots, so agreement between readers rests on few
  files.
* In the comparison no completed provider (gemini, groq) changed any answer, so the model layer's value here is corroboration and robustness to OCR failure, not extra accuracy, and it makes a run slower. The final model-layer run took 903.68 s.
* Model ids, quotas and prices change; the candidate lists and price table in
  `ghost_audit/llm.py` reflect October 2026 and are overridable from `.env`.
* Some vocabulary is still fixed in code:
  - the resource-id families (`account_`, `search_`, ...) and the mapping from
    a policy data category to the record kinds the parsers produce;
  - a new category in the rules file is reported, but it needs a parser
    mapping before it can be applied;
  - resource-to-system names: confirmed against the deletion worker's own target systems for 9 resource types; the other 5 come from the table in `config.py`.
* The learned schedule assumes the worker runs on a regular cadence. A dataset
  without one would fall back to the measured defaults (with a warning) or
  learn a poor schedule. The hold-out test is the way to tell.
* Retention and legal-hold conditions can only be checked when the evidence
  carries a date or status. In this dataset the retention dates of retained records lie in 2027 to 2034 and 0 record(s) lost their retention basis.
