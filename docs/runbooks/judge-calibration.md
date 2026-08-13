# Judge calibration runbook

This runbook defines how to calibrate the RAG judge against real human labels. The code and
synthetic tests do **not** constitute calibration evidence. Until a completed report contains
100–300 attested human labels, the judge status is `uncalibrated` and its output cannot be used as
ground truth or as the final authority.

## Safety boundary

- Work only on a frozen development snapshot. Do not load, preview, sample, or score the sealed
  gold snapshot during calibration.
- All judge requests go through `ProviderGateway.generate`; direct HTTP or SDK calls are forbidden.
- A paid run requires the separately approved dry-run plan hash and maximum projected cost. A
  cached fixture or offline parser test is not approval to call a provider.
- Prefer a judge model distinct from the generator. If none is available, record why the same
  model was unavoidable. Record the exact provider model ID, rubric version and SHA-256 hash,
  temperature (zero where supported), response cache state, and correlation ID.
- The judge receives the question, gold answer/evidence, model answer and claims, citations, and
  retrieved evidence. It must not receive a system name, configuration hash, prompt version,
  retriever identity, rank, price, or expected winner.

## Rubric contract

Rubric `judge-v1` has independent fields for correctness on `[0, 1]`, one faithfulness decision
for every material claim, one support decision for every model citation, and a benchmark-defect
flag. Every positive decision names supplied evidence IDs. The parser rejects unknown fields,
missing claim/citation decisions, invented IDs, Markdown fences, and malformed JSON. It never
repairs meaning or invents citations. The rubric's canonical JSON SHA-256 is stored with every
judge record.

Correctness is judged against supplied gold answer/evidence. Faithfulness asks whether each
material claim follows from retrieved context even when the claim is true elsewhere. Citation
support asks whether the cited unit supports the linked claim. Benchmark defect is reserved for a
problem demonstrated by supplied evidence, not judge uncertainty.

## Human sample and labeling

1. Freeze the response pool and verify response IDs are unique.
2. Use `plan_human_calibration` with a fixed seed and a sample size from 100 through 300. The pool
   must cover every observed system × question-type stratum and contain disagreements and known
   failures. Preserve the emitted IDs and stratum counts.
3. Blind reviewers to system/configuration identity and judge scores. Assign real reviewer IDs and
   record the human-attestation flag; automated, model-generated, or synthetic labels are invalid
   outside unit-test fixtures.
4. Have reviewers score the same `[0, 1]` correctness target and the preregistered binary threshold.
   Resolve label-process defects without changing model outputs. Keep original and adjudicated
   labels as separate immutable records.
5. Record exclusions before computing calibration statistics. Do not remove disagreements merely
   to improve correlation.

## Calibration report

Call `calibrate_judge` only with 100–300 unique, real-human-labeled response pairs. Publish:

- sample size and frozen response-list hash;
- Spearman rank correlation for continuous/ordinal scores;
- agreement and F1 at the preregistered binary threshold;
- signed mean judge-minus-human bias for every question type;
- exact judge model ID, rubric version/hash, temperature policy, seed, and snapshot IDs;
- missing strata, exclusions, adjudications, and counterexamples.

The resulting status is `calibrated-assistant-only`: calibration measures behavior but never makes
the judge final authority. Human labels remain authoritative for the final benchmark. If the
sample is absent, incomplete, synthetic, or outside the 100–300 range, report `uncalibrated` and
do not substitute judge scores for missing human scores.

## Paired uncertainty

For final comparisons, align systems by the same unique observation IDs and immutable document
cluster IDs. Use a fixed seed, at least 10,000 resamples, and the document-cluster paired bootstrap
as the primary sensitivity analysis. Report left-minus-right effect, 95% percentile interval,
sample count, cluster count, seed, resample count, and method. Observation-level paired bootstrap
may be reported only as an explicitly labeled non-final sensitivity analysis. Never publish an
interval or significance claim without the underlying paired samples.

## Current project status

Only deterministic metrics, strict judge parsing/execution boundaries, calibration planning and
statistics, and bootstrap implementations are present. No provider judge calls, human calibration,
or gold evaluation have been run by this task.
