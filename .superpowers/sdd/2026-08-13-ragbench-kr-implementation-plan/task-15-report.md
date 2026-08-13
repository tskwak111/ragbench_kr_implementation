# Task 15 report — immutable experiment runner framework

## Implementation

- Added a strict, frozen generation experiment configuration. Semantic duplicate detection hashes
  resolved snapshots, exact model and alias, prompt/evaluator versions, runtime policy, question
  cohort, and code commit while excluding the operator label and output location.
- Added run IDs formed from the semantic hash and a UTC timestamp, with one run per semantic hash.
  Full config bytes are independently hash-bound so even non-semantic config mutation is rejected.
- Added immutable per-question response/evidence/usage/error artifacts, atomic state replacement,
  append-only transition history, idempotent partial resume, cache-state preservation, and safe
  planned/running/completed/failed/cancelled/stopped transitions.
- Added exact dry-run plans with known or explicitly unknown cache counts, token maxima, net and
  VAT-gross worst-case cost, experiment budget cap, concurrency, exact model and alias, prompt
  version, destinations, and confirmation hash. Paid authorization requires execution intent, the
  live environment gate, exact plan hash, fresh prices, and both available and experiment budgets.
- Added a concurrency-5 bounded worker pool with per-question isolation. Concurrency 10 is invalid
  without persisted sample/error-window/429 evidence. Budget, batch, schema-error, and provider-error
  stops are explicit; threshold-stop resume requires a diagnosis acknowledgement.
- Added the fixed top-eight retrieval × three prompt × 500 development campaign planner without
  running it, plus calibrated same-cohort top-three selection and statistically competitive
  best-value retention.
- Added a billing reconciliation interface whose console-unavailable state never fabricates a
  provider total, a dry-run-default CLI, and three strict example YAML configurations.

## TDD evidence

Initial focused collection failed because the runner and generation-selection contracts did not
exist. Further RED cycles covered stale failed IDs after successful resume and operator-only fields
incorrectly affecting the semantic hash. Each failed for the intended reason and was followed by a
focused GREEN run.

Final focused result after independent review fixes:

```text
17 passed
```

## Verification

Fresh verification:

```text
ruff check .
All checks passed!

ruff format --check <Task 15 files>
7 files already formatted

mypy src/ragbench scripts/run_experiment.py
Success: no issues found in 57 source files

pytest -m 'not live and not gold' -q
407 passed, 4 skipped

git diff --check
exit 0
```

The repository-wide format check is not used because 37 pre-existing files outside Task 15 are
currently not normalized to this Ruff version; all Task 15 files pass the format check.

## Deliberately pending operations

No provider call, database-backed development run, billing-console read, human calibration, gold
access, or real 12,000-response campaign occurred. The CLI therefore remains fail-closed after all
paid gates in this local environment rather than constructing direct HTTP. Real execution must wire
the existing gateway-backed RAG executor, display a fresh plan, receive explicit approval, and then
record actual provider billing before any quality or winner claim is permitted.

## Independent review

The reviewer found three Important issues and no Critical issues: mismatched resume configs could
misstate cache reuse, a truncated directly-written result could be counted complete, and relative
destinations depended on process CWD. Fixes now bind resume to the exact semantic hash, validate
every existing result and publish it atomically, and canonicalize output paths relative to the YAML
file. Focused regressions cover each case.
