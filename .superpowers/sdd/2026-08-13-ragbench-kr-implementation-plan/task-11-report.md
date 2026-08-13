# Task 11 report — synthetic benchmark generation and automated validation

## Implementation

- Added strict, frozen Pydantic records for candidates, evidence spans, generator metadata, and
  validation status. Answerable candidates require a normalized gold answer and verbatim evidence;
  unanswerable candidates require no answer/evidence and one or more explicit absence assertions.
- Added a deterministic generation planner with the exact 1,500-item target distribution: 300
  fact, 300 numeric/table, 250 comparison, 250 multihop, 200 unanswerable, and 200 complex/summary.
  It applies document/page caps, rejects insufficient capacity, rotates across stable document/page
  windows, and hashes all plan inputs into an immutable plan identity.
- Added bounded generation prompts and `BenchmarkGenerator`. The only provider boundary is the
  existing metered/cached `ProviderGateway.generate`; server code validates every response against
  its assigned job/window and owns plan, batch, source-window, correlation, and cache provenance.
- Added memory and atomic file checkpoint repositories. File checkpoints use canonical JSON,
  content hashes, regular-file checks, and immutable `(plan_hash, batch_id)` identities so a
  completed paid batch resumes without a repeat provider call and corruption fails closed.
- Added controlled unanswerable construction with snapshot-wide absence checks. Added offline
  validation for exact/near duplicate families, exact/normalized/fuzzy evidence, page/chunk
  validity, numeric and answer support, answer leakage, transformed-fact presence, quotas,
  per-document caps, and configurable contamination terms.
- Added a deterministic validation report with rejection counts, cross-reference samples,
  accepted type/difficulty/document distributions, duplicate groups, and completion level. The
  normal reduced-scope completion floor is 1,000; 800–999 is explicitly `emergency_only`, not DOD.
- Added `configs/benchmark.yaml` and a dry-run-by-default `scripts/generate_benchmark.py`. Dry-run
  only loads local windows/config and emits the plan hash. Execution requires `--execute`,
  `--confirm-paid`, exact `--confirm-plan`, live environment enablement, an API key, a pricing
  snapshot no older than 24 hours, and projected cost below remaining settled-plus-reserved budget.
  Actual batches are validated and persisted; an accepted count below 1,000 returns nonzero.

## TDD evidence

The first focused RED run failed during collection because the benchmark package did not exist:

```text
ModuleNotFoundError: No module named 'ragbench.benchmark'
2 errors during collection
```

The cross-process checkpoint/report/CLI cycle was separately observed RED with missing
`FileBatchRepository` and `report_payload`. A self-review RED cycle then proved that model-supplied
correlation metadata was trusted and evidence could escape its assigned source window:

```text
AssertionError: assert None == 'corr-1'
Failed: DID NOT RAISE <class 'ValueError'>
```

The final focused run after server-owned provenance and bounded-window validation:

```text
uv --cache-dir /private/tmp/ragbench-task11-uv-cache run pytest \
  tests/unit/benchmark -q
21 passed in 0.38s
```

## Verification and limitations

Fresh pre-commit verification:

```text
uv --cache-dir /private/tmp/ragbench-task11-uv-cache run pytest \
  -m 'not live and not gold' -q
284 passed, 4 skipped in 2.18s

uv --cache-dir /private/tmp/ragbench-task11-uv-cache run ruff check .
All checks passed!

uv --cache-dir /private/tmp/ragbench-task11-uv-cache run mypy \
  src/ragbench scripts/generate_benchmark.py
Success: no issues found in 45 source files
```

`git diff --check` and a targeted credential-pattern scan were clean. No provider call, paid
action, PostgreSQL call, sealed-gold access, private corpus access, or real benchmark generation
was performed. Consequently, neither the 1,500 target nor the 1,000 normal completion floor is
claimed. Reaching either threshold remains an explicitly approved live data-operation after
fresh pricing and budget review.

## Independent review corrections

The first independent read-only review found no Critical issues and seven Important issues. Its
verdict was not ready: one-shot generation did not replace rejected strata; completion ignored
quota deficits; aggregate windows could misbind a span to the wrong page/chunk; complex support
was too permissive; controlled negatives lacked target-document transformation provenance;
validation settings were absent from artifact identity; and candidate IDs were only batch-local.

Each finding received an adversarial RED test and implementation correction:

- replacement plans now contain only deficient strata and the CLI resumes bounded replacement
  rounds until exact quotas, or until the explicit `--allow-reduced-scope` 1,000-item balanced
  distribution, are met;
- completion and reports expose per-type deficits and require the exact full or approved balanced
  reduced distribution rather than a raw total;
- source windows contain exact page/chunk units and evidence is checked only against its claimed
  unit;
- non-fact answer support fails closed unless all normalized answer content tokens occur in the
  evidence;
- controlled negatives record target document, original fact, transformed fact, and operation,
  then verify original presence, transformed absence, and question linkage in that document;
- validation config and corpus inputs produce a validation-run hash; final artifacts include both
  generation and validation hashes and use conflict-refusing atomic writes;
- candidate IDs are derived server-side from plan hash and job ordinal, while the validator also
  rejects duplicate IDs globally.

Fresh post-review verification raised the focused suite to 24 passing tests and the complete
offline suite to `287 passed, 4 skipped`; Ruff, strict mypy, and `git diff --check` remained clean.
