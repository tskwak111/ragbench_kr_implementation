# Task 6 report — Standard and Enhanced parsing framework

## Implemented framework

- Added the async `ParserPipeline.parse_corpus(snapshot_id, mode, resume=True)` interface and a
  small functional adapter. Every paid document request is a `ParseRequest` sent through the
  existing `ProviderGateway`; the pipeline does not call Upstage HTTP endpoints directly.
- Added deterministic dry-run plans containing each document/hash/page count, validated cache
  hits, projected new document calls and billable pages, current price-snapshot hash, base price,
  configurable VAT buffer, worst-case cost, settled spend, remaining hard-budget capacity, and an
  immutable plan hash.
- Paid execution fails closed without `--execute`, the exact current `--confirm-plan` hash,
  `RUN_LIVE_UPSTAGE_TESTS=1`, an API key, positive remaining budget, and a price snapshot that
  passes the existing 24-hour freshness check.
- Added per-document success/failure checkpoints keyed by corpus snapshot, source SHA-256, model,
  exact provider version, and parse mode. Resume validates the raw response hash and identity;
  corrupt successes are treated as cache misses and safely retried. Duplicate successful
  invocations are cache hits, while partial failures remain durable and independently retryable.
- Normalization preserves the full raw response and its canonical hash plus Markdown, HTML,
  elements, page mappings, latency, estimated cost, correlation ID, and status/error. It rejects
  unexpected model versions, malformed evidence, and provider billable-page/page-set drift.
- Enhanced planning requires a successful Standard checkpoint for every identical source hash and
  validates the complete Standard source-page set before it can plan Enhanced work.
- Added `SqlAlchemyParseRepository`, an idempotent PostgreSQL upsert, a follow-on Alembic migration,
  an offline SQL migration test, an optional configured-database round-trip test for CI, and a
  deterministic in-memory repository used by offline orchestration tests.
- Added `scripts/parse_corpus.py` for machine-readable dry-run/execute output. It validates the
  manifest-derived snapshot ID and uses the production SQL cache, budget repository, and gateway
  only after every execution gate has passed.

## Tested offline behavior

The Task 6 suites cover plan scope/cost/hash stability, one-page successful persistence,
plan-confirmation rejection, provider/local page reconciliation, partial batch failure, resume,
duplicate invocation, corrupt cache recovery, Standard/Enhanced parity, CLI live gates, and the
offline migration contract.

## Deliberately pending operations

No real Standard or Enhanced batch was executed. The repository still contains an empty draft
corpus manifest, and no live or paid authorization was supplied. Consequently this task does not
claim the planned 95% page-success gate, repaired/excluded effective corpus, 30-pair stratified
manual QA, or provider-console charge reconciliation. Those remain explicit operator steps after a
real corpus is frozen and a fresh dry-run plan is reviewed and exactly confirmed. Enhanced must be
run only after the Standard success/effective-corpus decision is recorded consistently.

## Verification

Fresh verification before commit is recorded in the task handoff. In the local environment the
configured PostgreSQL URL is absent, so database integration tests are expected to skip; schema
generation is still exercised through Alembic's PostgreSQL offline SQL path.

## Fix round 1 — provider adapter, gross budget, and concurrency integrity

- Updated the multipart provider contract to request both HTML and Markdown explicitly and made
  `output_formats` an orchestration-reserved parameter. Normalization now accepts the provider's
  top-level `model`, derives pages from documented element/page fields, accepts legacy
  `model_version`, and uses declared page-count fallback only when elements are absent. A respx
  contract passes a provider-shaped response through the real gateway and both pipeline modes.
- Paid normalization/model/page failures now persist the untouched response, canonical hash,
  correlation ID, resolved model/version, gross estimated cost, page evidence, and an explicit
  `reconciliation_required` status instead of replacing already-billed evidence with nulls/zeros.
- Added the explicit configured `billing_cost_multiplier=1.10`. Gateway reservations,
  settlements, parser checkpoints, and smoke previews use gross enforcement cost; price-book
  rates remain VAT-exclusive and already-settled usage is not multiplied again.
- Parse execution opens each source no-follow, requires a regular file, reads it once, verifies the
  byte hash immediately, and passes those exact verified bytes to `ParseRequest`. Replacement or
  mismatch never reaches the gateway. Size and nanosecond mtime are plan inputs so observable
  replacement invalidates prior confirmation; the execution-time byte hash remains authoritative.
- Resume summaries now separate cached and new successes while reporting whole-corpus successful
  document/page totals. Checkpoint lookup includes exact provider model and version.
- Provider cache entries now carry a canonical payload hash; corrupt memory/SQL envelopes are
  misses. Shared memory repositories coalesce concurrent pipelines. SQL repositories additionally
  support the existing bounded, distinct NullPool lock factory and PostgreSQL advisory locks, with
  the unique checkpoint constraint as the final cross-process guard.

No provider call, paid batch, manual QA, or console reconciliation was performed in this round.

Fix-round verification from the assigned worktree:

```text
ruff check .
All checks passed!

mypy src/ragbench scripts/parse_corpus.py
Success: no issues found in 21 source files

pytest -m 'not live and not gold' -q
127 passed, 3 skipped
```

The three skips are the live provider smoke and two PostgreSQL integration cases because no test
database URL was configured. The complete Alembic upgrade was also generated successfully through
the offline PostgreSQL SQL path. No skipped result is represented as executed evidence.

## Fix round 2 — sparse and blank page reconciliation

- Made the verified manifest/request page count authoritative. A successful checkpoint now has one
  mapping record for every expected page, including explicit `element_count: 0` and empty element
  indexes for blank pages; it never invents text or structural elements for those pages.
- Groups element indexes by their documented page/page-number field and rejects any element or
  provider page metadata outside the expected range. Provider page metadata is retained alongside
  its corresponding complete mapping record.
- Validates any returned `usage.pages`, `usage.billable_pages`, usage page count, or top-level
  billable/page count against the expected count. When elements are absent, an authoritative
  matching count (or an explicit empty element list) supports a consistently all-blank document.
- Paid page/schema drift continues to preserve raw response evidence and a
  `reconciliation_required` checkpoint.

No live calls or real corpus operations were performed in this round.

Fix-round verification: focused parser/resume tests `20 passed`; full offline suite `131 passed,
3 skipped`; Ruff and strict mypy both clean. The skip reasons are unchanged from fix round 1.
