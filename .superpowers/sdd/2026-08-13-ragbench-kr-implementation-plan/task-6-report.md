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
