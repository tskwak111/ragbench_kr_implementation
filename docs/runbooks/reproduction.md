# RAGBench-KR Reproduction Runbook

This runbook separates public, no-cost verification from database, paid-provider, and sealed-gold
operations. Commands assume the repository root as the current directory.

## Prerequisites

- Python 3.12
- `uv`
- Docker with Compose v2 only for PostgreSQL/API verification
- POSIX filesystem primitives for secure local corpus collection

Do not add credentials to tracked files. `.env` is ignored; `.env.example` intentionally contains
an empty `UPSTAGE_API_KEY`.

## A. Fresh-clone offline fixture (no key, DB, or network after install)

```bash
cp .env.example .env
uv sync --frozen --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ragbench
uv run pytest -m "not live and not gold" -q
uv run ragbench preflight --offline
uv run python scripts/run_retrieval_screen.py --config tests/fixtures/mini-screen.yaml
uv run python scripts/run_experiment.py --config tests/fixtures/mini-experiment.yaml --offline
uv run python scripts/run_experiment.py --config tests/fixtures/mini-experiment.yaml --offline
```

The preflight command truthfully reports unavailable Docker/database components and can exit nonzero
in an offline-only environment; that does not invalidate the subsequent fixture commands. The two
fixture runs use real project BM25/retrieval/generation metric code but deterministic public data,
make zero provider calls, and write content-addressed artifacts under `.ragbench/fixtures/`.
Re-running the experiment must leave one artifact for the same run ID.

The fixture's constructed perfect scores demonstrate wiring only. They must never be quoted as
benchmark results.

## B. PostgreSQL migrations and API

```bash
cp .env.example .env
docker compose up -d db
uv sync --frozen --all-groups
uv run alembic upgrade head
docker compose up --build -d --wait
curl --fail http://localhost:8000/health
docker compose ps
```

`/health` returns application version plus independent application, database, migration, and domain
service states. The default image truthfully remains application-level `degraded` until concrete
domain service bindings are configured; Compose health represents HTTP process liveness only.
The Compose API waits for PostgreSQL, applies Alembic migrations, then starts Uvicorn. PostgreSQL
and API cache data use named volumes. No Upstage key is passed to the container by default.

To stop containers without deleting evidence:

```bash
docker compose down
```

Do not add `--volumes` unless the named database/cache volumes are intentionally disposable.

## C. User-supplied corpus workflow

The following is an operator workflow, not permission to spend. Replace angle-bracket placeholders
with reviewed paths/IDs. Keep source PDFs outside Git and inside a private operator-owned root.

### 1. Collect and freeze provenance

Review every license and source before collection. Run `--help` for the full required metadata set:

```bash
uv run python scripts/collect_corpus.py --help
uv run python scripts/collect_corpus.py \
  --operator-approved \
  --approved-root <PRIVATE_APPROVED_ROOT> \
  --source <PRIVATE_APPROVED_ROOT/report.pdf> \
  --private-output-dir <PRIVATE_OUTPUT_DIR> \
  --raw-dir data/raw \
  --document-id <ID> \
  --title <TITLE> \
  --organization <ORG> \
  --year <YEAR> \
  --document-type report \
  --language ko \
  --sector public \
  --content-stratum mixed \
  --template-family <FAMILY> \
  --source-url <HTTPS_SOURCE> \
  --downloaded-at <ISO8601> \
  --license <LICENSE> \
  --redistribution-status <redistributable-or-nonredistributable> \
  --inclusion-rationale <RATIONALE> \
  --report <PRIVATE_OUTPUT_DIR/report.json> \
  --fragment <PRIVATE_OUTPUT_DIR/fragment.yaml>
```

Validate manifest hashes/page counts and manually inspect the first, middle, and final page of each
PDF before freezing a corpus snapshot.

### 2. Plan paid parsing before execution

```bash
uv run python scripts/parse_corpus.py \
  --manifest configs/corpus.yaml \
  --snapshot-id <CORPUS_SHA256> \
  --mode standard \
  --model-version <VERIFIED_MODEL_VERSION>
```

Save the dry-run plan. Recheck provider console balance and prices. Paid execution is allowed only
after explicit operator approval using the exact printed plan hash:

```bash
RUN_LIVE_UPSTAGE_TESTS=1 uv run python scripts/parse_corpus.py \
  --manifest configs/corpus.yaml \
  --snapshot-id <CORPUS_SHA256> \
  --mode standard \
  --model-version <VERIFIED_MODEL_VERSION> \
  --execute --confirm-plan <EXACT_PLAN_HASH>
```

Repeat for Enhanced only after Standard reaches its completeness/QA gate. Do not put a key on the
command line; load it through the private environment.

### 3. Build deterministic chunks and embeddings

```bash
uv run python scripts/build_chunks.py <COMPLETE_PARSE_CHECKPOINT_JSONL> <PRIVATE_CHUNK_OUTPUT>
uv run python scripts/build_embeddings.py \
  <IMMUTABLE_CHUNK_DATASET> \
  --corpus-snapshot-id <CORPUS_SHA256> \
  --dimension <VERIFIED_DIMENSION>
```

The embedding command above is a plan. Live embedding requires both `--live` and `--confirm-paid`,
plus current price/promotion verification and operator approval. Build all 14 core chunk snapshots
before stretch variants.

### 4. Generate and validate development questions

```bash
uv run python scripts/generate_benchmark.py \
  <SOURCE_WINDOWS_JSONL> \
  --corpus-snapshot-id <CORPUS_SHA256>
```

Inspect rejection distributions and manually validate candidates against original PDFs. Keep normal
runners blind to the sealed gold split. Gold access additionally requires explicit `--execute`,
`ALLOW_GOLD_ACCESS=1`, matching preregistration/config hashes, and the sealed snapshot.

### 5. Plan retrieval and generation experiments

```bash
uv run python scripts/run_retrieval_screen.py \
  --corpus-snapshot-id <CORPUS_SHA256> \
  --question-snapshot-id <DEV_SNAPSHOT> \
  --code-commit <GIT_COMMIT> \
  --snapshot-inventory <IMMUTABLE_INVENTORY_YAML>

uv run python scripts/run_experiment.py --config <RESOLVED_EXPERIMENT_YAML>
```

Paid generation additionally requires `--execute`, exact `--confirm-plan`, an
`--available-budget-usd` below actual balance, live opt-in, a fresh price book, and a concrete
gateway/RAG executor. Never bypass a fail-closed CLI message by calling provider HTTP directly.

### 6. Analyze and publish aggregate-only results

```bash
uv run python scripts/export_results.py \
  --input <CLEAN_ANALYSIS_INPUT_JSON> \
  --output <PUBLIC_EXPORT_DIR> \
  --public-salt <PRIVATE_RANDOM_SALT>
```

Verify output manifests bind corpus, questions, experiments, code, prompts, models, prices, and
metrics. Supply actual console gross cost for reconciliation before publishing a cost claim.

## Required operational gates before research claims

- Frozen, licensed corpus and manual page inspection complete
- Standard/Enhanced parse success, parity, manual QA, and billing reconciliation complete
- All 14 core chunk/index snapshots complete and pgvector parity measured on at least 50 queries
- Synthetic validation, human review, split leakage checks, and sealed-gold metadata complete
- Retrieval screen and development selection follow predeclared rules
- Judge calibration and human agreement reported
- One controlled preregistered gold run complete with paired confidence intervals
- Failure/cost analysis regenerated from immutable evidence

Until all relevant gates are satisfied, label outputs `PENDING_EVIDENCE` and publish no winner.
