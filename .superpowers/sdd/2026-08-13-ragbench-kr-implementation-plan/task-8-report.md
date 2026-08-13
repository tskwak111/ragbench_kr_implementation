# Task 8 report — Embed 2 indexing, cosine reference, and pgvector parity framework

## Implementation

- Added NumPy as a pinned project dependency and implemented `cosine_top_k(query, matrix, k,
  chunk_ids)`. It validates one-dimensional queries, two-dimensional matrices, matching dimensions,
  finite inputs, non-zero vectors, chunk-ID cardinality, and positive K. It L2-normalizes safely,
  returns Python float scores, clamps `k > n`, and ranks by descending cosine score followed by
  ascending chunk ID so ties do not depend on input order.
- Added the shared `SearchFilter`, `SearchHit`, and `Retriever` protocol contracts plus
  `DenseRetriever`. Query text is embedded through the snapshot's query-model path, and the SQL
  repository validates the exact corpus, parse, chunk-strategy, and embedding-snapshot identity
  before retrieval.
- Extended provider embedding responses with an optional resolved model ID. The guarded Upstage
  gateway propagates the response model when supplied and otherwise records the exact requested
  model. Provider parameters already participate in gateway cache keys, so an explicitly supported
  `input_type=query` or `input_type=document` cannot alias in cache.
- Added an `EmbeddingService` with exact item/token batch caps, stable request and persistence
  order, partial-snapshot resume, per-batch persistence, and finalization only after the expected
  chunk count exists. It rejects response count, model, dimension, non-finite value, and zero-vector
  mismatches before persistence and stores L2-normalized vectors.
- Added immutable embedding-snapshot metadata for corpus, parse, strategy, document/query model
  IDs, dimension, normalization, expected count, creation time, completion state, and index state.
  A separate `chunk_embedding` table carries versioned vectors without overloading the original
  chunk record or requiring chunk artifact IDs to be UUIDs.
- Added dimension consistency through a composite `(embedding_snapshot_id, dimension)` foreign key
  and `vector_dims(embedding) = dimension`. Snapshot finalization validates the complete row count,
  creates a partial expression HNSW index for that snapshot and dimension, records the generated
  index name/state, and only then marks the snapshot complete. Dimensions 1–2000 use
  `vector(N)/vector_cosine_ops`; 2001–4000 use `halfvec(N)/halfvec_cosine_ops`. Dimensions and index
  identifiers are generated from validated integers and UUIDs rather than caller-provided SQL.
- Added `scripts/build_embeddings.py`. It validates one immutable JSONL chunk dataset, derives a
  deterministic plan hash and UUID snapshot ID, prints a dry-run by default, and requires both
  `--live` and `--confirm-paid` for execution. Live execution additionally requires a key, a fresh
  unambiguous price snapshot, the global budget guard, the shared SQL response cache, and the
  guarded gateway. No direct provider client path exists in the builder.
- Added `notebooks/02_cosine_from_scratch.ipynb` with cosine equations, normalization and Top-K
  explanation, deterministic tie behavior, pgvector score conversion, and an output-free offline
  NumPy example.
- Added a configured-database parity integration test with multiple queries, explicit ties,
  identical ordered chunk-ID expectations, and absolute score tolerance `1e-5`.

## TDD evidence

The implementation followed observable RED/GREEN cycles.

1. Cosine reference RED:

   ```text
   uv --cache-dir /private/tmp/finproof-uv-cache run pytest \
     tests/unit/retrieval/test_cosine.py -q

   ERROR tests/unit/retrieval/test_cosine.py
   ModuleNotFoundError: No module named 'ragbench.retrieval'
   ```

   GREEN after adding the minimal retrieval contracts/reference:

   ```text
   ...........                                                              [100%]
   11 passed in 0.03s
   ```

2. Embedding service RED:

   ```text
   uv --cache-dir /private/tmp/finproof-uv-cache run pytest \
     tests/unit/embeddings/test_service.py -q

   ERROR tests/unit/embeddings/test_service.py
   ModuleNotFoundError: No module named 'ragbench.embeddings'
   ```

   GREEN together with the existing gateway contracts:

   ```text
   uv --cache-dir /private/tmp/finproof-uv-cache run pytest \
     tests/unit/embeddings/test_service.py \
     tests/unit/retrieval/test_cosine.py \
     tests/contract/providers/test_upstage_client.py -q

   ................................................                         [100%]
   48 passed in 0.29s
   ```

3. SQL repository and dense retriever RED:

   ```text
   uv --cache-dir /private/tmp/finproof-uv-cache run pytest \
     tests/unit/embeddings/test_repository.py \
     tests/unit/retrieval/test_dense.py -q

   ImportError: cannot import name 'hnsw_index_spec'
   ImportError: cannot import name 'DenseRetriever'
   ```

   GREEN after the SQL/index/retriever implementation:

   ```text
   uv --cache-dir /private/tmp/finproof-uv-cache run pytest \
     tests/unit/embeddings tests/unit/retrieval \
     tests/integration/retrieval/test_pgvector.py -q

   ..............................s                                          [100%]
   30 passed, 1 skipped in 0.16s
   ```

4. Builder RED:

   ```text
   uv --cache-dir /private/tmp/finproof-uv-cache run pytest \
     tests/unit/test_build_embeddings.py -q

   ERROR tests/unit/test_build_embeddings.py
   ModuleNotFoundError: No module named 'scripts'
   ```

   The test imports executable scripts by explicit file path, matching the project's existing
   script-test convention. GREEN after the builder implementation:

   ```text
   ...                                                                      [100%]
   3 passed in 0.19s
   ```

## Verification commands and results

Fresh pre-commit verification for implementation commit `54e0a65`:

```text
uv --cache-dir /private/tmp/finproof-uv-cache run pytest -q
197 passed, 4 skipped in 1.07s

uv --cache-dir /private/tmp/finproof-uv-cache run ruff check .
All checks passed!

uv --cache-dir /private/tmp/finproof-uv-cache run mypy src/ragbench
Success: no issues found in 32 source files

uv --cache-dir /private/tmp/finproof-uv-cache run alembic upgrade head --sql
Running upgrade 20260814_0002 -> 20260814_0003

uv --cache-dir /private/tmp/finproof-uv-cache run alembic downgrade \
  20260814_0003:20260814_0002 --sql
Running downgrade 20260814_0003 -> 20260814_0002

git diff --check
exit 0
```

Notebook validation parsed the artifact as JSON, checked `nbformat == 4`, and compiled every code
cell successfully:

```text
valid notebook: 6 cells
```

The four full-suite skips are external-resource tests. Task 8's pgvector parity test skips with an
explicit reason when its database URL/snapshot fixture variables are absent.

## Files

Created:

- `migrations/versions/20260814_0003_versioned_dense_index.py`
- `notebooks/02_cosine_from_scratch.ipynb`
- `scripts/build_embeddings.py`
- `src/ragbench/embeddings/{__init__,repository,service}.py`
- `src/ragbench/retrieval/{__init__,base,dense}.py`
- `tests/integration/retrieval/test_pgvector.py`
- `tests/unit/embeddings/{test_repository,test_service}.py`
- `tests/unit/retrieval/{test_cosine,test_dense}.py`
- `tests/unit/test_build_embeddings.py`

Modified:

- `pyproject.toml` and `uv.lock`
- `src/ragbench/core/config.py`
- `src/ragbench/db/models.py`
- `src/ragbench/providers/base.py`
- `src/ragbench/providers/upstage/client.py`

## Self-review

- Confirmed every provider embedding call flows through `ProviderGateway.embed`, retaining response
  caching, gross-cost reservation, settlement, retry bounds, and correlation evidence.
- Confirmed snapshot completion is not written until every expected vector exists and the HNSW DDL
  succeeds in the same PostgreSQL transaction. Retrieval requires both `complete` and index state
  `ready`.
- Confirmed the HNSW DDL interpolates only a parsed UUID and a strictly range-checked integer;
  vector values and retrieval filters remain bound SQL parameters.
- Confirmed NumPy and SQL share the same tie policy: score descending/distance ascending, then
  chunk ID ascending. SQL returns `1 - cosine_distance`, matching NumPy cosine scores.
- Found and corrected two issues during self-review before the implementation commit: chunk artifact
  IDs are deterministic provenance strings rather than database chunk UUIDs, so the new vector
  table stores those IDs directly; and the 64-character plan hash is not a PostgreSQL UUID, so the
  snapshot now derives a deterministic UUID while retaining the full plan hash separately.
- Confirmed the notebook is output-free and makes no claim of having exercised the provider or a
  database.

## DB and live limitations

No PostgreSQL service, populated Task 8 fixture snapshot, real chunk datasets, or provider approval
was available locally. Therefore:

- no migration was applied to a live database; only Alembic upgrade/downgrade SQL was generated;
- no HNSW index was physically built or query plan inspected;
- no live embedding request, paid or promotion-priced, was sent;
- no real core dataset embedding snapshot was completed;
- the required at-least-50-query NumPy/pgvector parity run remains pending; only the offline
  framework and configured multi-query/tie integration test were delivered;
- no deadline or free-pricing claim is made. The live builder requires a fresh price preflight and
  explicit paid confirmation even when a historical promotion date suggests a zero rate.

Implementation commit: `54e0a65 feat: add versioned dense retrieval index`.

## Fix round 1 — 4096D indexing, immutable chunk evidence, and self-contained parity

### Changes

- Replaced the late 4000-dimension rejection with an explicit storage/index plan validated by
  `build_plan` before gateway construction. Dimensions 1–2000 use full-vector HNSW. Dimensions
  2001–16000, including the configured 4096D path, index
  `subvector(embedding, 1, 2000)::vector(2000)`, over-fetch at least
  `max(20, candidate_factor * top_k)`, and rerank candidates with full-vector cosine distance.
  Snapshots persist the strategy and candidate factor.
- Made the partial-index snapshot UUID a parsed/generated literal in search SQL, matching the HNSW
  predicate exactly. Query vectors and document IDs remain bound parameters, and the indexed
  `ORDER BY` expression is byte-for-byte the expression used by the index.
- Added immutable `chunk_artifact` evidence keyed by `(embedding_snapshot_id, chunk_id)`, carrying
  document ID, verified content SHA-256, token count, and immutable source metadata. The manifest
  hash is part of snapshot identity, artifacts are registered before API calls, and vectors have a
  composite foreign key to their artifact.
- Hardened resume/idempotency: duplicate vectors are read and compared at pgvector's float32
  representation; different values fail rather than being hidden by `ON CONFLICT DO NOTHING`.
  Finalization compares the exact artifact/vector ID sets and checks the expected count. Database
  constraints enforce dimensions and finite, non-zero norms.
- Added per-snapshot PostgreSQL advisory transaction locks before create/register, persist, and
  finalize work, in addition to row locks once the snapshot exists. Concurrent duplicate builders
  therefore serialize before observing or changing snapshot state.
- Added optional canonical `document_ids` to `SearchFilter`; an empty tuple means no document
  restriction. Artifacts persist document identity and SQL applies the exact candidate filter.
- Added an exact `embedding-passage` price entry parallel to `embedding-query`, preventing the
  default document model from failing only after live execution starts.
- Migrated `retrieval_result.chunk_id` from UUID/FK-to-`chunk` to `VARCHAR(512)`, added a nullable
  legacy-compatible `embedding_snapshot_id`, and linked the pair to `chunk_artifact`. Downgrade
  fails explicitly if new non-UUID artifact IDs cannot be represented in the old schema.
- Replaced external parity fixture variables with a configured-DB self-contained integration test.
  It migrates a clean schema, seeds 20 normalized 2001D artifacts/vectors, exercises concurrent
  registration/persistence/finalization, runs 50 deterministic queries including ties, checks
  NumPy/pgvector ordered-ID parity at `1e-5`, checks document filtering, and uses `EXPLAIN` with
  sequential scans disabled to assert the generated HNSW index name is selected.
- Updated the notebook to explain the wide-vector subvector candidate stage and full reranking.

### TDD and verification evidence

The first focused run was RED at collection because the new manifest/index planning contracts did
not exist:

```text
ImportError: cannot import name 'chunk_manifest_hash' from
'ragbench.embeddings.repository'
```

After implementing the focused contracts:

```text
uv --cache-dir /private/tmp/finproof-uv-cache run pytest \
  tests/unit/embeddings/test_repository.py \
  tests/unit/embeddings/test_service.py \
  tests/unit/retrieval/test_dense.py \
  tests/unit/test_build_embeddings.py \
  tests/unit/providers/test_budget.py -q

.......................................                                  [100%]
39 passed in 0.18s
```

An intermediate full verification after the schema/repository integration reported:

```text
205 passed, 4 skipped in 1.11s
All checks passed!  # Ruff
Success: no issues found in 32 source files  # strict mypy
```

Alembic offline SQL generation covered the complete upgrade chain through revision
`20260814_0004` and the `20260814_0004:20260814_0003` downgrade. The final fresh verification and
commit are recorded in the task handoff.

### Remaining external limitation

No local PostgreSQL URL was available. The self-contained 50-query parity and `EXPLAIN` test is
implemented for CI but remained skipped locally solely on missing `RAGBENCH_TEST_DATABASE_URL`.
Accordingly, this round does not claim that a live database selected the index or passed parity.
No embedding API request was made.
