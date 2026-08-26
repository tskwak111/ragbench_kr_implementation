# RAGBench-KR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Korean long-document RAG benchmark that measures how parsing, chunking, retrieval, Top-K, and grounded prompting affect retrieval quality, answer quality, latency, and API cost.

**Architecture:** A Python monorepo separates deterministic domain logic from Upstage adapters, PostgreSQL persistence, FastAPI transport, experiment runners, and a late-stage dashboard. Every paid call passes through one cached, metered, budget-guarded gateway. Experiments are immutable YAML configurations and progress through retrieval screening, development-set generation evaluation, and a sealed human-validated gold test.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, pydantic-settings, SQLAlchemy 2.x async, Alembic, PostgreSQL 16 + pgvector, httpx, tenacity, tiktoken, rank-bm25, NumPy, pandas, PyArrow, Typer, pytest, pytest-asyncio, respx, Ruff, mypy, Docker Compose; optional React/Vite/TypeScript dashboard only after benchmark completion.

## Global Constraints

- Work in an isolated Git branch/worktree; never develop directly on the default branch.
- Use test-driven development: red test, minimal implementation, green test, refactor, commit.
- `pytest` must never make a real paid API call. All unit and contract tests use fakes or `respx`.
- Real Upstage calls run only when `RUN_LIVE_UPSTAGE_TESTS=1` and are excluded from normal CI.
- Never commit API keys, raw private documents, paid API responses, or database credentials.
- Every paid request must use a deterministic cache key, budget reservation, usage record, retry policy, and correlation ID.
- Use `Decimal` for money and integer token/page counts; never use binary float for budget enforcement.
- Default hard budget is `MAX_PROJECT_BUDGET_USD=135.00`, but the operator must set it no higher than the actual remaining promotional balance.
- Add a console reconciliation gate at least daily; local estimated cost is not the billing source of truth.
- As verified on 2026-08-13, official list prices are Solar Pro 3 input `$0.15/1M`, output `$0.60/1M`; Solar Pro 4 input `$0.30/1M`, output `$1.20/1M`; Document Parse Standard `$0.01/page`, Enhanced `$0.03/page`; Embed 2 is free through 2026-08-23 UTC and `$0.02/1M tokens` afterward. Prices exclude 10% VAT and must be rechecked before a paid batch.
- Store model IDs, API base URL, prices, and promotion end times in configuration; do not scatter provider identifiers through business logic.
- Pin the embedding model and vector dimension in each parse/index snapshot. Never compare retrieval runs built from different unrecorded embedding versions.
- Retain raw provider responses in the private cache, but publish only legally redistributable derived data and document provenance.
- Keep the 300-item gold split sealed until final model/configuration selection. Any viewed or tuned-on item moves to development data and is replaced.
- All claims in the README must be limited to the tested corpus, question set, and confidence intervals.
- Frontend, cloud deployment, semantic chunking, LangChain comparison, and load testing are stretch goals. They cannot delay the benchmark, gold test, cost analysis, or error analysis.

### Verified Provider References

- [Upstage API pricing](https://www.upstage.ai/pricing/api) — price and Embed 2 promotion values checked on 2026-08-13.
- [Upstage Embed documentation](https://console.upstage.ai/docs/capabilities/embed) — embedding capability and Korean/multilingual retrieval reference.
- [Upstage Document Parse Enhanced announcement](https://www.upstage.ai/blog/en/document-parse-enhanced) — Standard, Enhanced, and Auto behavior reference. The model ID shown in this older announcement is not a permanent contract; Task 4 must resolve the current console model ID before live use.
- [Upstage API getting started](https://console.upstage.ai/docs/getting-started) — credential and current API documentation entry point.

---

## 1. Product Definition and Research Questions

RAGBench-KR is an experiment platform, not primarily a chatbot. Its required outputs are a versioned corpus manifest, two parse variants, deterministic chunks, dense/BM25/hybrid retrieval, a cited Solar RAG path, synthetic and human-validated QA datasets, immutable experiment records, metrics, cost/latency logs, error taxonomy, and a final leaderboard.

The project must answer these questions with measured evidence:

1. How much does Enhanced parsing improve downstream retrieval and answer quality over Standard parsing?
2. Which chunk size/overlap works best for the selected Korean long-document corpus?
3. Which of dense, BM25, and RRF hybrid retrieval performs best by question type?
4. Does increasing Top-K improve answer correctness or merely increase noise, latency, and cost?
5. How much does a grounded/abstention prompt reduce unsupported answers?
6. Are the best-quality and best-value configurations different?
7. What fraction of final failures originates in parsing, chunk boundaries, retrieval, generation, citation, or benchmark defects?

### Non-goals

- General-purpose web search, autonomous browsing, user accounts, billing, production multi-tenancy, fine-tuning, security hardening beyond normal secret/input hygiene, and a polished commercial UI.
- Proving that one configuration is universally best for all Korean documents.
- Letting an LLM judge serve as the only ground truth.

## 2. Stage Gates and Revised Calendar

The original schedule started on 2026-08-10. Since implementation planning occurs on 2026-08-13, use this compressed calendar. Dates are targets; stage gates, not dates, authorize progression.

| Date | Gate | Required outcome |
|---|---|---|
| Aug 13 | G0 | Repository, config, Docker, CI, price/budget preflight |
| Aug 14 | G1 | Database, cached Upstage gateway, one-call smoke tests |
| Aug 15 | G2 | Corpus manifest frozen; licenses/provenance/page totals checked |
| Aug 16–17 | G3 | Standard and Enhanced parse snapshots complete and sampled |
| Aug 18 | G4 | Normalization and seven chunk variants complete |
| Aug 19 | G5 | Embed 2 index, NumPy cosine reference, pgvector parity complete |
| Aug 20 | G6 | BM25, RRF hybrid, cited RAG MVP complete |
| Aug 21 | G7 | Synthetic benchmark generated, validated, deduplicated |
| Aug 22 | G8 | Human-validated gold split sealed; judge calibration subset ready |
| Aug 23 UTC | G9 | All required Embed 2 indexing finished before promotion deadline |
| Aug 24 | G10 | Retrieval screening complete; top 8 configs selected by fixed rule |
| Aug 25 | G11 | Development generation runs and calibrated evaluation complete |
| Aug 26 | G12 | Top 3 sealed gold test, bootstrap CIs, paired comparisons complete |
| Aug 27 | G13 | Error/cost analysis, README, reproducibility package complete |
| Aug 28–31 | Buffer | Reruns first; dashboard and stretch work only if core gates are green |

**Stop rule:** if a gate fails, fix or explicitly reduce corpus/questions/run count according to the scope-reduction ladder. Do not skip measurement integrity to preserve the calendar.

## 3. Target Repository Structure

```text
ragbench-kr/
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── alembic.ini
├── migrations/
│   ├── env.py
│   └── versions/
├── configs/
│   ├── corpus.yaml
│   ├── prices.yaml
│   ├── prompts/v1_basic.txt
│   ├── prompts/v2_grounded.txt
│   ├── prompts/v3_abstain.txt
│   └── experiments/
├── data/
│   ├── README.md
│   ├── manifests/
│   ├── raw/.gitkeep
│   ├── parsed/.gitkeep
│   ├── benchmarks/.gitkeep
│   └── exports/.gitkeep
├── docs/
│   ├── decisions/
│   ├── runbooks/
│   └── reports/
├── src/ragbench/
│   ├── core/{config,errors,hashing,ids,money,versions}.py
│   ├── db/{base,models,session}.py
│   ├── providers/upstage/{client,schemas,pricing}.py
│   ├── ingestion/{manifest,parser,normalizer}.py
│   ├── chunking/{models,fixed,heading}.py
│   ├── embeddings/{service,repository}.py
│   ├── retrieval/{base,dense,bm25,rrf,service}.py
│   ├── rag/{context,prompts,citations,service}.py
│   ├── benchmark/{generation,validation,splits}.py
│   ├── evaluation/{retrieval,generation,judge,bootstrap,taxonomy}.py
│   ├── experiments/{config,planner,runner,selection}.py
│   ├── api/{app,dependencies,routes}.py
│   └── cli.py
├── scripts/
│   ├── collect_corpus.py
│   ├── parse_corpus.py
│   ├── build_chunks.py
│   ├── build_embeddings.py
│   ├── generate_benchmark.py
│   ├── run_retrieval_screen.py
│   ├── run_experiment.py
│   ├── run_gold_test.py
│   ├── export_results.py
│   └── reconcile_usage.py
├── notebooks/
│   ├── 01_chunk_inspection.ipynb
│   ├── 02_cosine_from_scratch.ipynb
│   └── 03_error_analysis.ipynb
└── tests/
    ├── unit/
    ├── contract/
    ├── integration/
    └── fixtures/
```

## 4. Stable Interfaces

These signatures are contracts for all tasks. Change them only through an explicit architecture-decision record and update every dependent task/test in the same commit.

```python
@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    billable_pages: int
    estimated_cost_usd: Decimal


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    parse_run_id: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    content: str
    token_count: int
    strategy: str
    ordinal: int


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float
    rank: int
    retriever: str


class Retriever(Protocol):
    async def search(self, query: str, *, top_k: int, filter: SearchFilter) -> list[SearchHit]: ...


class ProviderGateway(Protocol):
    async def parse(self, request: ParseRequest) -> ParsedDocument: ...
    async def embed(self, request: EmbedRequest) -> EmbedResponse: ...
    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...


class ExperimentRunner:
    async def run(
        self, config: ExperimentConfig, question_ids: Sequence[str]
    ) -> ExperimentSummary: ...
```

## 5. Implementation Tasks

### Task 1: Repository Bootstrap, Typed Configuration, and CI

**Files:** Create `pyproject.toml`, `.env.example`, `.gitignore`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `src/ragbench/core/config.py`, `src/ragbench/core/versions.py`, `tests/unit/core/test_config.py`.

**Interfaces:** Produces `Settings()`, `PriceBook.from_yaml(path)`, and `VersionBundle`; all later tasks consume them.

- [x] **Step 1: Write failing settings tests.** Assert missing `UPSTAGE_API_KEY` is allowed for offline commands, live mode requires it, budget parses as `Decimal("135.00")`, concurrency defaults to 5, retries to 5, and gold access defaults false.
- [x] **Step 2: Run** `pytest tests/unit/core/test_config.py -v` **and confirm import failure.**
- [x] **Step 3: Implement the package and settings.** Use `pydantic-settings`; include `database_url`, `upstage_api_key`, `upstage_base_url`, `max_project_budget_usd`, `max_concurrency`, `max_retries`, `run_live_upstage_tests`, `allow_gold_access`, cache/data paths, provider model IDs, and promotion deadline.
- [x] **Step 4: Add tool configuration.** Ruff checks `E,F,I,UP,B,SIM`; mypy uses strict mode for `src/ragbench`; pytest registers `live`, `integration`, and `gold` markers.
- [x] **Step 5: Add CI.** Run Ruff, mypy, and all non-live/non-gold tests on Python 3.12. Add a guard that fails if tracked files contain patterns matching `UPSTAGE_API_KEY=...` with a nonempty value.
- [x] **Step 6: Verify** `ruff check .`, `mypy src/ragbench`, and `pytest -m 'not live and not gold' -q` **all pass.**
- [x] **Step 7: Commit** `chore: bootstrap typed ragbench project`.

### Task 2: Database Schema and Migrations

**Files:** Create `src/ragbench/db/base.py`, `models.py`, `session.py`, `migrations/env.py`, initial migration, `tests/integration/db/test_schema.py`, `docker-compose.yml`.

**Interfaces:** Produces async session factory and models `Document`, `ParseRun`, `Chunk`, `Question`, `Experiment`, `ExperimentResponse`, `RetrievalResult`, `Metric`, `ApiUsage`, `ApiCacheEntry`, and `BudgetReservation`.

- [x] **Step 1: Write an integration test** that migrates an empty PostgreSQL database to head, inserts one document and experiment, enforces unique document SHA-256, and verifies vector extension availability.
- [ ] **Step 2: Start test DB** with `docker compose up -d db` and confirm the test fails because tables are absent.
- [x] **Step 3: Implement models.** Use UUID primary keys; UTC timestamps; JSONB for immutable configuration snapshots; `Numeric(12,6)` for USD; pgvector column whose dimension comes from a recorded embedding snapshot; foreign keys with restrictive deletes for experiment evidence.
- [x] **Step 4: Add indexes** on document hash, `(parse_run_id, strategy, ordinal)`, question split/type, experiment status, usage correlation ID, and vector HNSW after vectors are populated.
- [x] **Step 5: Generate and inspect migration.** Ensure downgrade removes only objects created by this migration.
- [ ] **Step 6: Run** `alembic upgrade head`, integration test, then `alembic downgrade base && alembic upgrade head`.
- [x] **Step 7: Commit** `feat: add experiment persistence schema`.

### Task 3: Deterministic IDs, Cache, Pricing, Budget Guard, and Retry Policy

**Files:** Create `core/hashing.py`, `core/ids.py`, `core/money.py`, `providers/upstage/pricing.py`, `providers/upstage/client.py`, `tests/unit/providers/test_budget.py`, `test_cache_key.py`, `tests/contract/providers/test_upstage_client.py`, `configs/prices.yaml`.

**Interfaces:** Produces `canonical_json_hash(value) -> str`, `PriceBook.estimate(request) -> Decimal`, `BudgetGuard.reserve(...)`, `BudgetGuard.settle(...)`, and `UpstageGateway` implementing `ProviderGateway`.

- [x] **Step 1: Write failing tests** for stable hashes across dictionary key order, price calculations for parse/chat/embed, rejection at the hard limit, release of failed reservations, cache hit without HTTP traffic, 429 retry with bounded exponential backoff, and no retry for 400/401.
- [x] **Step 2: Run focused tests** and confirm failures.
- [x] **Step 3: Implement canonical cache keys** from operation, exact model ID, provider parameters, prompt/context hashes, document SHA-256, and schema version. Never include the plaintext API key.
- [x] **Step 4: Implement atomic budget reservation** inside a database transaction: sum settled usage plus open reservations, compare projected cost with hard cap, insert reservation, then settle actual/estimated usage after response. A request without calculable upper bound must provide `max_output_tokens`.
- [x] **Step 5: Implement `httpx.AsyncClient` adapter** with explicit connect/read/write/pool timeouts, semaphore concurrency, `Retry-After` support, jittered backoff for 429/5xx/network errors, request correlation IDs, raw response cache, and redacted logs.
- [x] **Step 6: Add price preflight.** `ragbench prices verify` prints the loaded price snapshot and refuses paid batches when `verified_at` is older than 24 hours or promotion status is ambiguous.
- [x] **Step 7: Verify contract tests** with `respx`; assert the second identical call produces zero HTTP requests and one cache-hit usage record.
- [x] **Step 8: Commit** `feat: guard and cache all provider usage`.

### Task 4: API Smoke Tests and Operational Preflight

**Files:** Create `src/ragbench/cli.py`, `tests/unit/test_cli.py`, `tests/live/test_upstage_smoke.py`, `docs/runbooks/provider-smoke.md`.

**Interfaces:** Produces CLI commands `preflight`, `smoke solar`, `smoke parse`, `smoke embed`, and `usage status`.

- [x] **Step 1: Write CLI tests** using a fake gateway. Confirm offline preflight checks Docker, DB migration, writable cache path, price freshness, budget remaining, and secret presence without printing the secret.
- [x] **Step 2: Implement commands** with Typer and machine-readable `--json` output.
- [x] **Step 3: Add live tests** guarded by both the environment flag and pytest marker. Use one short Korean prompt, one 1-page fixture PDF, and one short embedding input; cap output and projected spend.
- [ ] **Step 4: Run offline tests.** Then, only with operator approval and credentials, run each live smoke exactly once and record timestamp/model/latency/usage IDs.
- [ ] **Step 5: Verify raw responses are cached** and rerunning smoke with identical input causes no paid request.
- [x] **Step 6: Commit** `feat: add provider smoke and preflight commands`.

### Task 5: Corpus Manifest, Provenance, and Freeze Gate

**Files:** Create `ingestion/manifest.py`, `scripts/collect_corpus.py`, `configs/corpus.yaml`, `data/README.md`, `tests/unit/ingestion/test_manifest.py`, `docs/decisions/001-corpus-policy.md`.

**Interfaces:** Produces `CorpusManifest.load()`, `validate()`, and immutable `corpus_snapshot_id` derived from sorted document hashes and metadata.

- [x] **Step 1: Define manifest schema** with `document_id`, title, organization, year, type, language, source URL, download date, license/redistribution status, local path, SHA-256, page count, and inclusion rationale.
- [x] **Step 2: Write failing validation tests** for duplicate content, missing provenance, unsupported license state, mismatched page count, malformed PDF, and total page distribution.
- [x] **Step 3: Implement collection script.** It copies approved PDFs into ignored `data/raw`, computes hashes/page counts, never silently overwrites, and produces a review report rather than auto-accepting unknown licenses.
- [x] **Step 4: Assemble 20–30 documents totaling 1,500–2,000 pages** across corporate/public reports, including table-heavy and text-heavy strata. Avoid a corpus dominated by one organization or one template.
- [x] **Step 5: Freeze snapshot** only after manifest validation passes and manually inspect random first/middle/last pages from every file for corruption.
- [x] **Step 6: Export a public-safe manifest** excluding nonredistributable document bytes while retaining source/provenance.
- [x] **Step 7: Commit** `data: freeze corpus manifest` (manifest only, not restricted PDFs).

### Task 6: Standard and Enhanced Parsing Pipeline

**Files:** Create `ingestion/parser.py`, `scripts/parse_corpus.py`, `tests/unit/ingestion/test_parser.py`, `tests/contract/ingestion/test_parse_resume.py`.

**Interfaces:** Produces `parse_corpus(snapshot_id, mode, resume=True) -> ParseSummary`; persists provider model/version, mode, raw response hash, markdown/html elements, page mappings, latency, cost, and status.

- [x] **Step 1: Write tests** for one successful page, partial failure, resumable batch, corrupted cached response, duplicate invocation, and page-count reconciliation.
- [x] **Step 2: Implement a dry-run planner** that lists documents/pages, cache hits, projected new calls, worst-case price including VAT buffer, and post-run remaining budget. Paid mode requires `--confirm-plan <plan-hash>`.
- [x] **Step 3: Implement Standard parse execution** with per-document checkpoints and raw response preservation.
- [x] **Step 4: Run Standard on the frozen corpus.** All 20 documents and 1,981 pages succeeded with `document-parse-260630`; no source was repaired or excluded. The deleted local checkpoints must be reconstructed before Step 6.
- [ ] **Step 5: Implement and run Enhanced against the identical effective corpus.** Reject mismatched source hashes or page sets.
- [ ] **Step 6: Perform stratified manual QA** on at least 30 page pairs: complex tables, financial statements, charts, multi-column pages, and ordinary prose. Record transcription/structure/visual-element findings with blinded mode labels where practical.
- [ ] **Step 7: Reconcile local page charges with provider console** before proceeding.
- [ ] **Step 8: Commit** `feat: add resumable dual-mode parsing`.

### Task 7: Normalization and Chunking Variants

**Files:** Create `chunking/models.py`, `ingestion/normalizer.py`, `chunking/fixed.py`, `chunking/heading.py`, `scripts/build_chunks.py`, `tests/unit/chunking/test_fixed.py`, `test_heading.py`, `test_normalizer.py`.

**Interfaces:** Produces `normalize(parsed) -> list[DocumentBlock]`, `FixedChunker(size, overlap).split(blocks)`, and `HeadingAwareChunker(target_size=600, overlap=100).split(blocks)` returning `ChunkRecord` values.

- [x] **Step 1: Write fixture-driven tests** for Korean text, tables, headings, empty pages, repeated headers/footers, page transitions, chunks smaller than overlap, and deterministic IDs.
- [x] **Step 2: Implement conservative normalization.** Normalize Unicode/line endings and repeated boilerplate but preserve numbers, table cell order, page anchors, headings, and original normalized-to-source provenance.
- [x] **Step 3: Implement token-aware fixed chunking** for `(300,0)`, `(300,100)`, `(600,0)`, `(600,100)`, `(1000,0)`, `(1000,100)`. Define overlap in tokens and prevent zero-progress loops.
- [x] **Step 4: Implement heading-aware chunking.** Prefer section boundaries, split oversized sections token-wise, merge undersized neighboring blocks within the same section, and retain page ranges.
- [x] **Step 5: Add invariants**: deterministic output, no lost non-boilerplate text, monotonic ordinals, valid page ranges, token count within documented tolerance, and unique chunk IDs containing parse snapshot and strategy hash.
- [ ] **Step 6: Build all 14 core chunk datasets** (2 parse modes × 7 strategies) and export size/coverage distributions.
- [ ] **Step 7: Manually inspect one text-heavy and one table-heavy document in `01_chunk_inspection.ipynb`; record boundary defects.**
- [ ] **Step 8: Commit** `feat: add provenance-preserving chunkers`.

### Task 8: Embed 2 Indexing, Cosine Reference, and pgvector Parity

**Files:** Create `embeddings/service.py`, `embeddings/repository.py`, `retrieval/dense.py`, `scripts/build_embeddings.py`, `notebooks/02_cosine_from_scratch.ipynb`, `tests/unit/retrieval/test_cosine.py`, `tests/integration/retrieval/test_pgvector.py`.

**Interfaces:** Produces `EmbeddingService.embed_chunks(snapshot)`, `cosine_top_k(query, matrix, k)`, and `DenseRetriever.search(...)`.

- [ ] **Step 1: Write cosine tests** with hand-calculated vectors, zero-vector rejection, tie-breaking by stable chunk ID, and Top-K bounds.
- [ ] **Step 2: Implement NumPy reference cosine search** and explain dot product, norms, normalization, sorting, and Top-K in the notebook.
- [ ] **Step 3: Write embedding service tests** for batching, request-size limits, caching, resume, model/dimension mismatch, and separate query/document input modes if the provider supports them.
- [ ] **Step 4: Implement index snapshots.** A snapshot records corpus, parse, chunk strategy, embedding model ID, dimension, normalization settings, created time, and complete/incomplete state. Retrieval refuses incomplete snapshots.
- [ ] **Step 5: Build required embeddings before 2026-08-23 UTC.** Run dry-run/cache audit first; prioritize all 14 core chunk datasets over stretch variants.
- [ ] **Step 6: Implement pgvector cosine retrieval** with document/parse/strategy filters.
- [ ] **Step 7: Run parity test** on at least 50 queries; NumPy and pgvector must return identical ordered chunk IDs within tie policy and score tolerance `1e-5`.
- [ ] **Step 8: Commit** `feat: add versioned dense retrieval index`.

### Task 9: BM25 and RRF Hybrid Retrieval

**Files:** Create `retrieval/base.py`, `retrieval/bm25.py`, `retrieval/rrf.py`, `retrieval/service.py`, `tests/unit/retrieval/test_bm25.py`, `test_rrf.py`, `test_service.py`.

**Interfaces:** `BM25Retriever` and `HybridRetriever` implement the stable `Retriever` protocol. `reciprocal_rank_fusion(rankings, k=60, weights=None)` returns deterministic hits.

- [ ] **Step 1: Write BM25 tests** for Korean whitespace/token normalization, exact numeric terms, empty query, repeated terms, and deterministic rank ties.
- [ ] **Step 2: Implement a documented baseline tokenizer.** Keep numeric strings, normalize case/Unicode, split Korean text conservatively; do not add morphological dependencies during the core experiment.
- [ ] **Step 3: Write RRF tests** with hand-computed `1/(60+rank)` scores, missing candidates, duplicate chunk IDs, weights, and ties.
- [ ] **Step 4: Implement hybrid retrieval** by over-fetching each branch to `max(20, 4*top_k)`, fusing, and returning top K. Record dense rank, sparse rank, component scores, and fused score.
- [ ] **Step 5: Add common filtering** so dense, sparse, and hybrid search the exact same chunk snapshot.
- [ ] **Step 6: Verify all three retrievers** on a fixed set of Korean fact, numeric, and paraphrase queries and save a comparison artifact.
- [ ] **Step 7: Commit** `feat: add sparse and hybrid retrieval`.

### Task 10: Context Builder, Grounded Generation, and Citations

**Files:** Create `rag/context.py`, `rag/prompts.py`, `rag/citations.py`, `rag/service.py`, three prompt files, `tests/unit/rag/test_context.py`, `test_citations.py`, `test_prompts.py`.

**Interfaces:** Produces `ContextBuilder.build(hits, token_budget) -> ContextBundle` and `RagService.answer(question, config) -> RagAnswer` with structured citations.

- [ ] **Step 1: Write tests** for stable context ordering, duplicate removal, token-budget truncation, citation IDs, invalid model citations, unanswerable response, and malicious instructions embedded in documents.
- [ ] **Step 2: Implement context serialization** with explicit delimiters and metadata: `chunk_id`, document title/ID, page range, section, and content. Treat document content as data, not instructions.
- [ ] **Step 3: Create prompts:** V1 basic; V2 context-only with required citations; V3 context-only with explicit abstention when evidence is insufficient. Require JSON output with `answer`, `citations`, and `abstained`.
- [ ] **Step 4: Implement strict response parsing** through Pydantic; repair only syntactic JSON once, never invent citations, and mark persistent schema failure as `GENERATION_SCHEMA_ERROR`.
- [ ] **Step 5: Validate citations** only against retrieved chunk IDs and attach source page/section server-side. Unsupported or unknown citations fail citation validation.
- [ ] **Step 6: Add `POST /query` service path** or CLI equivalent returning answer, retrieved evidence, citations, latency, usage, experiment/config IDs, and cached status.
- [ ] **Step 7: Verify with fake provider**, then a tiny live set after budget preflight.
- [ ] **Step 8: Commit** `feat: add cited grounded rag pipeline`.

### Task 11: Synthetic Benchmark Generation and Automated Validation

**Files:** Create `benchmark/generation.py`, `benchmark/validation.py`, `scripts/generate_benchmark.py`, `tests/unit/benchmark/test_generation.py`, `test_validation.py`, `configs/benchmark.yaml`.

**Interfaces:** Produces `QuestionCandidate` with question, normalized gold answer, evidence spans, page/chunk provenance, type, difficulty, answerable flag, generator metadata, and validation status.

- [ ] **Step 1: Write schema/validation tests** for invalid JSON, missing evidence, evidence not found in source, numeric mismatch, duplicate questions, leaked answer in question, impossible page, and malformed unanswerable item.
- [ ] **Step 2: Implement stratified generation planner** targeting 300 fact, 300 table/numeric, 250 comparison, 250 multi-hop, 200 unanswerable, and 200 complex/summary candidates. Balance documents and avoid generating all items from easy pages.
- [ ] **Step 3: Generate answerable items from bounded source windows** and require verbatim supporting spans plus reasoning metadata. Generate unanswerable questions by controlled transformation and verify their asserted facts are absent from the document snapshot.
- [ ] **Step 4: Implement automatic filters**: exact/semantic duplicate grouping, evidence substring/fuzzy check, page validity, answer-evidence consistency, type quotas, per-document caps, and contamination checks.
- [ ] **Step 5: Run generation in resumable batches** with cache and budget guard until 1,500 valid candidates or the reduced-scope threshold is reached.
- [ ] **Step 6: Produce a validation report** listing rejection counts by rule, distributions, and samples for manual review.
- [ ] **Step 7: Commit** `feat: generate traceable benchmark candidates`.

### Task 12: Human Validation, Splits, and Gold Sealing

**Files:** Create `benchmark/splits.py`, `data/benchmarks/review_template.csv`, `docs/runbooks/human-validation.md`, `tests/unit/benchmark/test_splits.py`.

**Interfaces:** Produces versioned `dev_auto`, `test_gold`, and `stress` snapshots; gold access requires `ALLOW_GOLD_ACCESS=1` and an explicit command.

- [ ] **Step 1: Define review columns**: natural question, answer exists, evidence correct, page correct, answer unambiguous, answerable label correct, type/difficulty correct, reviewer decision, corrected answer/evidence, notes, reviewer ID, timestamp.
- [ ] **Step 2: Write split tests** preventing question-family/document leakage where feasible, duplicate paraphrases across splits, accidental gold loading by normal runners, and mutation after sealing.
- [ ] **Step 3: Sample at least 300 candidates** stratified by type, difficulty, document, parse-sensitive pages, and answerability. Randomize order and hide generator confidence.
- [ ] **Step 4: Manually review every selected item** against the original PDF and parsed representations. Accept, correct, or reject; never approve using only the generated answer.
- [ ] **Step 5: Double-review at least 50 items** and calculate raw agreement and Cohen's kappa for categorical decisions. Resolve disagreements with written rules.
- [ ] **Step 6: Seal 300 gold items** only if quality threshold is met; otherwise use the scope floor of 150 and report the reduction. Hash the file, store only its metadata in normal development logs, and prohibit preview commands.
- [ ] **Step 7: Commit public-safe split metadata and review protocol**; keep restricted benchmark content according to source licensing.

### Task 13: Retrieval Metrics and Screening

**Files:** Create `evaluation/retrieval.py`, `experiments/config.py`, `experiments/planner.py`, `scripts/run_retrieval_screen.py`, `tests/unit/evaluation/test_retrieval_metrics.py`, `test_experiment_config.py`.

**Interfaces:** Produces Hit@K, evidence Recall@K, MRR, latency, and per-type aggregates for immutable `RetrievalExperimentConfig`.

- [ ] **Step 1: Write metric tests** using tiny rankings with hand-calculated Hit@K, multi-evidence Recall@K, MRR, no-evidence handling, macro/micro aggregation, and confidence interval inputs.
- [ ] **Step 2: Define YAML schema** containing corpus/parse/chunk/embedding snapshots, retriever, RRF parameters, K, question snapshot, random seed, code commit, and metric version.
- [ ] **Step 3: Generate the 126 core configurations**: 2 parse modes × 7 chunk strategies × 3 retrievers × 3 K values. Fail on duplicate semantic configurations.
- [ ] **Step 4: Run retrieval screening without generation** on development questions. Persist every ranked hit, not just aggregates.
- [ ] **Step 5: Select top 8 using a predeclared rule:** prioritize Recall@5/appropriate K, then MRR, then latency; require diversity so the shortlist does not consist solely of near-identical settings. Record rule before viewing outcomes.
- [ ] **Step 6: Export leaderboard** with overall and per-question-type metrics plus paired bootstrap confidence intervals for key comparisons.
- [ ] **Step 7: Commit** `feat: add reproducible retrieval screening`.

### Task 14: Generation Evaluators and Human Calibration

**Files:** Create `evaluation/generation.py`, `evaluation/judge.py`, `evaluation/bootstrap.py`, `tests/unit/evaluation/test_generation_metrics.py`, `test_judge_parser.py`, `docs/runbooks/judge-calibration.md`.

**Interfaces:** Produces exact/numeric normalization metrics, correctness, faithfulness, citation precision/recall, abstention accuracy, and calibrated judge records with prompt/model versions.

- [ ] **Step 1: Implement deterministic metrics first.** Tests cover Korean whitespace/punctuation normalization, percentages, commas/currency, acceptable aliases, citation-support mapping, correct abstention, false abstention, and false answer.
- [ ] **Step 2: Define judge rubric** with independent fields and cited rationale: correctness 0–1, every-claim faithfulness, citation support, and benchmark-defect flag. Judge sees question, gold/evidence, model answer, and retrieved context but not configuration identity.
- [ ] **Step 3: Use a distinct judge model/configuration** from the primary generator when available; store the precise model and rubric hash. Temperature is zero where supported.
- [ ] **Step 4: Human-score 100–300 sampled responses** balanced across systems/question types. Include disagreements and failures, not only easy correct answers.
- [ ] **Step 5: Calibrate judge** using correlation for ordinal/continuous scores and agreement/F1 for binary thresholds. Report bias by question type and do not use an uncalibrated judge as the final authority.
- [ ] **Step 6: Add paired bootstrap** with fixed seed and at least 10,000 resamples for final confidence intervals; keep document-cluster bootstrap as the preferred sensitivity analysis.
- [ ] **Step 7: Commit** `feat: add calibrated rag evaluation`.

### Task 15: Immutable Experiment Runner and Development Runs

**Files:** Create `experiments/runner.py`, `experiments/selection.py`, `scripts/run_experiment.py`, `tests/unit/experiments/test_runner.py`, `tests/contract/experiments/test_resume.py`, example YAML configs.

**Interfaces:** `run_experiment.py --config ... [--dry-run] [--resume]`; experiment ID is derived from config hash plus run timestamp, while semantic duplicate detection uses config hash alone.

- [ ] **Step 1: Write state-machine tests** for `PLANNED -> RUNNING -> COMPLETED`, partial failure/resume, cancellation, cached response reuse, config mutation rejection, and budget exhaustion.
- [ ] **Step 2: Implement dry-run plan** showing question count, expected cached/new calls, worst-case tokens/cost, concurrency, model/prompt versions, and output destinations. Paid execution requires plan-hash confirmation.
- [ ] **Step 3: Implement bounded worker pool** starting at concurrency 5; increase to 10 only after measured 429/error rate remains acceptable. Preserve per-question isolation and idempotency.
- [ ] **Step 4: Run top 8 retrieval configs × 3 prompts** over 500 development questions first. Review quality/cost/error rates before expanding toward 1,000 questions or 15,000–24,000 responses.
- [ ] **Step 5: Stop automatically** at budget cap, operator-set batch cap, abnormal schema-error rate, or sustained provider-error threshold. Resume only after diagnosis.
- [ ] **Step 6: Select top 3** using a predeclared multi-objective rule across correctness, faithfulness, citation, abstention, latency, and cost. Keep one best-value candidate when statistically competitive.
- [ ] **Step 7: Reconcile provider billing and commit** `feat: run resumable generation experiments`.

### Task 16: Sealed Gold Test and Statistical Comparison

**Files:** Create `scripts/run_gold_test.py`, `evaluation/taxonomy.py`, `tests/unit/evaluation/test_taxonomy.py`, `docs/reports/gold-test-template.md`.

**Interfaces:** Gold runner accepts exactly the frozen top-three config hashes and sealed gold snapshot; it cannot accept arbitrary configs after unsealing.

- [ ] **Step 1: Verify preregistration artifact** contains top-three hashes, metrics, primary comparison, bootstrap method, exclusions, and stopping rule before gold access.
- [ ] **Step 2: Enable gold access for one controlled run** and execute all three configurations over the same 300 items with deterministic ordering and cached restart support.
- [ ] **Step 3: Compute paired results** overall and by type: correctness, faithfulness, citation, abstention, latency, cost, Hit/Recall/MRR, 95% confidence intervals, and effect sizes.
- [ ] **Step 4: Do not tune after viewing gold.** Any bug fix that changes outputs invalidates the affected run; document it, version a new test snapshot only when justified, and label the previous result.
- [ ] **Step 5: Optionally compare Solar Pro 3 vs Pro 4** on a fixed, budgeted subset only after core gold results exist; label it exploratory unless preregistered.
- [ ] **Step 6: Commit aggregate report inputs** without leaking restricted question content.

### Task 17: Error Analysis, Cost Analysis, and Final Claims

**Files:** Create `notebooks/03_error_analysis.ipynb`, `scripts/export_results.py`, `docs/reports/final-analysis.md`, tests for export aggregation.

**Interfaces:** Produces machine-readable CSV/Parquet tables and publication-ready figures from immutable experiment IDs.

- [ ] **Step 1: Freeze taxonomy**: `PARSER_ERROR`, `RETRIEVAL_MISS`, `CHUNK_BOUNDARY`, `TABLE_ERROR`, `RETRIEVAL_NOISE`, `GENERATION_ERROR`, `HALLUCINATION`, `BAD_CITATION`, `FALSE_ABSTENTION`, `BENCHMARK_DEFECT`, plus one primary and optional secondary label.
- [ ] **Step 2: Sample 50–100 failures** stratified across top systems and question types. Inspect original PDF, both parses, chunks, rankings, answer, and citations in that order.
- [ ] **Step 3: Calculate cost** by operation/model/config/question type, cached versus new calls, parse amortization, marginal cost per quality point, and actual console reconciliation delta.
- [ ] **Step 4: Produce core tables/plots:** final leaderboard; Standard vs Enhanced paired difference; chunk heatmap; retriever × type; K trade-off; prompt/abstention effect; accuracy-cost Pareto frontier; latency distribution; failure taxonomy.
- [ ] **Step 5: Answer all seven research questions** with scope-limited claims, confidence intervals/effect sizes, observed limitations, and counterexamples.
- [ ] **Step 6: Verify exports regenerate from a clean DB snapshot** and contain experiment/config/data/code version identifiers.
- [ ] **Step 7: Commit** `docs: add benchmark error and cost analysis`.

### Task 18: FastAPI, Reproduction, README, and Optional Dashboard

**Files:** Create `api/app.py`, dependencies/routes, API tests, complete `README.md`, `docs/runbooks/reproduction.md`; optionally `frontend/` only after core completion.

**Interfaces:** Required endpoints: `GET /health`, `POST /documents`, `GET /documents/{id}`, `POST /search`, `POST /query`, `POST /experiments` (plan/queue only), `GET /experiments`, `GET /experiments/{id}`, and `GET /experiments/{id}/metrics`.

- [ ] **Step 1: Write API contract tests** for health readiness, schema validation, unknown IDs, search filters, query citations, experiment dry-run response, and refusal to hold a request open for a large batch.
- [ ] **Step 2: Implement thin routes** delegating to domain services. Use consistent error envelopes and correlation IDs; never expose secrets, raw provider errors, or restricted benchmark records.
- [ ] **Step 3: Make `docker compose up --build` start PostgreSQL and API**, run migrations safely, expose health checks, and use named volumes. Dashboard is optional profile `--profile ui`.
- [ ] **Step 4: Write README in results-first order:** Problem, Key Results, Architecture, Dataset, Methodology, Experiments, Retrieval, Generation, Error Analysis, Cost, Demo, Reproduction, Limitations, Ethics/licensing.
- [ ] **Step 5: Add exact reproduction commands** for an offline fixture benchmark and for a user-supplied corpus. A reviewer must be able to run unit tests and a tiny no-cost fixture experiment without an Upstage key.
- [ ] **Step 6: If all core gates pass, build four dashboard views:** Overview, Leaderboard, Compare, RAG Playground. Read only aggregate/public-safe endpoints.
- [ ] **Step 7: Run final verification matrix** below and commit `docs: publish reproducible ragbench results`.

## 6. Required Experiment Configuration

```yaml
schema_version: 1
name: enhanced-heading-hybrid-k5-v3
snapshots:
  corpus: CORPUS-SHA256
  parse: PARSE-SHA256
  chunks: CHUNKS-SHA256
  embeddings: EMBEDDINGS-SHA256
  questions: DEV-SHA256
retrieval:
  type: hybrid
  top_k: 5
  dense_candidates: 20
  sparse_candidates: 20
  rrf_k: 60
generation:
  provider: upstage
  model: solar-pro3
  prompt_version: v3
  temperature: 0
  max_output_tokens: 512
evaluation:
  deterministic_version: v1
  judge_model: solar-pro4
  judge_prompt_version: v1
runtime:
  concurrency: 5
  max_retries: 5
  seed: 20260813
```

At load time, resolve symbolic model aliases to exact provider model IDs and store both. The resolved immutable configuration—not merely the YAML filename—defines the experiment hash.

## 7. Metrics Definitions

- **Hit@K:** 1 if at least one gold evidence chunk is in the first K results, else 0.
- **Evidence Recall@K:** number of distinct required evidence units retrieved in top K divided by total required evidence units.
- **MRR:** reciprocal rank of the first relevant result; 0 if absent.
- **Answer correctness:** deterministic exact/numeric/alias checks where applicable plus calibrated human/judge rubric for open answers.
- **Faithfulness:** proportion of material answer claims supported by retrieved context; unsupported claims are failures even if factually true externally.
- **Citation precision:** cited evidence units that support a claim divided by all cited units.
- **Citation recall:** answer claims requiring support that have at least one supporting citation divided by all such claims.
- **Abstention accuracy:** correct behavior on answerable and unanswerable items; also report false-answer and false-abstention rates separately.
- **Best value:** Pareto-efficient configuration with no statistically meaningful quality loss beyond the preregistered tolerance and lower total/marginal cost.

## 8. Budget and Operational Controls

Before every paid batch, save and review a dry-run artifact containing current provider-console balance, local settled usage, open reservations, cached request count, new request count, maximum pages/tokens, price snapshot timestamp, VAT assumption, worst-case batch cost, hard-stop threshold, and plan hash.

Initial allocation is a planning envelope, not permission to spend:

| Work | Planning amount |
|---|---:|
| Standard parse, 2,000 pages | $20 |
| Enhanced parse, 2,000 pages | $60 |
| Synthetic QA | $6 |
| Development RAG generations | $23 |
| Judge evaluation | $7 |
| Small Pro 4 comparison | $5 |
| Embed 2 before free deadline | approximately $0 |
| Reserve/buffer | $14 |

The line items consume the entire `$135` hard-stop envelope before any estimation drift or VAT. The runner therefore authorizes batches incrementally and treats optional Pro 4 comparison and expanded generations as first cuts. Console balance and expiry terms override this table.

## 9. Scope-Reduction Ladder

If time or budget is insufficient, reduce in this order and record the decision:

1. Remove dashboard and deployment.
2. Remove Pro 4 exploratory comparison, semantic chunking, LangChain comparison, and load tests.
3. Reduce development generations from 24,000 toward 10,000 while preserving balanced sampling.
4. Reduce synthetic candidates from 1,500 toward 800.
5. Reduce gold from 300 toward a minimum of 150, retaining stratification and human review.
6. Reduce corpus from 30 toward 20 documents while preserving at least 1,000 varied pages if feasible.

Never cut dual parsing, fixed/heading chunks, dense/BM25/hybrid retrieval, citations, benchmark validation, retrieval metrics, answer/faithfulness/citation/abstention evaluation, budget guard, gold sealing, or error analysis.

## 10. Final Verification Matrix

Run from a fresh clone with documented prerequisites:

```bash
cp .env.example .env
docker compose up -d db
uv sync --all-extras
uv run alembic upgrade head
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ragbench
uv run pytest -m "not live and not gold" -q
uv run ragbench preflight --offline
uv run python scripts/run_retrieval_screen.py --config tests/fixtures/mini-screen.yaml
uv run python scripts/run_experiment.py --config tests/fixtures/mini-experiment.yaml --offline
docker compose up --build -d
curl --fail http://localhost:8000/health
```

Expected results:

- All static checks and non-live tests pass.
- Migration is reproducible on an empty database.
- Offline fixture screening and generation evaluation complete without a provider key or network call.
- API health reports application, database, migration, and version state.
- Re-running the same fixture experiment reuses cached outputs and creates no duplicate responses.
- Every published table identifies corpus, question, experiment, code, prompt, model, price, and metric versions.

## 11. Definition of Done

- [ ] 1,500–2,000 diverse pages are represented by a validated, frozen manifest, or a documented reduced scope meets the floor.
- [ ] Standard and Enhanced parse snapshots cover the identical corpus and have manual paired QA.
- [ ] Seven chunk strategies per parse mode pass invariants and visual inspection.
- [ ] Embed 2 snapshots are complete, cached, versioned, and parity-tested against NumPy cosine search.
- [ ] Dense, BM25, and RRF hybrid retrievers share filters and return reproducible rankings.
- [ ] RAG responses are structured, grounded, abstention-aware, and citation-validated.
- [ ] At least 1,000 synthetic candidates and 150–300 human-validated gold items exist with traceable evidence.
- [ ] Retrieval, generation, citation, abstention, latency, and cost metrics are implemented and tested.
- [ ] Retrieval screening, development evaluation, and sealed gold evaluation are separated.
- [ ] LLM judge scores are calibrated against human judgments and limitations are disclosed.
- [ ] All paid calls use cache, usage tracking, budget reservation, retries, and console reconciliation.
- [ ] Final leaderboard, confidence intervals, Pareto analysis, and failure taxonomy answer the seven research questions.
- [ ] A fresh reviewer can reproduce tests and a mini offline experiment from the README.

## 12. Codex Execution Prompt

Give Codex this file and use the following instruction:

```text
Implement RAGBench-KR from docs/superpowers/plans/2026-08-13-ragbench-kr-implementation-plan.md.

First inspect the repository and report any conflict between the plan and existing code. Then execute exactly one numbered Task at a time using TDD. Before each Task, restate its file scope and interfaces. After each Task, run the focused tests plus the relevant global checks, inspect the diff, update the checkboxes/evidence in the plan, and stop for review. Never call a live or paid Upstage API unless I explicitly approve the displayed dry-run plan hash and projected maximum cost. Never open the sealed gold set except during Task 16 after the top-three configuration hashes have been frozen. Preserve unrelated user changes and do not commit secrets, restricted documents, or raw paid responses.
```

## 13. Implementation Evidence Log

Codex must append one row after every completed task; do not mark a task complete without command output from the current worktree.

| Task | Commit | Tests/checks | Result | Paid calls | Cost reconciled | Notes |
|---:|---|---|---|---:|---|---|
| 1 | `f7b562c`, `144ce70`, `603aeab` | Ruff, strict mypy, 4 non-live/non-gold tests, secret-guard matrix, `git diff --check` | pass | 0 | n/a | `PriceBook.from_yaml` deferred to Task 3 by operator decision. |
| 2 | `0713f1c`, `4309cb4` | Ruff, strict mypy, 8 passed/1 DB skip, offline Alembic SQL | partial | 0 | n/a | Code review approved; live PostgreSQL upgrade/downgrade/upgrade pending because Docker/PostgreSQL is unavailable locally. |
| 3 | `0b1b47f`..`5451f07` | Ruff/format, strict mypy, 63 passed/1 DB skip, respx contracts, secret guard | pass | 0 | n/a | Review approved after budget/cache/concurrency hardening; live PostgreSQL remains CI-only. |
| 4 | `50dcc5e`, `8807497` | Ruff/format, strict mypy, 78 passed/2 skips, offline preflight | partial | 0 | pending | CLI/review complete; approved live smoke and persistent SQL-cache rerun remain pending. |
| 5 | `f57966e`..`734a4d9` | Ruff, strict mypy, 108 passed/2 skips, 30 ingestion tests | partial | 0 | n/a | Framework approved; real corpus, licensing/manual inspection, freeze and public manifest pending. |
| 6 | `4a770af`..`2f81936` | Ruff, strict mypy, 131 passed/3 skips, offline Alembic SQL | partial | 0 | pending | Framework approved; real dual parse, 95% gate, paired QA and console reconciliation pending. |
| 7 | `c7c23c4`, `dea3cd3` | Ruff, strict mypy, 164 passed/3 skips, wheel asset audit | partial | 0 | n/a | Framework approved; real 14 datasets and notebook inspection pending parsed corpus. |
| 8 | — | — | pending | 0 | pending | — |
| 9 | — | — | pending | 0 | n/a | — |
| 10 | — | — | pending | 0 | pending | — |
| 11 | — | — | pending | 0 | pending | — |
| 12 | — | — | pending | 0 | n/a | — |
| 13 | — | — | pending | 0 | n/a | — |
| 14 | — | — | pending | 0 | pending | — |
| 15 | — | — | pending | 0 | pending | — |
| 16 | — | — | pending | 0 | pending | — |
| 17 | — | — | pending | 0 | n/a | — |
| 18 | — | — | pending | 0 | n/a | — |

---

**Plan status:** Ready for repository-specific review. Task 1 may begin only after Codex confirms whether the target repository is empty or already contains conventions that require path/dependency adjustments.
