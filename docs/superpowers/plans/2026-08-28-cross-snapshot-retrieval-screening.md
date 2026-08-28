# Cross-Snapshot Retrieval Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build bounded benchmark source windows, preserve reviewed evidence as a runnable development snapshot, map that evidence across all 14 chunk snapshots, and execute the fixed provider-free retrieval screen.

**Architecture:** Reuse the existing normalizer, immutable JSONL artifacts, embedding database, BM25/dense/hybrid retrievers, and screening runner. Evidence remains canonical as document/page/verbatim spans; a separate content-addressed binding dataset resolves one target chunk per span and snapshot. Dense and hybrid screening use a private cache-only query embedder, populated only by a separately approved paid command.

**Tech Stack:** Python 3.12, Pydantic 2, stdlib JSON/hash/file APIs, SQLAlchemy async, PostgreSQL/pgvector, NumPy, PyYAML, pytest, Ruff, mypy

**Spec:** `docs/superpowers/specs/2026-08-28-cross-snapshot-retrieval-screening-design.md`

## Global Constraints

- Do not add dependencies; the standard library and installed project packages cover every task.
- Keep source windows, questions, bindings, ranked hits, query vectors, and backups under `.ragbench/` with mode `0600` files and `0700` directories.
- Never load `test_gold`; normal execution accepts only an authorized `dev_auto` snapshot.
- Use exact NFKC plus whitespace normalization for evidence binding; never use fuzzy or semantic fallback.
- Missing evidence bindings remain in the retrieval-recall denominator.
- Use `retrieval-screen-v2`, `retrieval-v2`, and `retrieval-checkpoint-v2`; never reinterpret v1 files.
- Every paid command defaults to dry-run and requires `--execute`, `--confirm-paid`, and the exact displayed plan hash.
- Real retrieval execution must contain no provider object and no provider fallback.
- Preserve unrelated untracked workspace files and stage only files named by the active task.
- Run non-live, non-gold verification only unless a later step explicitly displays and receives paid approval.

## File Map

- `scripts/build_source_windows.py`: normalize complete Enhanced checkpoints into bounded immutable `SourceWindow` JSONL.
- `src/ragbench/benchmark/development.py`: validate and publish reviewed `dev_auto` question artifacts.
- `src/ragbench/benchmark/evidence.py`: exact span-to-chunk mapping and immutable binding datasets.
- `src/ragbench/embeddings/query_cache.py`: content-addressed query vectors and provider-free `QueryEmbedder`.
- `scripts/build_query_embeddings.py`: dry-run, price, approve, and populate the private query-vector cache.
- `src/ragbench/evaluation/retrieval.py`: span-level metrics with nullable missing targets.
- `src/ragbench/experiments/screening.py`: v2 run identity, persisted raw metrics, and resumable screening.
- `scripts/run_retrieval_screen.py`: real inventory loading, retriever construction, 126-run execution, shortlist, and leaderboard.

---

### Task 1: Deterministic Bounded Source Windows

**Files:**
- Create: `scripts/build_source_windows.py`
- Modify: `scripts/build_chunks.py:34-211`
- Modify: `scripts/generate_benchmark.py:74-103`
- Create: `tests/unit/benchmark/test_build_source_windows.py`

**Interfaces:**
- Consumes: complete parse-checkpoint mappings and `GenerationConfig.source_window_max_chars`.
- Produces: `build_source_windows(checkpoints, output_dir, *, max_chars, max_pages=2) -> SourceWindowBuildResult` and a CLI accepting checkpoint JSONL, output directory, and benchmark config.

- [ ] **Step 1: Export the two existing artifact helpers without changing behavior**

Rename `scripts.build_chunks._validate` to `validate_checkpoints`, rename `_write_immutable` to `write_immutable`, update internal calls, and rename `scripts.generate_benchmark._load_config` to `load_config`.

- [ ] **Step 2: Write failing source-window tests**

```python
def test_windows_are_bounded_deterministic_and_drop_existing_boilerplate(tmp_path: Path) -> None:
    checkpoints = (_standard_checkpoint(), _enhanced_checkpoint())
    first = build_source_windows(checkpoints, tmp_path / "one", max_chars=40, max_pages=2)
    second = build_source_windows(tuple(reversed(checkpoints)), tmp_path / "two", max_chars=40, max_pages=2)

    assert first.windows_path.read_bytes() == second.windows_path.read_bytes()
    rows = tuple(SourceWindow.model_validate_json(line) for line in first.windows_path.read_text().splitlines())
    assert rows
    assert all(len(row.content) <= 40 for row in rows)
    assert all(row.page_end - row.page_start < 2 for row in rows)
    assert all("반복 머리글" not in row.content for row in rows)
    assert stat.S_IMODE(first.windows_path.stat().st_mode) == 0o600


def test_oversized_single_block_fails_with_document_and_page(tmp_path: Path) -> None:
    with pytest.raises(SourceWindowBuildError, match=r"doc-enhanced.*page 2"):
        build_source_windows(
            (_standard_checkpoint(), _enhanced_checkpoint(content="가" * 41)),
            tmp_path,
            max_chars=40,
        )
```

The test module defines `_standard_checkpoint()` and `_enhanced_checkpoint(content="본문")` as complete three-page mappings with the same corpus snapshot/source hash, distinct modes, complete page mappings, provider identity fields, one repeated header on all three pages, and one paragraph on page 2. `content` replaces that paragraph only.

- [ ] **Step 3: Run the tests and verify the missing module failure**

Run: `uv run pytest tests/unit/benchmark/test_build_source_windows.py -q`

Expected: collection fails because `scripts.build_source_windows` does not exist.

- [ ] **Step 4: Implement the greedy window builder**

```python
@dataclass(frozen=True, slots=True)
class SourceWindowBuildResult:
    windows_path: Path
    metadata_path: Path
    corpus_snapshot_id: str
    parse_snapshot_id: str
    window_count: int
    document_count: int
    content_sha256: str


def build_source_windows(
    checkpoints: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    max_chars: int,
    max_pages: int = 2,
) -> SourceWindowBuildResult:
    if max_chars <= 0 or max_pages <= 0:
        raise SourceWindowBuildError("window bounds must be positive")
    corpus, by_mode, parse_snapshots = validate_checkpoints(checkpoints)
    units_by_document: dict[str, list[SourceUnit]] = {}
    for checkpoint in sorted(by_mode["enhanced"], key=lambda row: str(row["document_id"])):
        normalized = dict(checkpoint)
        normalized["parse_snapshot_id"] = parse_snapshots["enhanced"]
        units_by_document[str(checkpoint["document_id"])] = [
            SourceUnit(page=block.page, chunk_id=block.block_id, content=block.content)
            for block in normalize(normalized)
            if block.content.strip() and not block.is_boilerplate and block.block_kind != "empty_page"
        ]
    windows = _pack_windows(
        units_by_document,
        corpus_snapshot_id=corpus,
        parse_snapshot_id=parse_snapshots["enhanced"],
        max_chars=max_chars,
        max_pages=max_pages,
    )
    payload = "".join(window.model_dump_json() + "\n" for window in windows)
    content_sha256 = hashlib.sha256(payload.encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    windows_path = output_dir / f"source-windows-{content_sha256}.jsonl"
    metadata_path = output_dir / f"source-windows-{content_sha256}.metadata.json"
    write_immutable(windows_path, payload)
    write_immutable(
        metadata_path,
        _metadata_json(
            corpus,
            parse_snapshots["enhanced"],
            windows,
            content_sha256,
            windows_path.name,
        ),
    )
    return SourceWindowBuildResult(
        windows_path,
        metadata_path,
        corpus,
        parse_snapshots["enhanced"],
        len(windows),
        len(units_by_document),
        content_sha256,
    )
```

`_pack_windows` must preserve document/block order, include the newline inserted between units in the character count, flush before exceeding either bound, reject an oversized unit with document/page identity, and hash builder version, corpus, parse snapshot, document, page range, ordered block IDs, and normalized content into `window_id`.

- [ ] **Step 5: Add CLI config loading and immutable-conflict coverage**

The CLI calls `load_config(args.config)`, passes `generation.source_window_max_chars`, and prints one JSON object with exact keys `windows_path`, `metadata_path`, `corpus_snapshot_id`, `parse_snapshot_id`, `window_count`, `document_count`, and `content_sha256`. Optional `--result-json` writes those same public-safe fields through `write_immutable`. It never prints source text. Add a test that a conflicting existing artifact raises `SourceWindowBuildError` without overwriting bytes.

- [ ] **Step 6: Verify Task 1**

Run: `uv run pytest tests/unit/benchmark/test_build_source_windows.py tests/unit/chunking/test_build_chunks.py tests/unit/benchmark/test_generation.py -q`

Run: `uv run ruff check scripts/build_source_windows.py scripts/build_chunks.py scripts/generate_benchmark.py tests/unit/benchmark/test_build_source_windows.py`

Run: `uv run mypy src/ragbench`

Expected: all commands exit `0`.

- [ ] **Step 7: Commit Task 1**

```bash
git add scripts/build_source_windows.py scripts/build_chunks.py scripts/generate_benchmark.py tests/unit/benchmark/test_build_source_windows.py
git commit -m "feat: build bounded benchmark source windows"
```

---

### Task 2: Reviewed Development Question Artifact

**Files:**
- Create: `src/ragbench/benchmark/development.py`
- Create: `scripts/build_development_snapshot.py`
- Modify: `src/ragbench/benchmark/splits.py:20-171,352-411,515-588`
- Modify: `data/benchmarks/review_template.csv`
- Modify: `docs/runbooks/human-validation.md:20-45`
- Modify: `tests/unit/benchmark/test_splits.py`
- Create: `tests/unit/benchmark/test_development.py`

**Interfaces:**
- Consumes: accepted `QuestionCandidate` values, final `ReviewRecord` values, and an authorized `SplitSnapshot(name=dev_auto)`.
- Produces: `DevelopmentQuestion`, `DevelopmentQuestionSnapshot`, `build_development_snapshot`, `publish_development_snapshot`, and `load_development_snapshot`.

- [ ] **Step 1: Add stable candidate identity to review records**

Add `candidate_id` as the first `REVIEW_COLUMNS` entry and a required nonblank field on `ReviewRecord`. Group agreement and adjudication by `candidate_id`; require both records to have the same candidate ID and natural question. Keep `corrected_evidence` as a CSV string, but define it as a JSON array of `EvidenceSpan` objects in the runbook.

- [ ] **Step 2: Write failing development-snapshot tests**

```python
def test_development_snapshot_joins_by_candidate_id_and_applies_structured_correction(tmp_path: Path) -> None:
    candidate = _candidate("candidate-1", evidence_text="기존 근거")
    corrected = EvidenceSpan(text="정정 근거", document_id="doc-1", page=2, chunk_id="block-2")
    review = _review(
        "candidate-1",
        ReviewDecision.CORRECT,
        natural_question="정정된 질문?",
        corrected_evidence=json.dumps([corrected.model_dump(mode="json")], ensure_ascii=False),
    )
    split = _dev_split(("candidate-1",))

    snapshot = build_development_snapshot((candidate,), (review,), split)

    assert snapshot.questions[0].question_id == "candidate-1"
    assert snapshot.questions[0].prompt == "정정된 질문?"
    assert snapshot.questions[0].evidence_spans == (corrected,)
    path = tmp_path / "dev.jsonl"
    publish_development_snapshot(snapshot, path)
    assert load_development_snapshot(path, expected_content_hash=snapshot.content_hash) == snapshot
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_development_snapshot_rejects_free_form_correction_and_gold_membership() -> None:
    with pytest.raises(DevelopmentSnapshotError, match="corrected evidence JSON"):
        build_development_snapshot(
            (_candidate("candidate-1"),),
            (_review("candidate-1", ReviewDecision.CORRECT, corrected_evidence="2페이지 참고"),),
            _dev_split(("candidate-1",)),
        )
    with pytest.raises(PermissionError, match="dev_auto"):
        build_development_snapshot(
            (_candidate("candidate-1"),),
            (_review("candidate-1", ReviewDecision.ACCEPT),),
            _gold_split(("candidate-1",)),
        )
```

The test module's `_candidate(id, evidence_text="기존 근거")` returns a valid accepted answerable `QuestionCandidate`; `_review(id, decision, natural_question="질문?", corrected_evidence="")` returns a timezone-aware final `ReviewRecord`; `_dev_split(ids)` and `_gold_split(ids)` return internally consistent `SplitSnapshot` objects whose membership hash is `canonical_json_hash(tuple(sorted(ids)))`.

- [ ] **Step 3: Run tests and verify the missing module failure**

Run: `uv run pytest tests/unit/benchmark/test_development.py tests/unit/benchmark/test_splits.py -q`

Expected: collection fails because `ragbench.benchmark.development` does not exist.

- [ ] **Step 4: Implement immutable development models and membership validation**

```python
class DevelopmentQuestion(_FrozenModel):
    question_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    question_type: str = Field(min_length=1)
    answerable: bool
    document_cluster_id: str = Field(min_length=1)
    evidence_spans: tuple[EvidenceSpan, ...]

    @model_validator(mode="after")
    def _evidence_matches_answerability(self) -> Self:
        if self.answerable != bool(self.evidence_spans):
            raise ValueError("answerable development questions require evidence")
        return self


class DevelopmentQuestionSnapshot(_FrozenModel):
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    membership_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    questions: tuple[DevelopmentQuestion, ...] = Field(min_length=1)
```

`build_development_snapshot` must require exactly one non-rejected final review per split member, no extra review/candidate IDs, accepted candidates to retain generated spans, corrected candidates to parse and validate the JSON evidence, and unanswerable questions to retain zero spans. Derive `document_cluster_id` from the sorted evidence document IDs, or from the unanswerable transform target document when no evidence exists.

- [ ] **Step 5: Publish and load the owner-only artifact**

Rename `splits._publish_private_immutable` and `_read_private_regular` to public `publish_private_immutable` and `read_private_regular`, update gold callers, and reuse them for the development file. The file format is one metadata header JSON object followed by sorted question JSON objects; the header contains schema `development-questions-v1`, split snapshot ID, membership hash, content hash, and item count. The loader verifies file mode, hash, count, unique IDs, and exact membership before returning content.

- [ ] **Step 6: Add the materialization CLI and review protocol**

`scripts/build_development_snapshot.py` accepts private candidate JSONL, final-review JSONL, and private split JSON, writes one immutable development JSONL, and prints metadata without prompts or evidence. Update the header-only CSV and runbook to specify `candidate_id` and JSON evidence corrections.

- [ ] **Step 7: Verify Task 2**

Run: `uv run pytest tests/unit/benchmark/test_splits.py tests/unit/benchmark/test_development.py -q`

Run: `uv run ruff check src/ragbench/benchmark/splits.py src/ragbench/benchmark/development.py scripts/build_development_snapshot.py tests/unit/benchmark/test_development.py`

Run: `uv run mypy src/ragbench`

Expected: all commands exit `0`.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/ragbench/benchmark/development.py src/ragbench/benchmark/splits.py scripts/build_development_snapshot.py data/benchmarks/review_template.csv docs/runbooks/human-validation.md tests/unit/benchmark/test_splits.py tests/unit/benchmark/test_development.py
git commit -m "feat: materialize reviewed development questions"
```

---

### Task 3: Exact Cross-Snapshot Evidence Bindings

**Files:**
- Create: `src/ragbench/benchmark/evidence.py`
- Create: `scripts/build_evidence_bindings.py`
- Modify: `src/ragbench/benchmark/generation.py:858-885,932-934`
- Modify: `src/ragbench/benchmark/validation.py:286-306`
- Create: `tests/unit/benchmark/test_evidence_bindings.py`

**Interfaces:**
- Consumes: `DevelopmentQuestionSnapshot`, target `ChunkRecord` rows, and target chunk-snapshot ID.
- Produces: `EvidenceBinding`, `EvidenceBindingDataset`, `bind_evidence`, `publish_bindings`, and `load_bindings`.

- [ ] **Step 1: Write failing exact-binding tests**

```python
def test_binding_filters_document_and_page_then_chooses_smallest_exact_container() -> None:
    span = EvidenceSpan(text="매출액은 1,234억 원", document_id="doc-1", page=3, chunk_id="source-block")
    snapshot = _development_snapshot(span)
    chunks = (
        _chunk("wrong-document", "doc-2", 3, 3, "매출액은 1,234억 원"),
        _chunk("wrong-page", "doc-1", 4, 4, "매출액은 1,234억 원"),
        _chunk("large", "doc-1", 2, 3, "앞 문장 매출액은 1,234억 원 뒤 문장"),
        _chunk("small", "doc-1", 3, 3, "매출액은   1,234억 원"),
    )

    dataset = bind_evidence(snapshot, "chunk-snapshot-a", chunks)

    assert dataset.records[0].status == "exact"
    assert dataset.records[0].target_chunk_id == "small"
    assert dataset.records[0].candidate_count == 2


def test_missing_binding_is_retained_and_punctuation_is_not_fuzzy_matched() -> None:
    span = EvidenceSpan(text="영업이익: 10억", document_id="doc-1", page=1, chunk_id="source")
    dataset = bind_evidence(
        _development_snapshot(span),
        "chunk-snapshot-a",
        (_chunk("punctuation-changed", "doc-1", 1, 1, "영업이익 10억"),),
    )

    assert dataset.records[0].status == "missing"
    assert dataset.records[0].target_chunk_id is None
    assert dataset.records[0].candidate_count == 0
```

The test module's `_development_snapshot(span)` creates one answerable question whose content/membership hashes are recomputed from that question. `_chunk(id, document, page_start, page_end, content)` returns a valid `ChunkRecord` using one fixed parse snapshot and strategy, positive token count, and stable remaining identity fields.

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `uv run pytest tests/unit/benchmark/test_evidence_bindings.py -q`

Expected: collection fails because `ragbench.benchmark.evidence` does not exist.

- [ ] **Step 3: Add one shared exact-evidence normalizer**

Add the following public helper to `benchmark/generation.py`, require `_validate_candidate_window` to use it for assigned source units, and make automatic validation reject a candidate before fuzzy-quality checks when exact containment fails:

```python
def normalize_evidence_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())
```

Add a generation test showing `"영업이익: 10억"` is not accepted from source text `"영업이익 10억"`.

- [ ] **Step 4: Implement binding records and exact normalization**

```python
BINDING_ALGORITHM_VERSION = "exact-nfkc-whitespace-v1"


class EvidenceBinding(_FrozenModel):
    evidence_span_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_id: str = Field(min_length=1)
    span_index: int = Field(ge=0)
    target_chunk_snapshot_id: str = Field(min_length=1)
    target_chunk_id: str | None
    status: Literal["exact", "missing"]
    candidate_count: int = Field(ge=0)
    algorithm_version: Literal["exact-nfkc-whitespace-v1"]
```

For each span, hash question ID, zero-based ordinal, and the full `EvidenceSpan.model_dump()` into `evidence_span_id`. Candidate chunks must match document ID, contain the evidence page, and contain the normalized span exactly. Rank candidates by `(len(normalized_chunk) - len(normalized_span), chunk_id)` and keep one target.

- [ ] **Step 5: Implement and validate the binding dataset**

`EvidenceBindingDataset` stores schema `evidence-bindings-v1`, development content hash, target snapshot ID, algorithm version, ordered records, and a content hash. It must contain exactly one record for every evidence span, preserve two spans that resolve to one chunk, reject duplicate/extra records, and contain no records for unanswerable questions.

- [ ] **Step 6: Add immutable JSONL I/O and CLI**

The CLI accepts a verified development JSONL, one target chunk JSONL, target chunk-snapshot ID, and output directory. It parses chunks with `ChunkRecord(**row)`, rejects mixed parse modes/strategies and duplicate chunk IDs, writes a `0600` content-addressed binding file, and prints only counts/hashes including exact and missing totals.

- [ ] **Step 7: Verify Task 3**

Run: `uv run pytest tests/unit/benchmark/test_evidence_bindings.py tests/unit/benchmark/test_generation.py tests/unit/benchmark/test_validation.py -q`

Run: `uv run ruff check src/ragbench/benchmark/evidence.py src/ragbench/benchmark/generation.py src/ragbench/benchmark/validation.py scripts/build_evidence_bindings.py tests/unit/benchmark/test_evidence_bindings.py`

Run: `uv run mypy src/ragbench`

Expected: all commands exit `0`.

- [ ] **Step 8: Commit Task 3**

```bash
git add src/ragbench/benchmark/evidence.py src/ragbench/benchmark/generation.py src/ragbench/benchmark/validation.py scripts/build_evidence_bindings.py tests/unit/benchmark/test_evidence_bindings.py tests/unit/benchmark/test_generation.py tests/unit/benchmark/test_validation.py
git commit -m "feat: bind evidence across chunk snapshots"
```

---

### Task 4: Span-Level Retrieval Metrics and V2 Configuration

**Files:**
- Modify: `src/ragbench/evaluation/retrieval.py:10-212`
- Modify: `src/ragbench/experiments/config.py:35-105`
- Modify: `src/ragbench/experiments/planner.py:70-118`
- Modify: `src/ragbench/offline.py:140-170`
- Modify: `scripts/run_retrieval_screen.py:83-100`
- Modify: `tests/unit/evaluation/test_retrieval_metrics.py`
- Modify: `tests/unit/evaluation/test_experiment_config.py`
- Modify: `tests/unit/evaluation/test_retrieval_screen.py`
- Modify: `tests/unit/evaluation/test_run_retrieval_screen.py`
- Modify: `tests/unit/test_offline_fixture.py`

**Interfaces:**
- Consumes: ranked chunk IDs and one nullable target chunk ID per reviewed span.
- Produces: `RetrievalCase.evidence_targets: tuple[str | None, ...]` and v2 semantic configuration hashes.

- [ ] **Step 1: Replace chunk-set tests with span-target tests**

```python
def test_recall_counts_missing_and_duplicate_chunk_targets_as_distinct_spans() -> None:
    case = RetrievalCase(
        question_id="q1",
        question_type="fact",
        ranked_chunk_ids=("shared", "noise"),
        evidence_targets=("shared", "shared", None),
        latency_ms=4.0,
        document_cluster_id="doc-1",
    )

    metric = evaluate_retrieval(case, k=2)

    assert metric.evidence_count == 3
    assert metric.retrieved_evidence_count == 2
    assert metric.hit_at_k == 1.0
    assert metric.evidence_recall_at_k == pytest.approx(2 / 3)
    assert metric.mrr == 1.0


def test_all_missing_answerable_evidence_scores_zero() -> None:
    metric = evaluate_retrieval(
        RetrievalCase("q1", "fact", ("noise",), (None, None), 1.0, "doc-1"),
        k=1,
    )
    assert (metric.hit_at_k, metric.evidence_recall_at_k, metric.mrr) == (0.0, 0.0, 0.0)
```

- [ ] **Step 2: Run focused tests and verify old set behavior fails**

Run: `uv run pytest tests/unit/evaluation/test_retrieval_metrics.py -q`

Expected: failures show that `RetrievalCase` lacks `evidence_targets` and rejects duplicate targets.

- [ ] **Step 3: Implement span-aware scoring**

```python
@dataclass(frozen=True, slots=True)
class RetrievalCase:
    question_id: str
    question_type: str
    ranked_chunk_ids: tuple[str, ...]
    evidence_targets: tuple[str | None, ...]
    latency_ms: float
    document_cluster_id: str | None = None


def evaluate_retrieval(case: RetrievalCase, *, k: int) -> RetrievalMetric:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not case.evidence_targets:
        return RetrievalMetric(
            question_id=case.question_id,
            question_type=case.question_type,
            document_cluster_id=case.document_cluster_id,
            k=k,
            evidence_count=0,
            retrieved_evidence_count=0,
            hit_at_k=None,
            evidence_recall_at_k=None,
            mrr=None,
            latency_ms=case.latency_ms,
        )
    top = set(case.ranked_chunk_ids[:k])
    retrieved = sum(target is not None and target in top for target in case.evidence_targets)
    ranks = {
        chunk_id: rank for rank, chunk_id in enumerate(case.ranked_chunk_ids, start=1)
    }
    first_rank = min(
        (ranks[target] for target in case.evidence_targets if target is not None and target in ranks),
        default=None,
    )
    return RetrievalMetric(
        question_id=case.question_id,
        question_type=case.question_type,
        document_cluster_id=case.document_cluster_id,
        k=k,
        evidence_count=len(case.evidence_targets),
        retrieved_evidence_count=retrieved,
        hit_at_k=float(retrieved > 0),
        evidence_recall_at_k=retrieved / len(case.evidence_targets),
        mrr=0.0 if first_rank is None else 1.0 / first_rank,
        latency_ms=case.latency_ms,
    )
```

Carry `document_cluster_id` into `RetrievalMetric`, `BootstrapInput`, and `PairedBootstrapInput` so later paired document-cluster resampling remains possible. Keep empty evidence targets explicitly unscored.

- [ ] **Step 4: Migrate the configuration identity to v2**

Change `RetrievalExperimentConfig.schema_version` to `Literal["retrieval-screen-v2"]`, `metric_version` to `Literal["retrieval-v2"]`, and update `generate_core_retrieval_configs`. Update the dry-run payload and every existing screening test fixture to v2, then refresh fixed semantic-hash assertions from generated output; do not accept v1 values in the new model.

- [ ] **Step 5: Update offline fixtures and callers**

Convert existing fixture `evidence_chunk_ids` to `tuple[str | None, ...]` only at the `RetrievalCase` construction boundary. Keep public fixture YAML unchanged so offline examples remain readable and backward compatible.

- [ ] **Step 6: Verify Task 4**

Run: `uv run pytest tests/unit/evaluation/test_retrieval_metrics.py tests/unit/evaluation/test_experiment_config.py tests/unit/evaluation/test_retrieval_screen.py tests/unit/evaluation/test_run_retrieval_screen.py tests/unit/test_offline_fixture.py -q`

Run: `uv run ruff check src/ragbench/evaluation/retrieval.py src/ragbench/experiments/config.py src/ragbench/experiments/planner.py src/ragbench/offline.py`

Run: `uv run mypy src/ragbench`

Expected: all commands exit `0` and the planner still produces exactly 126 unique configs.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/ragbench/evaluation/retrieval.py src/ragbench/experiments/config.py src/ragbench/experiments/planner.py src/ragbench/offline.py scripts/run_retrieval_screen.py tests/unit/evaluation/test_retrieval_metrics.py tests/unit/evaluation/test_experiment_config.py tests/unit/evaluation/test_retrieval_screen.py tests/unit/evaluation/test_run_retrieval_screen.py tests/unit/test_offline_fixture.py
git commit -m "feat: score retrieval by canonical evidence span"
```

---

### Task 5: Gated Query-Vector Cache

**Files:**
- Create: `src/ragbench/embeddings/query_cache.py`
- Modify: `src/ragbench/embeddings/service.py:20-146`
- Create: `scripts/build_query_embeddings.py`
- Create: `tests/unit/embeddings/test_query_cache.py`
- Create: `tests/unit/test_build_query_embeddings.py`

**Interfaces:**
- Consumes: exact development prompts, complete `EmbeddingSnapshot` metadata, the existing metered gateway, and the pinned tokenizer.
- Produces: `QueryVectorRecord`, `QueryEmbeddingPlan`, `CachedQueryEmbedder`, `plan_query_embeddings`, and an explicitly gated cache-builder CLI.

- [ ] **Step 1: Write failing cache-only embedder tests**

```python
@pytest.mark.asyncio
async def test_cached_query_embedder_returns_exact_vector_without_provider() -> None:
    record = _record(text="질문", model_id="embedding-query", dimension=3, vector=(1.0, 0.0, 0.0))
    embedder = CachedQueryEmbedder((record,), {"snapshot-a": _snapshot(dimension=3)})

    vector = await embedder.embed_query("질문", snapshot_id="snapshot-a", input_tokens=2)

    assert vector == (1.0, 0.0, 0.0)


@pytest.mark.asyncio
async def test_cached_query_embedder_never_falls_back_on_missing_or_mismatched_identity() -> None:
    embedder = CachedQueryEmbedder((), {"snapshot-a": _snapshot(dimension=3)})
    with pytest.raises(QueryCacheError, match="missing"):
        await embedder.embed_query("질문", snapshot_id="snapshot-a", input_tokens=2)
```

The test module's `_snapshot(snapshot_id="snapshot-a", dimension=3)` returns a complete ready L2 `EmbeddingSnapshot` with query model `embedding-query`; `_record(text="질문", model_id="embedding-query", dimension=3, vector=(1.0, 0.0, 0.0))` hashes the exact text and empty provider parameters into a valid `QueryVectorRecord`.

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `uv run pytest tests/unit/embeddings/test_query_cache.py -q`

Expected: collection fails because `ragbench.embeddings.query_cache` does not exist.

- [ ] **Step 3: Implement content-addressed cache records**

```python
@dataclass(frozen=True, slots=True)
class QueryVectorRecord:
    question_sha256: str
    query_model_id: str
    dimension: int
    provider_params_hash: str
    input_tokens: int
    vector: tuple[float, ...]

    @property
    def key(self) -> str:
        return canonical_json_hash(
            {
                "question_sha256": self.question_sha256,
                "query_model_id": self.query_model_id,
                "dimension": self.dimension,
                "provider_params_hash": self.provider_params_hash,
            }
        )
```

`CachedQueryEmbedder` receives records, a snapshot-ID mapping, and the exact provider-parameter mapping used by the cache, hashes the exact UTF-8 prompt, verifies model, dimension, provider-parameter hash, token count, finite values, and unit norm, then returns the vector. It has no gateway field and raises on every miss. Default provider parameters are `{}`; `{"input_type": "query"}` is used only when the cache-builder receives `--supports-input-type`.

- [ ] **Step 4: Add batched query embedding to the existing service**

Add `EmbeddingService.embed_queries(snapshot_id, queries: Sequence[tuple[str, int]]) -> tuple[tuple[float, ...], ...]`. It validates the complete snapshot once, batches by existing item/token limits, calls `EmbedRequest` with query mode, and reuses `_validate_response`. Preserve `embed_query` by delegating a one-item tuple to this method.

- [ ] **Step 5: Write the dry-run and paid-gate tests**

```python
def test_query_plan_deduplicates_shared_model_snapshots_and_prices_only_misses() -> None:
    plan = plan_query_embeddings(
        _questions("같은 질문", "다른 질문"),
        (_snapshot(snapshot_id="a"), _snapshot(snapshot_id="b")),
        existing_records=(_record(text="같은 질문"),),
        price_book=_price_book(),
        billing_multiplier=Decimal("1.1"),
        token_counter=lambda text: 10,
    )
    assert plan.unique_model_groups == 1
    assert plan.cache_hits == 1
    assert plan.new_vectors == 1
    assert plan.maximum_cost_usd == Decimal("0.001100")


def test_execute_requires_exact_displayed_plan_hash() -> None:
    with pytest.raises(QueryCacheError, match="plan hash"):
        require_query_execution_gate(
            execute=True,
            confirm_paid=True,
            confirmed_plan_hash="0" * 64,
            required_plan_hash="1" * 64,
            live_enabled=True,
        )
```

`_questions(*prompts)` creates valid `DevelopmentQuestion` values with those exact prompts. `_price_book()` uses an embedding input rate of USD 100 per million tokens, a fresh verification timestamp, and no promotion discount, so one ten-token cache miss with multiplier `1.1` costs exactly USD `0.001100`. Add `require_query_execution_gate` to the produced interfaces; it returns normally only when all four gates and the exact plan hash match.

- [ ] **Step 6: Implement the cache-builder CLI**

The default output contains plan hash, question count, unique model groups, total input tokens, cache hits, new vectors, current price snapshot timestamp, operator-supplied provider balance, VAT multiplier, and maximum cost. Execution requires `--execute --confirm-paid`, `--confirm-plan` equal to the displayed plan hash, `--provider-balance-usd`, `RUN_LIVE_UPSTAGE_TESTS=1`, a fresh price book, available local and provider balances, and an API key. All calls pass through `UpstageGateway`; successful vectors are written as one `0600` immutable content-addressed JSONL file.

- [ ] **Step 7: Verify Task 5**

Run: `uv run pytest tests/unit/embeddings/test_query_cache.py tests/unit/test_build_query_embeddings.py tests/unit/embeddings/test_service.py -q`

Run: `uv run ruff check src/ragbench/embeddings/query_cache.py src/ragbench/embeddings/service.py scripts/build_query_embeddings.py tests/unit/embeddings/test_query_cache.py tests/unit/test_build_query_embeddings.py`

Run: `uv run mypy src/ragbench`

Expected: all commands exit `0`; fake gateways observe batched query mode and no live test runs.

- [ ] **Step 8: Commit Task 5**

```bash
git add src/ragbench/embeddings/query_cache.py src/ragbench/embeddings/service.py scripts/build_query_embeddings.py tests/unit/embeddings/test_query_cache.py tests/unit/test_build_query_embeddings.py
git commit -m "feat: cache gated query embeddings"
```

---

### Task 6: V2 Resumable Screening Core

**Files:**
- Modify: `src/ragbench/experiments/screening.py:20-508`
- Modify: `tests/unit/evaluation/test_retrieval_screen.py`

**Interfaces:**
- Consumes: `DevelopmentQuestionSnapshot`, matching `EvidenceBindingDataset`, `RetrievalExperimentConfig`, and `BoundRetriever`.
- Produces: v2 `RetrievalScreenRunner`, `ScreeningRunRecord`, and durable `FileScreeningStore` records containing hits, latency, raw metrics, and aggregate evaluation.

- [ ] **Step 1: Rewrite runner tests around development questions and bindings**

```python
@pytest.mark.asyncio
async def test_screen_counts_missing_binding_and_persists_final_evaluation(tmp_path: Path) -> None:
    questions = _development_snapshot(two_spans=True)
    bindings = _bindings_for_questions(questions, targets=("c1", None))
    store = FileScreeningStore(tmp_path / "runs")
    runner = RetrievalScreenRunner(
        store=store,
        inventory=_inventory(),
        development_authorization=_authorization(questions),
    )

    result = await runner.run(_config(), questions, bindings, _bound(FakeRetriever()))
    reloaded = FileScreeningStore(tmp_path / "runs").get(result.run_id)

    assert reloaded.evaluation is not None
    assert reloaded.evaluation.overall.micro_evidence_recall_at_k == pytest.approx(0.5)
    assert reloaded.binding_dataset_hash == bindings.content_hash
    assert reloaded.metric_version == "retrieval-v2"
```

Rename the existing test `_snapshot` helper to `_development_snapshot`; it returns the same two prompts as valid `DevelopmentQuestion` values and recomputes content/membership hashes. `_bindings_for_questions` emits one ordered record per span for the configured chunk snapshot, `_authorization` wraps the matching `dev_auto` `SplitSnapshot`, and the existing `_inventory`, `_config`, `_bound`, and `FakeRetriever` helpers retain their current identities while moving to v2.

- [ ] **Step 2: Run tests and verify v1 runner incompatibility**

Run: `uv run pytest tests/unit/evaluation/test_retrieval_screen.py -q`

Expected: failures show the runner does not accept binding datasets and the file store drops final evaluation.

- [ ] **Step 3: Replace portable chunk IDs with binding lookup**

Remove `ScreeningQuestion` and `ScreeningQuestionSnapshot`. The runner iterates `DevelopmentQuestionSnapshot.questions`, obtains the ordered nullable target tuple from `EvidenceBindingDataset`, searches only the prompt, and creates `RetrievalCase` with the question's document cluster ID.

- [ ] **Step 4: Extend immutable run identity and collision checks**

```python
def run_identity(
    self,
    config: RetrievalExperimentConfig,
    questions: DevelopmentQuestionSnapshot,
    bindings: EvidenceBindingDataset,
) -> str:
    return canonical_json_hash(
        {
            "kind": "retrieval-screen-run-v2",
            "config_hash": config.semantic_hash,
            "question_snapshot_id": questions.snapshot_id,
            "question_content_hash": questions.content_hash,
            "binding_dataset_hash": bindings.content_hash,
            "metric_version": config.metric_version,
        }
    )
```

Add `binding_dataset_hash` and `metric_version` to `ScreeningRunRecord`, compare them in `begin`, and reject a binding target that differs from `config.chunk_snapshot_id`.

- [ ] **Step 5: Persist and restore the complete evaluation**

Change file schema to `retrieval-checkpoint-v2`. `_flush` serializes `evaluation` with `asdict` when present. `_load` reconstructs `RetrievalMetric`, `RetrievalAggregate`, `BootstrapInput`, and `RetrievalEvaluation`; it rejects complete records without evaluation, running/interrupted records with evaluation, inconsistent question IDs, nonfinite metrics, or an integrity-hash mismatch.

- [ ] **Step 6: Preserve atomic resume behavior**

Keep one atomic replace after each completed question. Resume skips only completed question IDs under the same config/question/binding/metric identity. On an exception, persist `interrupted`; on completion, persist v2 raw and aggregate metrics before returning.

- [ ] **Step 7: Verify Task 6**

Run: `uv run pytest tests/unit/evaluation/test_retrieval_screen.py tests/contract/experiments/test_resume.py -q`

Run: `uv run ruff check src/ragbench/experiments/screening.py tests/unit/evaluation/test_retrieval_screen.py`

Run: `uv run mypy src/ragbench`

Expected: all commands exit `0`, including process-lifetime resume and corrupted-checkpoint rejection.

- [ ] **Step 8: Commit Task 6**

```bash
git add src/ragbench/experiments/screening.py tests/unit/evaluation/test_retrieval_screen.py tests/contract/experiments/test_resume.py
git commit -m "feat: persist v2 retrieval screening evidence"
```

---

### Task 7: Real Provider-Free 126-Configuration Runner

**Files:**
- Modify: `scripts/run_retrieval_screen.py:1-105`
- Modify: `tests/unit/evaluation/test_run_retrieval_screen.py`
- Modify: `src/ragbench/experiments/selection.py:198-225`
- Modify: `tests/unit/evaluation/test_retrieval_selection.py`

**Interfaces:**
- Consumes: private v2 inventory YAML, development JSONL, 14 binding JSONL files, optional query-vector cache, PostgreSQL embedding snapshots, and code commit.
- Produces: `execute_retrieval_grid`, 126 durable run checkpoints, the existing top-eight shortlist, and a private leaderboard whose rows reference persisted bootstrap inputs.

- [ ] **Step 1: Expand inventory tests to bind immutable chunk files**

```python
def test_execute_rejects_chunk_hash_mismatch_before_search(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, corrupt_hash=True)
    with pytest.raises(SystemExit, match="chunk dataset hash"):
        SCRIPT.main(_execute_args(tmp_path, inventory))


def test_full_execution_requires_query_cache_but_bm25_only_does_not(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="query-vector cache"):
        SCRIPT.main(_execute_args(tmp_path, _inventory(tmp_path)))
    assert SCRIPT.main([*_execute_args(tmp_path, _inventory(tmp_path)), "--bm25-only"]) == 0
```

Extend the existing `_inventory(tmp_path)` helper to write all 14 tiny chunk/binding files and their real SHA-256 values; `corrupt_hash=True` replaces exactly one declared chunk hash. `_execute_args(tmp_path, inventory)` returns all required v2 execute arguments, including a synthetic authorized development artifact, checkpoint/output roots, corpus ID, code commit, and the supplied inventory path, while intentionally omitting the query cache.

- [ ] **Step 2: Run tests and verify the deliberate execution blocker fails them**

Run: `uv run pytest tests/unit/evaluation/test_run_retrieval_screen.py -q`

Expected: failures reach the existing `real retrieval screening is not available` exit.

- [ ] **Step 3: Validate the private inventory and artifacts**

Each binding row must include parse mode/snapshot, chunk strategy/snapshot, embedding snapshot UUID, chunk dataset path, chunk dataset SHA-256, expected chunk count, and binding dataset path/hash. Require the exact 14 mode/strategy pairs, one corpus ID, regular non-symlink files, matching hashes/counts, unique chunk IDs, and binding target IDs equal to the declared chunk snapshot.

- [ ] **Step 4: Build and reuse concrete retrievers**

```python
def build_retrievers(
    row: RuntimeBinding,
    *,
    repository: SqlAlchemyEmbeddingRepository,
    query_embedder: CachedQueryEmbedder | None,
) -> dict[RetrieverName, BoundRetriever]:
    search_filter = SearchFilter(
        row.corpus_snapshot_id,
        row.parse_snapshot_id,
        row.chunk_strategy,
        row.embedding_snapshot_id,
    )
    sparse = BM25Retriever(
        BM25IndexSnapshot(
            search_filter,
            tuple(BM25Document(chunk.chunk_id, chunk.document_id, chunk.content) for chunk in row.chunks),
        )
    )
    output: dict[RetrieverName, BoundRetriever] = {
        "bm25": BoundRetriever("bm25", sparse, None)
    }
    if query_embedder is not None:
        dense = DenseRetriever(
            query_embedder,
            repository,
            token_counter=lambda text: len(encoding().encode(text)),
        )
        output["dense"] = BoundRetriever("dense", dense, None)
        output["hybrid"] = BoundRetriever(
            "hybrid",
            HybridRetriever(dense, sparse),
            RRFConfig(),
        )
    return output
```

Construct each BM25/dense/hybrid family once per snapshot and reuse it across K values. Verify all 14 database embedding snapshots are complete, ready, dimension-consistent with the cache, and lineage-consistent before the first question runs.

- [ ] **Step 5: Execute the fixed grid through the existing runner**

Convert `main` to call an async `execute_retrieval_grid(...)` that receives the validated runtime inventory, development snapshot, screening store, embedding repository, and optional cache-only query embedder. The CLI constructs the PostgreSQL repository; tests pass a deterministic in-memory fake. Full mode runs all 126 configs; `--bm25-only` runs the 42 BM25 configs and does not shortlist. Use one `FileScreeningStore` root, one authorized development snapshot, the matching binding dataset for each chunk snapshot, and the current code commit in every config. The command contains no gateway construction and rejects `--execute` if any required identity is missing.

- [ ] **Step 6: Export outcomes without leaking questions or mislabeling K**

Rename `ScreeningOutcome.hit_at_5`, `recall_at_5`, and `micro_recall_at_5` to `hit_at_k`, `recall_at_k`, and `micro_recall_at_k`. Build one outcome from each of the 126 evaluations, keep the frozen selection rule restricted to `config.top_k == 5`, and call `select_retrieval_shortlist(..., size=8)`. Update leaderboard schema to `retrieval-leaderboard-v2` so every row includes its actual K, metric version, binding hash, run ID, per-type metrics, and bootstrap-input hash; add the eight shortlist config hashes at the root. It must not contain prompts, evidence text, or vectors and must continue stating that confidence intervals are not computed in this stage.

- [ ] **Step 7: Verify Task 7**

Run: `uv run pytest tests/unit/evaluation/test_run_retrieval_screen.py tests/unit/evaluation/test_retrieval_selection.py -q`

Run: `uv run ruff check scripts/run_retrieval_screen.py src/ragbench/experiments/selection.py tests/unit/evaluation/test_run_retrieval_screen.py`

Run: `uv run mypy src/ragbench`

Expected: all commands exit `0`; fake full execution creates 126 results, fake BM25-only execution creates 42, and provider-call count remains zero.

- [ ] **Step 8: Commit Task 7**

```bash
git add scripts/run_retrieval_screen.py src/ragbench/experiments/selection.py tests/unit/evaluation/test_run_retrieval_screen.py tests/unit/evaluation/test_retrieval_selection.py
git commit -m "feat: run real provider-free retrieval screening"
```

---

### Task 8: Generation Preflight, Full Verification, and First Free Corpus Operation

**Files:**
- Modify: `scripts/generate_benchmark.py:176-250`
- Create: `tests/unit/benchmark/test_generate_benchmark_script.py`
- Modify: `2026-08-13-ragbench-kr-implementation-plan.md:350-390`
- Modify: `docs/reports/offline-operations.md:108-120`

**Interfaces:**
- Consumes: all implementation tasks and the complete private parse-checkpoint export.
- Produces: verified source-window artifacts and an exact benchmark-generation dry-run containing hashes and maximum cost; no paid request.

- [ ] **Step 1: Write a failing dry-run price test**

```python
def test_generation_dry_run_displays_exact_maximum_cost_without_gateway(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = SCRIPT.run_preflight(
        windows=_windows(),
        generation_config=GenerationConfig(batch_size=2),
        validation_config=_validation_config(),
        corpus_snapshot_id="corpus-a",
        model_id="solar-pro3",
        price_book=_fresh_price_book(),
        billing_multiplier=Decimal("1.1"),
        max_replacement_rounds=2,
        allow_reduced_scope=False,
    )

    assert result["live_executed"] is False
    assert result["initial_plan_cost_usd"] == "0.000110"
    assert result["maximum_cost_usd"] == "0.000330"
    assert len(result["plan_hash"]) == 64
    assert len(result["campaign_hash"]) == 64
```

The test module reuses two valid `SourceWindow` objects from generation tests through a local `_windows()` factory, returns `ValidationConfig(quotas=GenerationConfig(batch_size=2).quotas)` from `_validation_config()`, and creates a fresh `PriceBook` fixture whose generation rates make the initial two-batch plan cost exactly USD `0.000110` after the `1.1` multiplier.

- [ ] **Step 2: Run the test and verify cost fields are absent**

Run: `uv run pytest tests/unit/benchmark/test_generate_benchmark_script.py -q`

Expected: failure shows `run_preflight` or the two exact cost fields do not exist.

- [ ] **Step 3: Refactor one shared preflight path**

Add `run_preflight(...) -> dict[str, object]` that creates the existing plan/campaign identity, calls `PriceBook.verify_paid_batch`, computes `projected_generation_cost`, applies the configured billing multiplier exactly once, and returns initial and maximum cost as six-decimal strings. Both dry-run output and paid execution use this same object; paid execution must reject a recomputed identity mismatch.

- [ ] **Step 4: Verify the generation preflight**

Run: `uv run pytest tests/unit/benchmark/test_generate_benchmark_script.py tests/unit/benchmark/test_generation.py -q`

Expected: all tests pass and fake dry-run code constructs no gateway.

- [ ] **Step 5: Run the full non-live, non-gold suite**

Run: `uv run pytest -m "not live and not gold" -q`

Expected: zero failures.

- [ ] **Step 6: Run static and repository checks**

Run: `uv run ruff check .`

Run: `uv run mypy src/ragbench`

Run: `git diff --check`

Expected: all commands exit `0`.

- [ ] **Step 7: Build private source windows without provider calls**

Run:

```bash
uv run python scripts/build_source_windows.py .ragbench/exports/parse-checkpoints-complete-2026-08-28.jsonl .ragbench/benchmarks/source-windows --config configs/benchmark.yaml --result-json .ragbench/benchmarks/source-window-build-result.json
```

Expected: JSON containing one Enhanced parse snapshot, nonzero document/window counts, artifact hashes, and no question/source text.

- [ ] **Step 8: Verify deterministic replay**

Run the identical command again with the same immutable result path:

```bash
uv run python scripts/build_source_windows.py .ragbench/exports/parse-checkpoints-complete-2026-08-28.jsonl .ragbench/benchmarks/source-windows --config configs/benchmark.yaml --result-json .ragbench/benchmarks/source-window-build-result.json
```

Expected: identical paths and hashes, no overwrite, and no provider usage row.

- [ ] **Step 9: Run the benchmark-generation dry-run from verified metadata**

Resolve the exact values directly from the deterministic builder result:

```bash
uv run python -c 'import json,subprocess; from pathlib import Path; row=json.loads(Path(".ragbench/benchmarks/source-window-build-result.json").read_text()); raise SystemExit(subprocess.run(["uv", "run", "python", "scripts/generate_benchmark.py", row["windows_path"], "--corpus-snapshot-id", row["corpus_snapshot_id"], "--config", "configs/benchmark.yaml"]).returncode)'
```

This command remains dry-run because it omits `--execute`; it prints candidate target, batch count, plan hash, campaign hash, verified price timestamp, initial cost, maximum cost, and `live_executed: false`.

- [ ] **Step 10: Stop at the paid gate**

Run `uv run ragbench usage status --json`, then record its settled/remaining values beside the exact dry-run plan/campaign hashes, user-reported provider-console balance, cache count, request count, token bounds, price timestamp, VAT multiplier, and maximum generation cost. Do not add `--execute`, `--confirm-paid`, or `--confirm-plan` until the user explicitly approves that displayed amount and hash.

- [ ] **Step 11: Update public-safe evidence and commit**

Mark only the implemented code portions of Tasks 11 and 13 complete; keep real candidate generation, human review, query-vector precomputation, and real screening unchecked. Record only counts/hashes in `docs/reports/offline-operations.md`.

```bash
git add scripts/generate_benchmark.py tests/unit/benchmark/test_generate_benchmark_script.py 2026-08-13-ragbench-kr-implementation-plan.md docs/reports/offline-operations.md
git commit -m "docs: record retrieval screening preflight"
```

- [ ] **Step 12: Push reviewed commits**

Run: `git push origin main`

Expected: GitHub `main` advances through the Task 8 documentation commit without private artifacts.
