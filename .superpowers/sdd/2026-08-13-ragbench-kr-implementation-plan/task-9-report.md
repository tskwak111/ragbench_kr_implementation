# Task 9 report — BM25 and RRF hybrid retrieval

## Implementation

- Added a dependency-free Korean lexical baseline tokenizer. It applies Unicode NFKC
  normalization and case folding, preserves comma-grouped and decimal numeric strings, splits
  conservatively at punctuation/whitespace, and intentionally does not use a morphological
  analyzer. Repeated query terms are de-duplicated so repeated wording cannot multiply identical
  lexical evidence.
- Added immutable `BM25Document` and `BM25IndexSnapshot` inputs plus `BM25Retriever`. The retriever
  implements standard Okapi BM25 with explicit validated `k1=1.2` and `b=0.75`, rejects chunk-ID
  ambiguity, validates the corpus/parse/chunk/embedding snapshot identity, applies canonical
  document filters, returns no arbitrary hits for an empty query, and breaks equal scores by chunk
  ID.
- Added deterministic `reciprocal_rank_fusion`. It implements `weight / (k + rank)` with one-based
  sequential ranks, includes the union of candidates, validates weights and `k`, rejects duplicate
  chunk IDs inside one component ranking, and breaks fused-score ties by chunk ID.
- Added `HybridRetriever`. It passes the exact same `SearchFilter` instance to dense and sparse
  branches, over-fetches each branch by exactly `max(20, 4 * top_k)`, fuses with RRF, and returns a
  stable top K. Each hybrid hit carries dense/sparse ranks, dense/sparse component scores, and its
  fused score in immutable evidence.
- Extended the stable `SearchHit` contract with optional fusion evidence while retaining backwards
  compatibility for dense hits. `SearchFilter` now rejects blank snapshot/document identities and
  continues to canonicalize document IDs.
- Added `artifacts/retrieval/korean-fixed-comparison.json`, a deterministic public-safe synthetic
  fixture covering fact, exact numeric, and paraphrase queries. Its executable test runs the real
  NumPy cosine reference, BM25, and hybrid fusion and compares ordered chunk IDs. It is explicitly
  not presented as a real-corpus quality result.

## TDD evidence

The first focused RED run failed during collection because the three production modules did not
exist:

```text
ModuleNotFoundError: No module named 'ragbench.retrieval.bm25'
ModuleNotFoundError: No module named 'ragbench.retrieval.rrf'
ModuleNotFoundError: No module named 'ragbench.retrieval.service'
4 errors during collection
```

After the minimal implementations, the focused suite was GREEN:

```text
uv --cache-dir /private/tmp/ragbench-task9-uv-cache run pytest \
  tests/unit/retrieval/test_bm25.py tests/unit/retrieval/test_rrf.py \
  tests/unit/retrieval/test_service.py tests/unit/retrieval/test_comparison_artifact.py -q

17 passed in 0.05s
```

The self-review validation test for blank filter identities was RED because `SearchFilter` did not
reject an empty corpus snapshot ID. Adding identity validation made the focused BM25 suite GREEN.
The independently calculated BM25 literals also caught and corrected an error in the test's first
hand calculation before it was accepted as evidence.

## Regression/debugging evidence

The first full-suite collection exposed two same-basename modules:
`tests/unit/embeddings/test_service.py` and `tests/unit/retrieval/test_service.py`. A minimal
two-file `--collect-only` reproduction confirmed pytest imported both as top-level `test_service`.
Making only the retrieval test directory a package isolated its module name; the repeat collected
all 17 service tests and the subsequent complete offline suite passed.

## Verification and limitations

Fresh final command evidence is recorded in the task handoff/commit. Verification covers the
focused tests, all non-live/non-gold tests, Ruff, strict mypy, `git diff --check`, and a scan of the
new files for credential-like values.

No provider call, paid action, PostgreSQL call, sealed-gold access, private corpus access, or real
corpus retrieval evaluation was performed. The comparison artifact is deliberately synthetic and
provider-free; real-corpus quality comparison remains a data-operations task after corpus,
parsing, chunks, and embeddings exist.
