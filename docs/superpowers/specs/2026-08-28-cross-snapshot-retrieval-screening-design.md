# Cross-Snapshot Evidence Mapping and Real Retrieval Screening Design

**Date:** 2026-08-28
**Status:** proposed
**Scope:** real-corpus benchmark source windows, cross-snapshot evidence binding, and retrieval-screen execution

## Problem

The project has 14 complete chunk/embedding snapshots: two parse modes times seven chunk
strategies. Chunk IDs are intentionally bound to a parse snapshot and chunk strategy, so the same
source fact has different chunk IDs in different snapshots.

`QuestionCandidate` already records a useful canonical evidence locator: verbatim `text`,
`document_id`, `page`, and the source unit's `chunk_id`. However, current retrieval metrics compare
ranked chunk IDs directly with those source IDs. A question generated from one snapshot therefore
cannot be scored fairly against the other 13 snapshots. In addition,
`scripts/run_retrieval_screen.py --execute` deliberately fails because no real snapshot loader,
retriever factory, or durable result store is bound.

## Goals

- Generate benchmark candidates from bounded, page-aware source windows without another parse or
  embedding call.
- Preserve one human-reviewable evidence locator per supporting span.
- Deterministically bind every evidence span to at most one representative chunk in each of the 14
  snapshots.
- Count an unmapped span as a retrieval failure, not as an unscored question.
- Run the same development questions over the fixed BM25, dense, and hybrid grid.
- Persist every ranked hit, evidence binding, mapping miss, latency, and aggregate metric.
- Keep gold sealed and keep all mapping and retrieval-screen work provider-free.

## Non-goals

- Fuzzy or semantic evidence matching that can silently create false relevance labels.
- New parsing, chunking, embedding, generation models, or dashboard work.
- Production API deployment.
- Automatic approval of generated benchmark questions.

## Source windows

Add `scripts/build_source_windows.py`. It reads the complete private parse-checkpoint export and
reuses `ragbench.ingestion.normalizer.normalize`; it does not introduce a second normalizer.

The Enhanced parse is the generation source because it retains the richest tables and visual
structure. Human review against the original PDF remains mandatory, so unsupported Enhanced-only
descriptions are rejected before a development or gold snapshot is created.

For each document, the builder:

1. Drops empty and repeated boilerplate blocks.
2. Keeps each normalized block as one `SourceUnit`, using the deterministic block ID as the existing
   `chunk_id` provenance field.
3. Groups ordered blocks into windows of at most two adjacent pages and at most the configured
   `source_window_max_chars`.
4. Never splits a normalized block. A single oversized block fails with its document/page identity
   instead of being silently truncated.
5. Writes an immutable, owner-only JSONL file plus public-safe metadata containing only counts and
   hashes.

“Repeated boilerplate” is not a new heuristic: it means the existing normalizer's
`DocumentBlock.is_boilerplate` flag. Empty-page markers and blocks with blank normalized content are
also excluded.

Window identity binds corpus snapshot, Enhanced parse snapshot, document, page range, ordered block
IDs, normalized content hash, and builder version. Rebuilding identical input must produce identical
bytes.

## Canonical evidence and per-snapshot binding

Do not replace `EvidenceSpan`. Its existing `text`, `document_id`, `page`, and source `chunk_id`
fields are the canonical evidence locator and generation provenance.

Add a separate immutable binding record for each `(question, evidence span, target chunk snapshot)`:

- evidence-span identity;
- target chunk snapshot identity;
- `target_chunk_id`, nullable;
- status: `exact` or `missing`;
- candidate count before deterministic selection;
- binding algorithm version.

Evidence-span identity is the hash of question ID, zero-based span ordinal, and the complete
canonical locator. The ordinal preserves two distinct reviewed spans even when they map to the same
target chunk.

The mapper loads the immutable target chunk JSONL and considers only chunks from the same document
whose page range contains the evidence page. It applies Unicode NFKC and whitespace normalization to
both the verbatim evidence and chunk content. Exact containment is required. When several chunks
contain the span, choose the chunk with the fewest excess normalized characters, then the stable
chunk ID. No fuzzy fallback is allowed.

A missing mapping is meaningful end-to-end evidence that the target parse/chunk snapshot does not
contain the reviewed span. It remains in the scoring denominator and can never be counted as a hit.

## Development question artifact

The public-safe `SplitSnapshot` contains membership only, so it is not itself a runnable question
dataset. After review and adjudication, materialize one owner-only, immutable development JSONL file
containing question ID, exact prompt, type, answerability, and the reviewed `EvidenceSpan` records.
Corrections must be converted back into validated `EvidenceSpan` values; free-form correction text
is not accepted by the screening loader. Its membership must exactly match the authorized
`dev_auto` snapshot and its content hash becomes part of every screening run identity. The loader
has no path or capability for `test_gold`.

## Retrieval metrics

Score reviewed evidence spans rather than assuming source chunk IDs are portable:

- **Hit@K:** at least one mapped evidence target appears in the top K; all-missing evidence yields 0.
- **Evidence Recall@K:** retrieved evidence-span targets divided by all reviewed evidence spans,
  including missing mappings.
- **MRR:** reciprocal rank of the first mapped evidence target; 0 when none is retrieved.

Each reviewed span has at most one representative target per snapshot, preventing larger overlap
strategies from inflating the relevance denominator. A retrieved chunk satisfies every reviewed
span bound to that chunk, but each span still contributes one unit to the recall denominator.
Existing aggregate and paired-bootstrap logic remains reusable after the span-level per-question
metric rows are materialized.

Unanswerable questions remain explicitly unscored for retrieval and are evaluated later by the RAG
abstention stage.

## Real screening execution

Extend the existing screening path rather than adding a second runner:

1. A private snapshot inventory binds each of the 14 embedding snapshots to its immutable chunk
   dataset and parse/chunk metadata.
2. A development-question loader reads only the approved development snapshot, never sealed gold.
3. The evidence mapper materializes or reuses content-addressed bindings for every question and
   target snapshot.
4. Existing BM25, dense, and hybrid retrievers run the predeclared configuration grid at K=3/5/10.
5. The existing atomic `FileScreeningStore` is upgraded to a versioned checkpoint that persists the
   binding-dataset hash, metric version, all ranked hits, raw per-question metrics, latencies, and
   final aggregate artifacts. Bindings themselves remain in their immutable JSONL dataset.
6. Resume skips only rows whose complete immutable identity already exists.
7. After all 126 configurations complete, the existing frozen shortlist rule and leaderboard export
   consume the new outcomes unchanged.

The execution command remains opt-in but has no provider or paid flag because all document vectors
already exist and live query embedding calls are forbidden in this stage. Dense query vectors must
be precomputed in a separately displayed, paid batch before real screening, or the dense and hybrid
configurations stay blocked. The immutable cache keys exact question text, query model, dimension,
and provider parameters; identical query-model snapshots reuse one vector. A cache-only
`QueryEmbedder` validates those identities and never falls back to the provider. BM25-only screening
can run without that batch.

## Error handling and safety

- Reject mixed corpus snapshots, incomplete embedding snapshots, missing chunk datasets, hash
  mismatches, duplicate IDs, page-range violations, and mutable output conflicts.
- Reject candidate evidence that is not an exact substring of its assigned source window before any
  cross-snapshot mapping.
- Record mapping misses; never repair them with fuzzy matching.
- Refuse real screening when question, inventory, code commit, or metric-version identity is absent.
- Include the binding-dataset hash and metric version in the run identity; never resume a v1
  checkpoint under the new span-level metric.
- Use `retrieval-v2` and `retrieval-checkpoint-v2` for the span-level metric and checkpoint schemas;
  do not reinterpret existing v1 artifacts.
- Keep source windows, questions, bindings, ranked hits, and database backups ignored/private.
- Never load sealed gold during source-window building, development generation, or retrieval
  screening.

## Testing

Use TDD in this order:

1. Source-window tests: deterministic bytes, page/character bounds, boilerplate exclusion,
   oversized-block rejection, immutable conflict rejection.
2. Evidence-binding tests: exact mapping, page/document filtering, deterministic smallest-container
   selection, whitespace normalization, and missing status without fuzzy fallback.
3. Metric tests: mapped hits, missing spans in the denominator, multiple evidence spans, MRR, and
   unanswerable handling.
4. Screening integration test: two snapshots with different chunk IDs receive the same question,
   persist all hits, and produce deterministic paired rows.
5. Query-cache test: exact identity hit, model/dimension mismatch rejection, missing-vector failure,
   and zero provider fallback.
6. Existing non-live/non-gold suite, strict mypy, Ruff, notebook checks, and `git diff --check`.

## Delivery order

1. Build and verify private source windows.
2. Run benchmark-generation dry-run and display exact plan/campaign hashes and maximum cost.
3. Execute paid candidate generation only after explicit approval.
4. Human-review candidates and create the development snapshot; gold remains sealed.
5. Build cross-snapshot evidence bindings.
6. Dry-run the query-vector batch and display exact question count, input tokens, cache hits, and
   maximum cost; precompute only after separate exact approval.
7. Run and report the real retrieval screen.
