# Offline operations report

Date: 2026-08-14 (Asia/Seoul)

This report covers the remaining work that can be completed without an external provider API. It does not claim a provider parse, embedding, generation, judge, gold-set review, or FastAPI deployment run.

## Corpus

`configs/corpus.yaml` now contains 20 Korean official-publication PDFs totaling 1,981 pages. The content includes public and corporate reports, table-heavy, text-heavy, and mixed documents from multiple organizations and template families.

- Corpus snapshot ID: `7077dab5f83cd024b87a30a1ab05a411804879b85a334a2e7b25f04562cdaa5f`
- Local raw directory: `data/raw/` in the primary workspace, mode `0700`, excluded from Git
- Redistribution: all records are conservatively marked `nonredistributable`
- Structural checks: PDF magic, strict PDF parse, SHA-256, and declared page counts passed for all 20 files
- Visual checks: first, middle, and last page of every PDF were rendered and inspected (60 pages total); no broken or unreadable sample was found
- Freeze gate: an in-memory `frozen` clone passes every automated size, diversity, provenance, and file-integrity gate

The committed manifest remains `draft`. The project requires human source/license/page review before changing that field to `frozen`; automated inspection is not represented as human approval. Raw PDF bytes are deliberately absent from the repository.

## PostgreSQL and pgvector

A disposable local database was tested with PostgreSQL 16.15 and pgvector 0.8.6. The full migration chain and the two configured integration modules passed against the real extension:

```text
tests/integration/db/test_schema.py
tests/integration/retrieval/test_pgvector.py
6 passed
```

The integration run covered schema/evidence constraints, atomic budget transitions, parse-checkpoint persistence, concurrent embedding snapshot creation, vector persistence and finalization, 50-query NumPy parity, document filtering, and an HNSW `EXPLAIN` access-path assertion.

The complete non-live/non-gold suite with that database enabled passed with `461 passed, 1 skipped`.

Real-database execution exposed and fixed four issues that the offline fakes did not reveal:

1. Async savepoint use in the schema constraint test was syntactically incorrect.
2. Embedding snapshot children could flush before their parent snapshot under SQLAlchemy ordering; the parent is now explicitly flushed first.
3. pgvector's async driver requires list/array vector values rather than tuples.
4. Dense search always joined the artifact table, preventing the unfiltered HNSW path; the join is now emitted only for document filters.

The parity fixture now uses exact tied vector pairs and float32-safe separation. Its small 100-row `EXPLAIN` probe disables explicit sorting so that it verifies index eligibility rather than PostgreSQL's rational preference for a btree scan and sort on a tiny relation.

## Parse and offline replay

The Standard parse dry-run completed without a provider call:

- projected calls: 20
- projected pages: 1,981
- base cost: USD 19.810000
- VAT-buffered worst case: USD 21.791000
- remaining configured budget: USD 113.209000
- plan hash: `beafc28b8a07c88d937d82c6a237eef50408ab1d98c878c79e9bba290f6ed6b2`
- price snapshot hash: `e68984ea05ccc439a6f2f6da71f63279a606f5441dc752feb004aec1c9791bc5`

Enhanced dry-run correctly refuses to proceed until successful Standard checkpoints exist. Because provider parsing is excluded, real parsed checkpoints, chunks, embeddings, and corpus-backed retrieval experiments cannot be produced honestly.

The public synthetic offline fixtures remain reproducible:

- retrieval screening: 2 questions, BM25, Hit/Recall/MRR evidence all `1.0`, provider calls `0`
- experiment replay: stable run ID `24acd230caac6247e1ae72069139ce661`; second run reused the artifact with `new_responses=0` and provider calls `0`

These are framework smoke results only, not real-corpus benchmark scores.

## Remaining external gates

- Human review of source selection, licensing, and rendered pages before corpus freeze
- Provider-backed Standard and Enhanced parsing
- Provider-backed document/query embeddings
- Real-corpus chunking, indexing, screening, generation, judge calibration, and final experiments that depend on provider artifacts
- Authorized reviewer work for candidate/gold sealing
- Any live API deployment or HTTP smoke test

No external provider API, paid call, live test, gold data, or private review artifact was accessed during this run.
