# Offline operations report

Date: 2026-08-14 (Asia/Seoul)

This report originally covered work completed without an external provider API. The Standard parse
status below was updated after the separately authorized live batch.

## Corpus

`configs/corpus.yaml` now contains 20 Korean official-publication PDFs totaling 1,981 pages. The content includes public and corporate reports, table-heavy, text-heavy, and mixed documents from multiple organizations and template families.

- Corpus snapshot ID: `7077dab5f83cd024b87a30a1ab05a411804879b85a334a2e7b25f04562cdaa5f`
- Local raw directory: `data/raw/` in the primary workspace, mode `0700`, excluded from Git
- Redistribution: all records are conservatively marked `nonredistributable`
- Structural checks: PDF magic, strict PDF parse, SHA-256, and declared page counts passed for all 20 files
- Visual checks: first, middle, and last page of every PDF were rendered and inspected (60 pages total); no broken or unreadable sample was found
- Freeze gate: the committed `frozen` manifest passes every automated size, diversity, provenance, and file-integrity gate

The operator reviewed the 20-document approval packet and approved the complete corpus on
2026-08-21 for local evaluation with external redistribution prohibited. The committed manifest is
therefore `frozen`. Raw PDF bytes remain deliberately absent from the repository.

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

The authorized Standard parse completed on 2026-08-22:

- successful documents/pages: 20/20 and 1,981/1,981
- failed documents/pages: 0/0
- resolved provider model: `document-parse-260630`
- recorded local gross budget cost: USD 21.791000
- open budget reservations: 0
- 11 PDFs over the synchronous 100-page limit were split in memory and merged with global page and element IDs

Provider-console evidence supplied on 2026-08-26 reports 2,081 Standard pages and USD 20.81 item
cost/used credit: 283 pages on 2026-08-21 and 1,798 pages on 2026-08-22. This is 100 pages and USD
1.00 above the successful corpus. The difference is consistent with the initial oversized-request
failure, but the console export does not identify the exact request, so the discrepancy remains
open.

On 2026-08-24 the local parse checkpoints were deleted when integration tests reset the same
database used for runtime evidence. No file cache or backup exists. Runtime and integration test
databases are now separated and destructive integration resets refuse the configured runtime
database.

The operator approved reconstruction under a USD 21.791 gross cap on 2026-08-26. The rerun restored
all 20 documents and 1,981 pages with 31 provider calls, no failures, no open reservations, and no
page-range or duplicate-element defects. A repeat dry run reported 20 cache hits and zero new calls.
The database was immediately backed up to the ignored local path
`.ragbench/backups/ragbench-standard-2026-08-26.dump` (SHA-256
`d7d621a5446dbb93001f2f006c6cba62f62cfd62abb60826c5801bf786a1b2cd`).

Standard-only manual QA covered 30 stratified pages. It conditionally passed for RAG use: ordinary
prose and most tables retained useful content, while chart series, units, and some digits were not
reliable enough to ground numeric answers. See `docs/reports/standard-parse-qa.md`. This does not
satisfy the planned blinded Standard-versus-Enhanced paired QA.

Enhanced parsing subsequently completed for 3 documents and 185 pages: both climate reports and
`mobis-sustainability-2024`. The local gross cost is USD 6.105000 with no open reservations. The
partial database backup is `.ragbench/backups/ragbench-enhanced-partial-185p-2026-08-26.dump`
(SHA-256 `416dbbfa170a4f4bde70e2ec9d2806cb55b98cd11eac84fb4a7ebe1698f9159d`). Remaining paid requests
were stopped after repeated provider `401 Unauthorized` responses; failed attempts produced no
successful API usage rows.

A preliminary, unblinded 30-page paired review found 3 Enhanced wins, 3 Standard wins, 14 ties, and
10 mixed/unsafe results. Enhanced clearly repaired several chart structures but also added
unsupported visual descriptions and one digit error. See `docs/reports/enhanced-partial-qa.md`.
This partial review does not satisfy the identical-corpus or blinded paired-QA gates.

The public synthetic offline fixtures remain reproducible:

- retrieval screening: 2 questions, BM25, Hit/Recall/MRR evidence all `1.0`, provider calls `0`
- experiment replay: stable run ID `24acd230caac6247e1ae72069139ce661`; second run reused the artifact with `new_responses=0` and provider calls `0`

These are framework smoke results only, not real-corpus benchmark scores.

## Remaining external gates

- Remaining Enhanced parsing (17 documents/1,796 pages) and blinded corpus-wide paired manual QA
- Provider-backed document/query embeddings
- Real-corpus chunking, indexing, screening, generation, judge calibration, and final experiments that depend on provider artifacts
- Authorized reviewer work for candidate/gold sealing
- Any live API deployment or HTTP smoke test

The 100-page console discrepancy, gold data, and private review artifacts remain pending.
