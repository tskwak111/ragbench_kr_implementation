# Task 5 report — corpus manifest framework and freeze gate

## Implemented framework

- Added the `pypdf` runtime dependency and a typed YAML manifest model.
- Added local PDF magic, structure, SHA-256, and page-count validation; duplicate ID/content detection; provenance and redistribution checks; document/page/organization target enforcement; and a freeze-only gate.
- Defined a canonical `corpus_snapshot_id` from sorted stable metadata and content hashes, explicitly excluding operator-local paths.
- Added a public-safe export that retains source/provenance while excluding local paths and document bytes.
- Added `scripts/collect_corpus.py`: an operator-approved local-PDF collector with approved-root, symlink, traversal, PDF-magic, overwrite, atomic-copy, hashing, page-count, review-report, and private manifest-fragment safeguards.
- Added the ignored `data/raw/` directory, an empty `configs/corpus.yaml` with `status: draft`, data instructions, and ADR 001.

## Deliberately not completed

No real corpus documents, licenses, hashes, page totals, manual page inspections, or frozen snapshot were created or claimed. The draft starter manifest is intentionally empty and `validate(freeze=True)` fails its document/page targets. Source selection, licensing review, acquisition, manual first/middle/last-page inspection, target compliance, and the eventual status change to `frozen` remain data-operations work for a human reviewer.

## Verification

Executed successfully from the task worktree:

```text
.venv/bin/pytest tests/unit/ingestion -q
10 passed

.venv/bin/ruff check .
All checks passed!

.venv/bin/mypy src/ragbench
Success: no issues found in 19 source files

.venv/bin/pytest -m 'not live and not gold' -q
88 passed, 2 skipped
```

## Plan and commit state

Task 5 framework/code is ready for review. The plan's corpus acquisition and actual freeze steps remain pending by design. No misleading `data: freeze corpus manifest` commit was created; the intended code-only commit message is `feat: add corpus manifest freeze gate` and is pending the parent integration/review process.

## Fix round 1 — descriptor-safe collection and distribution gate

- Reworked the collector around fail-closed POSIX directory/file descriptors with `O_NOFOLLOW`, regular-file checks, staged descriptor hashing/PDF parsing/page counting, fsync, and atomic hard-link publication that cannot overwrite an existing raw PDF.
- Staged and private YAML output names are unpredictable (`tempfile` candidate names) and are published with the same directory-descriptor no-replace pattern. The collector rejects unsafe source paths, symlink redirection, non-private output locations, and platforms without the required primitives.
- Added sector, content-stratum, and template-family metadata plus freeze checks for corporate/public coverage, table-heavy/text-heavy coverage, and template-family concentration. A frozen manifest always triggers freeze validation even without `freeze=True`.
- Added duplicate-YAML-key rejection and fully deterministic snapshot ordering even when documents have colliding hashes before validation.
- Added offline regressions for symlink sources, no-overwrite races, malformed staging cleanup/retry, unknown-license reports, output redirection, frozen status validation, diversity targets, duplicate YAML keys, and snapshot ordering.

Fresh verification for this fix round:

```text
.venv/bin/ruff check .
All checks passed!

.venv/bin/mypy src/ragbench scripts/collect_corpus.py
Success: no issues found in 20 source files

.venv/bin/pytest tests/unit/ingestion -q
16 passed

.venv/bin/pytest -m 'not live and not gold' -q
94 passed, 2 skipped
```
