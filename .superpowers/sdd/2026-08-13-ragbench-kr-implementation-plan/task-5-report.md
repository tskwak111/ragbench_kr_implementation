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

## Fix round 2 — transactional multi-artifact collection

Collection now treats the final PDF link as its commit marker. It validates operator metadata before staging source bytes, then stages and fsyncs the PDF, private manifest fragment, and review report before it exposes any final output. It preflights all final names, publishes fragment then report, performs remaining directory fsync work, and publishes the PDF last.

Every staged artifact records its device/inode. If metadata or publication fails, rollback stats each final name through its directory descriptor and unlinks it only if it still identifies that invocation's staged inode. Pre-existing and concurrent replacement files are not removed. Temporary names are always cleaned up. The only operations after the PDF commit link are bookkeeping and staged-file cleanup/descriptor closure.

Added regressions for metadata-validation cleanup and retry, pre-existing fragment/report preservation before PDF publication, and injected failures after each publication link. Fresh verification:

```text
.venv/bin/ruff check .
All checks passed!

.venv/bin/mypy src/ragbench scripts/collect_corpus.py
Success: no issues found in 20 source files

.venv/bin/pytest tests/unit/ingestion -q
22 passed

.venv/bin/pytest -m 'not live and not gold' -q
100 passed, 2 skipped
```

## Fix round 3 — idempotent publication without final-path rollback

Removed final-output rollback entirely. The collector now holds an exclusive `fcntl.flock` in the verified private directory from preflight through staging, publication, and directory fsync. Existing final artifact links are accepted only if their bytes exactly match the staged artifact; they represent a safe retry of a prior partial cooperative run. Conflicting bytes fail the collection before later publication. No error path unlinks a final PDF, fragment, or report.

The PDF stays the final commit marker. The private output directory is fsynced before the PDF link; the raw directory is fsynced immediately after that link. Record/model validation occurs before source staging, including the positive page-count model constraint. Staging cleanup is entered immediately after each stage and compares the expected inode first; cleanup relies on the documented trusted private-directory/cooperating-collector boundary rather than claiming protection against a malicious same-UID writer.

Added regressions for idempotent partial metadata retry, raw-directory fsync ordering after the PDF commit marker, and replaced-temp-name preservation. Existing publication failure tests assert that no final unlink rollback occurs.

## Fix round 4 — exclusive destination boundary and zero-page cleanup

The collector now opens both `raw_dir` and `private_output_dir` through the existing component-by-component no-follow path, then verifies the opened directory descriptor is owned by the current effective UID and has no group/world permission bits. Existing unsafe directories fail closed before PDF staging with an actionable `chmod 0700` error. Unpredictable `O_EXCL` names and the exclusive directory boundary prevent other OS users from replacing temp names; same-UID processes remain the explicitly trusted/cooperating boundary, with no protection claimed against a malicious same-UID process.

Staging cleanup now tracks only the unpredictable names generated and opened exclusively by the invocation and unlinks those names directly within the enforced directory boundary. Final published names are still never unlinked on failure. `_stage_pdf` rejects a parsed PDF with zero pages before it returns, keeping that error inside its own staging cleanup scope.

Added regressions proving zero-page PDFs leave no temp or final artifacts, `0755` raw/private directories fail before staging, a mocked wrong-owner directory fails before staging, and `0700` directories succeed. Existing idempotent retry, no-final-rollback, and raw-directory fsync-order regressions continue to pass. Updated the data setup instructions and corpus-policy ADR with the `0700` requirement and exact same-UID trust boundary.

Fresh verification for this fix round:

```text
.venv/bin/ruff check .
All checks passed!

.venv/bin/mypy src/ragbench scripts/collect_corpus.py
Success: no issues found in 20 source files

.venv/bin/pytest tests/unit/ingestion -q
30 passed

.venv/bin/pytest -m 'not live and not gold' -q
108 passed, 2 skipped
```
