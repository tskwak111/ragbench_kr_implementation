# Task 7 report — normalization and chunking variants

## Implemented framework

- Added immutable `DocumentBlock` and `ChunkRecord` contracts carrying document, parse-snapshot,
  source-hash, page, section, block-kind, source-element, token-window, and strategy provenance.
- Added conservative NFC/line-ending/whitespace normalization. Headings, page anchors, numbers,
  table cell order, raw source content, and empty pages are retained. A header/footer is tagged as
  boilerplate only when the identical typed value occurs exactly once on at least three distinct
  pages; tagged blocks remain in the normalized artifact for audit and reconstruction.
- Pinned `tiktoken==0.11.x` with the `cl100k_base` encoding in the strategy snapshot. Fixed
  chunking implements the six specified `(size, overlap)` variants using tokenizer-token windows,
  rejects non-progressing configurations, avoids replacement characters at Korean byte
  boundaries, and asserts gap-free coverage, monotonic ordinals, and unique deterministic IDs.
- Added heading-aware `600/100` chunking. It groups blocks by section path, merges adjacent
  undersized blocks only within that section, keeps a table atomic unless that table itself is
  oversized, token-splits oversized blocks, and retains source block IDs and page ranges.
- Added an offline build script that accepts complete successful Standard and Enhanced checkpoint
  exports for one identical corpus, rejects incomplete pages, mixed corpus/parse snapshots,
  duplicate documents, failed checkpoints, and mode/source mismatches, and creates the planned
  14 immutable JSONL dataset snapshots plus tokenizer/config metadata and token distributions.
  Existing snapshots are accepted only when byte-identical and are never silently overwritten.

## Tested behavior

Fixture-driven tests cover Korean Unicode/token boundaries, tables and structured cell order,
heading hierarchies, empty pages, page transitions, evidence-based repeated headers/footers,
content shorter than overlap, invalid overlap configurations, deterministic IDs, table atomicity,
all 14 planned variants, complete/mixed checkpoint rejection, and immutable export behavior.

## Deliberately pending operations

No parsed production corpus exists in this worktree. Therefore no claim is made that the 14 real
datasets were built, and no text-heavy/table-heavy notebook inspection or boundary-defect
observation was fabricated. Run `scripts/build_chunks.py` only after complete immutable Standard
and Enhanced parse checkpoint exports exist for the same effective corpus. The manual inspection
step remains pending until those artifacts are available. SQL chunk persistence is deferred; the
stable model and private-permission JSONL artifacts provide the Task 7 handoff boundary.

## Verification

The final fresh Ruff, strict mypy, and full offline pytest results are recorded in the task handoff.
