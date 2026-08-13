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

## Fix round 1 — real checkpoint compatibility and lossless offline token windows

- Removed the synthetic `parse_snapshot_id` input requirement. The builder now consumes actual
  `dataclasses.asdict(ParseCheckpoint)` records and derives each mode's parse snapshot ID from the
  corpus snapshot, mode, provider model/version, and sorted source/raw-response hashes. That ID is
  propagated consistently into blocks, chunk records, IDs, and dataset metadata.
- Replaced character-offset repair with exact token-byte prefix accounting. Windows start and end
  only at boundaries that are both whole-token and valid UTF-8 boundaries. If a Unicode codepoint
  spans multiple tokens, a window or its effective overlap expands only enough to include that
  codepoint; stored token ranges and counts describe the actual included tokens.
- Heading-aware splitting now serializes every section exactly once and chunks its single token
  stream. Token offsets never reset between internal blocks, and per-chunk page/block provenance
  comes from exact character intersections with the serialized section.
- Vendored the official OpenAI `cl100k_base.tiktoken` vocabulary under the tiktoken MIT license.
  Runtime loading reads only the packaged asset, verifies SHA-256
  `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`, and performs no network
  or registry lookup. A wheel build confirmed the vocabulary, provenance note, and license are
  packaged.
- Hardened immutable writes with `lstat`, `O_NOFOLLOW`, regular-file and effective-user ownership
  checks, byte comparison, and descriptor-based permission repair to mode `0600`. Symlinks fail
  closed. Added explicit source-element reconstruction coverage for normalization.

No real corpus build or notebook inspection was performed in this fix round.
