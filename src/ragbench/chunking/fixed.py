"""Fixed token-window chunking with source provenance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragbench.chunking.models import ChunkRecord, DocumentBlock
from ragbench.chunking.tokenizer import encoding, safe_token_windows, tokenizer_snapshot
from ragbench.core.hashing import canonical_json_hash


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int
    block: DocumentBlock


def _source_text(blocks: Sequence[DocumentBlock]) -> tuple[str, tuple[_Span, ...]]:
    parts: list[str] = []
    spans: list[_Span] = []
    position = 0
    for block in blocks:
        if block.is_boilerplate or not block.content:
            continue
        if parts:
            parts.append("\n\n")
            position += 2
        start = position
        parts.append(block.content)
        position += len(block.content)
        spans.append(_Span(start, position, block))
    return "".join(parts), tuple(spans)


def _overlapping(spans: Sequence[_Span], start: int, end: int) -> tuple[DocumentBlock, ...]:
    return tuple(span.block for span in spans if span.end > start and span.start < end)


class FixedChunker:
    def __init__(self, size: int, overlap: int = 0) -> None:
        if size <= 0 or overlap < 0 or size <= overlap:
            raise ValueError("size must be positive and greater than non-negative overlap")
        self.size = size
        self.overlap = overlap
        self.strategy = f"fixed-{size}-{overlap}"
        self.strategy_hash = canonical_json_hash(
            {
                "strategy": "fixed",
                "size": size,
                "overlap": overlap,
                "tokenizer": tokenizer_snapshot(),
            }
        )[:16]

    def split(self, blocks: Sequence[DocumentBlock]) -> list[ChunkRecord]:
        if not blocks:
            return []
        identities = {
            (item.document_id, item.parse_snapshot_id, item.source_sha256) for item in blocks
        }
        if len(identities) != 1:
            raise ValueError("blocks must belong to one document and parse snapshot")
        document, parse_snapshot, _ = next(iter(identities))
        text, spans = _source_text(blocks)
        records: list[ChunkRecord] = []
        for ordinal, window in enumerate(safe_token_windows(text, self.size, self.overlap)):
            sources = _overlapping(spans, window.char_start, window.char_end)
            if not sources:
                continue
            pages = [item.page for item in sources]
            sections = {item.section_path for item in sources}
            section = next(iter(sections)) if len(sections) == 1 else ()
            identity = canonical_json_hash(
                {
                    "document": document,
                    "ordinal": ordinal,
                    "token_start": window.token_start,
                    "token_end": window.token_end,
                    "content": window.content,
                }
            )[:24]
            chunk_id = f"{parse_snapshot}:{self.strategy_hash}:{identity}"
            records.append(
                ChunkRecord(
                    chunk_id,
                    document,
                    parse_snapshot,
                    self.strategy,
                    self.strategy_hash,
                    ordinal,
                    min(pages),
                    max(pages),
                    section,
                    window.content,
                    len(encoding().encode(window.content)),
                    window.token_start,
                    window.token_end,
                    tuple(item.block_id for item in sources),
                )
            )
        if records:
            total_tokens = len(encoding().encode(text))
            if records[0].token_start != 0 or records[-1].token_end != total_tokens:
                raise RuntimeError("chunk windows do not cover the normalized document")
            if any(
                right.token_start > left.token_end or right.ordinal != left.ordinal + 1
                for left, right in zip(records, records[1:], strict=False)
            ):
                raise RuntimeError("chunk windows contain a gap or non-monotonic ordinal")
            if len({item.chunk_id for item in records}) != len(records):
                raise RuntimeError("chunk identifiers are not unique")
        return records
