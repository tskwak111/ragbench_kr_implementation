"""Stable normalized-block and chunk records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BlockKind = Literal[
    "heading", "paragraph", "table", "image", "header", "footer", "other", "empty_page"
]


@dataclass(frozen=True, slots=True)
class DocumentBlock:
    block_id: str
    document_id: str
    parse_snapshot_id: str
    source_sha256: str
    page: int
    section_path: tuple[str, ...]
    block_kind: BlockKind
    content: str
    source_content: str
    source_element_indexes: tuple[int, ...]
    is_boilerplate: bool

    def __post_init__(self) -> None:
        if self.page <= 0:
            raise ValueError("page must be positive")
        if not self.block_id or not self.document_id or not self.parse_snapshot_id:
            raise ValueError("block identity fields cannot be empty")


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    parse_snapshot_id: str
    strategy: str
    strategy_hash: str
    ordinal: int
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    content: str
    token_count: int
    token_start: int
    token_end: int
    source_block_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.ordinal < 0 or self.page_start <= 0 or self.page_end < self.page_start:
            raise ValueError("invalid chunk ordinal or page range")
        if self.token_count < 0 or self.token_start < 0 or self.token_end < self.token_start:
            raise ValueError("invalid token range")
