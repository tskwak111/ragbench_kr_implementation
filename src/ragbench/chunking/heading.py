"""Section-preferred chunking with token splitting for oversized sections."""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.chunking.fixed import FixedChunker
from ragbench.chunking.models import ChunkRecord, DocumentBlock
from ragbench.chunking.tokenizer import encoding, tokenizer_snapshot
from ragbench.core.hashing import canonical_json_hash


class HeadingAwareChunker:
    def __init__(self, target_size: int = 600, overlap: int = 100) -> None:
        if target_size <= 0 or overlap < 0 or target_size <= overlap:
            raise ValueError("target_size must be positive and greater than overlap")
        self.target_size = target_size
        self.overlap = overlap
        self.strategy = f"heading-{target_size}-{overlap}"
        self.strategy_hash = canonical_json_hash(
            {
                "strategy": "heading",
                "target_size": target_size,
                "overlap": overlap,
                "tokenizer": tokenizer_snapshot(),
            }
        )[:16]

    def split(self, blocks: Sequence[DocumentBlock]) -> list[ChunkRecord]:
        usable = [item for item in blocks if not item.is_boilerplate and item.content]
        groups: list[list[DocumentBlock]] = []
        for block in usable:
            if not groups or groups[-1][0].section_path != block.section_path:
                groups.append([block])
            else:
                groups[-1].append(block)
        output: list[ChunkRecord] = []
        for group in groups:
            candidates: list[ChunkRecord] = []
            pending: list[DocumentBlock] = []

            for block in group:
                if len(encoding().encode(block.content)) > self.target_size:
                    if pending:
                        candidates.extend(
                            FixedChunker(self.target_size, self.overlap).split(pending)
                        )
                        pending.clear()
                    candidates.extend(FixedChunker(self.target_size, self.overlap).split([block]))
                    continue
                proposed = [*pending, block]
                proposed_text = "\n\n".join(item.content for item in proposed)
                if pending and len(encoding().encode(proposed_text)) > self.target_size:
                    candidates.extend(FixedChunker(self.target_size, self.overlap).split(pending))
                    pending.clear()
                pending.append(block)
            if pending:
                candidates.extend(FixedChunker(self.target_size, self.overlap).split(pending))
            for chunk in candidates:
                ordinal = len(output)
                identity = canonical_json_hash(
                    {
                        "document": chunk.document_id,
                        "ordinal": ordinal,
                        "content": chunk.content,
                        "blocks": chunk.source_block_ids,
                    }
                )[:24]
                chunk_id = f"{chunk.parse_snapshot_id}:{self.strategy_hash}:{identity}"
                output.append(
                    ChunkRecord(
                        chunk_id,
                        chunk.document_id,
                        chunk.parse_snapshot_id,
                        self.strategy,
                        self.strategy_hash,
                        ordinal,
                        chunk.page_start,
                        chunk.page_end,
                        group[0].section_path,
                        chunk.content,
                        chunk.token_count,
                        chunk.token_start,
                        chunk.token_end,
                        chunk.source_block_ids,
                    )
                )
        return output
