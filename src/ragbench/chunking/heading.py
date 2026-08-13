"""Section-preferred chunking with one token stream per section."""

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
        token_base = 0
        for group in groups:
            section_text = "\n\n".join(block.content for block in group)
            candidates = FixedChunker(self.target_size, self.overlap).split(group)
            for candidate in candidates:
                ordinal = len(output)
                token_start = token_base + candidate.token_start
                token_end = token_base + candidate.token_end
                identity = canonical_json_hash(
                    {
                        "document": candidate.document_id,
                        "ordinal": ordinal,
                        "token_start": token_start,
                        "token_end": token_end,
                        "content": candidate.content,
                        "blocks": candidate.source_block_ids,
                    }
                )[:24]
                chunk_id = f"{candidate.parse_snapshot_id}:{self.strategy_hash}:{identity}"
                output.append(
                    ChunkRecord(
                        chunk_id,
                        candidate.document_id,
                        candidate.parse_snapshot_id,
                        self.strategy,
                        self.strategy_hash,
                        ordinal,
                        candidate.page_start,
                        candidate.page_end,
                        group[0].section_path,
                        candidate.content,
                        candidate.token_count,
                        token_start,
                        token_end,
                        candidate.source_block_ids,
                    )
                )
            token_base += len(encoding().encode(section_text))
        return output
