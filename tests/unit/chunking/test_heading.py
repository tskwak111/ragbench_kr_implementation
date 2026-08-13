from ragbench.chunking.heading import HeadingAwareChunker
from ragbench.chunking.models import DocumentBlock
from ragbench.chunking.tokenizer import encoding


def _block(identity, text, page, section, kind="paragraph"):
    return DocumentBlock(
        identity, "doc", "parse-snapshot", "a" * 64, page, section, kind, text, text, (page,), False
    )


def test_heading_chunker_merges_same_section_but_not_across_sections():
    blocks = [
        _block("h1", "# 개요", 1, ("개요",), "heading"),
        _block("p1", "짧은 본문", 1, ("개요",)),
        _block("p2", "계속되는 본문", 2, ("개요",)),
        _block("h2", "# 결론", 3, ("결론",), "heading"),
        _block("p3", "결론 본문", 3, ("결론",)),
    ]
    chunks = HeadingAwareChunker(target_size=100, overlap=10).split(blocks)
    assert len(chunks) == 2
    assert chunks[0].section_path == ("개요",)
    assert chunks[0].page_start == 1 and chunks[0].page_end == 2
    assert chunks[1].section_path == ("결론",)


def test_heading_chunker_splits_only_oversized_table_and_retains_order():
    table = "\n".join(f"행{i} | {i * 10} | {i * 20}" for i in range(80))
    chunks = HeadingAwareChunker(target_size=40, overlap=8).split(
        [_block("table", table, 4, ("재무",), "table")]
    )
    assert len(chunks) > 1
    assert all(chunk.page_start == chunk.page_end == 4 for chunk in chunks)
    assert all(chunk.token_count <= 40 for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_heading_chunker_does_not_split_small_table_inside_oversized_section():
    before = "서론 내용 " * 30
    table = "항목 | 값\n매출 | 10"
    after = "후속 내용 " * 30
    chunks = HeadingAwareChunker(target_size=40, overlap=8).split(
        [
            _block("before", before, 1, ("재무",)),
            _block("table", table, 2, ("재무",), "table"),
            _block("after", after, 3, ("재무",)),
        ]
    )
    table_chunks = [chunk for chunk in chunks if "table" in chunk.source_block_ids]
    assert table_chunks
    assert "항목" in "".join(chunk.content for chunk in table_chunks)
    assert "매출" in "".join(chunk.content for chunk in table_chunks)


def test_heading_defaults_are_target_600_overlap_100():
    chunker = HeadingAwareChunker()
    assert chunker.target_size == 600
    assert chunker.overlap == 100


def test_oversized_multiblock_section_uses_one_global_overlapping_token_stream():
    first = "가나다라 " * 120
    table = "항목 | 값\n매출 | 10\n비용 | 7"
    second = "마바사아 " * 120
    chunks = HeadingAwareChunker(target_size=80, overlap=20).split(
        [
            _block("first", first, 1, ("재무",)),
            _block("table", table, 2, ("재무",), "table"),
            _block("second", second, 3, ("재무",)),
        ]
    )
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(
        right.token_start > left.token_start
        for left, right in zip(chunks, chunks[1:], strict=False)
    )
    assert all(
        20 <= left.token_end - right.token_start <= 23
        for left, right in zip(chunks, chunks[1:], strict=False)
    )
    assert chunks[-1].token_end == len(encoding().encode(f"{first}\n\n{table}\n\n{second}"))
    assert any("table" in chunk.source_block_ids for chunk in chunks)
    assert all(
        chunk.content.find("항목") <= chunk.content.find("매출")
        for chunk in chunks
        if "항목" in chunk.content
    )
