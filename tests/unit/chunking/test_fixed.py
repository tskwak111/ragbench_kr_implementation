import pytest
import tiktoken

from ragbench.chunking.fixed import FixedChunker
from ragbench.chunking.models import DocumentBlock


def _block(content, *, page=1, kind="paragraph", section=("절",)):
    return DocumentBlock(
        "block-1",
        "doc",
        "parse-snapshot",
        "a" * 64,
        page,
        section,
        kind,
        content,
        content,
        (0,),
        False,
    )


def test_fixed_chunker_is_token_aware_deterministic_and_preserves_korean():
    text = "대한민국 공공 데이터 성능 검증. " * 80
    chunker = FixedChunker(30, 7)
    first = chunker.split([_block(text)])
    second = chunker.split([_block(text)])

    assert first == second
    assert [chunk.ordinal for chunk in first] == list(range(len(first)))
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert all(chunk.chunk_id.startswith("parse-snapshot:") for chunk in first)
    assert all(chunk.token_count <= 30 for chunk in first)
    assert all("�" not in chunk.content for chunk in first)
    encoding = tiktoken.get_encoding("cl100k_base")
    source_tokens = encoding.encode(text)
    spans = [(chunk.token_start, chunk.token_end) for chunk in first]
    assert spans[0][0] == 0 and spans[-1][1] == len(source_tokens)
    assert all(
        right[0] <= left[1] and right[0] > left[0]
        for left, right in zip(spans, spans[1:], strict=False)
    )


def test_fixed_chunker_keeps_page_ranges_tables_and_ignores_audited_boilerplate():
    blocks = [
        _block("첫 페이지", page=1),
        DocumentBlock(
            "table",
            "doc",
            "parse-snapshot",
            "a" * 64,
            2,
            ("절",),
            "table",
            "A | B\n1 | 2",
            "A | B\n1 | 2",
            (1,),
            False,
        ),
        DocumentBlock(
            "header", "doc", "parse-snapshot", "a" * 64, 2, (), "header", "반복", "반복", (2,), True
        ),
    ]
    chunks = FixedChunker(100, 10).split(blocks)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1 and chunks[0].page_end == 2
    assert chunks[0].content == "첫 페이지\n\nA | B\n1 | 2"
    assert "반복" not in chunks[0].content


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, -1), (10, 10), (5, 8)])
def test_fixed_chunker_rejects_non_progressing_configuration(size, overlap):
    with pytest.raises(ValueError):
        FixedChunker(size, overlap)


def test_content_shorter_than_overlap_produces_one_chunk():
    assert len(FixedChunker(20, 10).split([_block("짧은 문서")])) == 1
