from ragbench.ingestion.manifest import DocumentRecord
from ragbench.ingestion.normalizer import normalize, reconstruct_normalized_text
from ragbench.ingestion.parser import CorpusSnapshot, ParseCheckpoint


def _checkpoint(tmp_path, elements, pages=3):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"pdf")
    document = DocumentRecord(
        document_id="doc",
        title="doc",
        organization="org",
        year=2025,
        document_type="report",
        language="ko",
        sector="public",
        content_stratum="mixed",
        template_family="family",
        license="reviewed",
        redistribution_status="nonredistributable",
        local_path=path,
        sha256="a" * 64,
        page_count=pages,
        inclusion_rationale="fixture",
    )
    snapshot = CorpusSnapshot("parse-snapshot", (document,))
    raw = {
        "model_version": "v1",
        "content": {"markdown": "", "html": ""},
        "elements": elements,
        "pages": [{"page": page, "source_page": page} for page in range(1, pages + 1)],
        "usage": {"pages": pages},
    }
    return ParseCheckpoint.success_for_test(snapshot, document, mode="standard", raw=raw)


def test_normalize_preserves_korean_tables_headings_pages_and_source(tmp_path):
    parsed = _checkpoint(
        tmp_path,
        [
            {"page": 1, "category": "heading", "content": "# 결과\r\n"},
            {"page": 1, "category": "paragraph", "content": "매출  1,234 원\t증가"},
            {"page": 2, "category": "table", "content": "항목 | 2024 | 2025\r\n매출 | 10 | 20"},
        ],
    )

    blocks = normalize(parsed)

    assert [(b.page, b.block_kind, b.content) for b in blocks] == [
        (1, "heading", "# 결과"),
        (1, "paragraph", "매출 1,234 원\t증가"),
        (2, "table", "항목 | 2024 | 2025\n매출 | 10 | 20"),
        (3, "empty_page", ""),
    ]
    assert blocks[1].section_path == ("결과",)
    assert blocks[2].content.index("2024") < blocks[2].content.index("2025")
    assert reconstruct_normalized_text(blocks, include_boilerplate=True) == "\n\n".join(
        b.content for b in blocks
    )
    assert blocks[0].source_content == "# 결과\r\n"


def test_repeated_header_footer_are_tagged_but_retained_for_audit(tmp_path):
    elements = []
    for page in range(1, 4):
        elements.extend(
            [
                {"page": page, "category": "header", "content": "공통 보고서"},
                {"page": page, "category": "paragraph", "content": f"본문 {page}"},
                {"page": page, "category": "footer", "content": "내부용"},
            ]
        )
    blocks = normalize(_checkpoint(tmp_path, elements))

    boilerplate = [block for block in blocks if block.is_boilerplate]
    assert [(block.block_kind, block.content) for block in boilerplate] == [
        ("header", "공통 보고서"),
        ("footer", "내부용"),
        ("header", "공통 보고서"),
        ("footer", "내부용"),
        ("header", "공통 보고서"),
        ("footer", "내부용"),
    ]
    assert "본문 1" in reconstruct_normalized_text(blocks)
    assert "공통 보고서" not in reconstruct_normalized_text(blocks)


def test_two_page_repetition_is_not_enough_to_remove_content(tmp_path):
    parsed = _checkpoint(
        tmp_path,
        [
            {"page": 1, "category": "header", "content": "중요 제목"},
            {"page": 2, "category": "header", "content": "중요 제목"},
        ],
        pages=2,
    )
    assert all(not block.is_boilerplate for block in normalize(parsed))


def test_markdown_heading_levels_form_stable_section_paths(tmp_path):
    parsed = _checkpoint(
        tmp_path,
        [
            {"page": 1, "category": "heading", "content": "# 상위"},
            {"page": 1, "category": "heading", "content": "## 하위"},
            {"page": 1, "category": "paragraph", "content": "내용"},
            {"page": 2, "category": "heading", "content": "# 다음"},
        ],
        pages=2,
    )
    blocks = normalize(parsed)
    assert [block.section_path for block in blocks] == [
        ("상위",),
        ("상위", "하위"),
        ("상위", "하위"),
        ("다음",),
    ]


def test_empty_page_uses_preceding_section_even_when_elements_are_unsorted(tmp_path):
    parsed = _checkpoint(
        tmp_path,
        [
            {"page": 1, "category": "heading", "content": "# 이전"},
            {"page": 3, "category": "heading", "content": "# 다음"},
        ],
    )
    blocks = normalize(parsed)
    assert [(block.page, block.section_path) for block in blocks] == [
        (1, ("이전",)),
        (2, ("이전",)),
        (3, ("다음",)),
    ]


def test_structured_table_content_is_serialized_without_reordering_cells(tmp_path):
    parsed = _checkpoint(
        tmp_path,
        [{"page": 1, "category": "table", "content": [["항목", "값"], ["매출", 10]]}],
        pages=1,
    )
    assert normalize(parsed)[0].content == '[["항목", "값"], ["매출", 10]]'


def test_every_source_element_is_reconstructable_from_normalized_blocks(tmp_path):
    elements = [
        {"page": 1, "category": "heading", "content": "# 제목\r\n"},
        {"page": 1, "category": "paragraph", "content": "금액  10"},
        {"page": 2, "category": "table", "content": [["A", 1], ["B", 2]]},
    ]
    blocks = normalize(_checkpoint(tmp_path, elements, pages=2))
    assert [index for block in blocks for index in block.source_element_indexes] == [0, 1, 2]
    assert [block.source_content for block in blocks] == [
        "# 제목\r\n",
        "금액  10",
        '[["A", 1], ["B", 2]]',
    ]
