"""Conservative normalization of parse evidence into auditable blocks."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ragbench.chunking.models import BlockKind, DocumentBlock
from ragbench.core.hashing import canonical_json_hash
from ragbench.ingestion.parser import ParseCheckpoint

_SPACE_RUN = re.compile(r"(?<!\n)[ \f\v]{2,}")
_KINDS: dict[str, BlockKind] = {
    "heading": "heading",
    "title": "heading",
    "paragraph": "paragraph",
    "text": "paragraph",
    "table": "table",
    "image": "image",
    "header": "header",
    "footer": "footer",
}


def _text(value: Any, *, preserve_structure: bool = False) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("markdown", "text", "html"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    if preserve_structure and isinstance(value, (list, Mapping)):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    return ""


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(_SPACE_RUN.sub(" ", line).rstrip() for line in value.split("\n")).strip("\n")


def _heading_label(content: str) -> str:
    return content.lstrip("#").strip()


def _heading_level(content: str) -> int:
    markers = len(content) - len(content.lstrip("#"))
    return max(1, min(markers or 1, 6))


def normalize(parsed: ParseCheckpoint | Mapping[str, Any]) -> list[DocumentBlock]:
    """Normalize without losing source evidence; repeated boilerplate remains tagged/auditable."""
    if isinstance(parsed, ParseCheckpoint):
        snapshot = parsed.snapshot_id
        document = parsed.document_id
        source_hash = parsed.source_sha256
        pages = parsed.expected_pages
        elements: Sequence[Mapping[str, Any]] = parsed.elements
    else:
        snapshot = str(parsed.get("parse_snapshot_id") or parsed.get("snapshot_id") or "")
        document = str(parsed.get("document_id") or "")
        source_hash = str(parsed.get("source_sha256") or "")
        pages = int(parsed.get("expected_pages") or 0)
        elements = tuple(item for item in parsed.get("elements", ()) if isinstance(item, Mapping))
    staged: list[tuple[int, BlockKind, str, str, int]] = []
    counts: Counter[tuple[BlockKind, str]] = Counter()
    repeated_pages: dict[tuple[BlockKind, str], set[int]] = {}
    for index, element in enumerate(elements):
        page = int(element.get("page") or element.get("page_number") or 0)
        if page <= 0 or page > pages:
            raise ValueError("element page outside declared page range")
        kind = _KINDS.get(str(element.get("category", "")).lower(), "other")
        raw = _text(element.get("content", ""), preserve_structure=kind == "table")
        content = _normalized(raw)
        staged.append((page, kind, content, raw, index))
        if kind in ("header", "footer") and content:
            key = (kind, content)
            counts[key] += 1
            repeated_pages.setdefault(key, set()).add(page)
    repeated = {
        key for key, seen in repeated_pages.items() if len(seen) >= 3 and counts[key] == len(seen)
    }
    blocks: list[DocumentBlock] = []
    section_parts: list[str] = []
    staged_by_page: dict[int, list[tuple[int, BlockKind, str, str, int]]] = {}
    for item in staged:
        staged_by_page.setdefault(item[0], []).append(item)
    for page in range(1, pages + 1):
        page_items = sorted(staged_by_page.get(page, []), key=lambda item: item[4])
        if not page_items:
            identity = canonical_json_hash(
                {"parse": snapshot, "document": document, "empty_page": page}
            )
            blocks.append(
                DocumentBlock(
                    identity,
                    document,
                    snapshot,
                    source_hash,
                    page,
                    tuple(section_parts),
                    "empty_page",
                    "",
                    "",
                    (),
                    False,
                )
            )
            continue
        for _, kind, content, raw, index in page_items:
            if kind == "heading" and content:
                level = _heading_level(content)
                section_parts = section_parts[: level - 1]
                section_parts.append(_heading_label(content))
            identity = canonical_json_hash(
                {"parse": snapshot, "document": document, "element": index, "content": content}
            )
            blocks.append(
                DocumentBlock(
                    identity,
                    document,
                    snapshot,
                    source_hash,
                    page,
                    tuple(section_parts),
                    kind,
                    content,
                    raw,
                    (index,),
                    (kind, content) in repeated,
                )
            )
    return blocks


def reconstruct_normalized_text(
    blocks: Sequence[DocumentBlock], *, include_boilerplate: bool = False
) -> str:
    return "\n\n".join(
        block.content for block in blocks if include_boilerplate or not block.is_boilerplate
    )
