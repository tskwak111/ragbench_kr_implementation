"""Behavioral tests for the corpus provenance and freeze gate."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest
import yaml

from ragbench.ingestion.manifest import CorpusManifest, CorpusManifestValidationError


def _write_pdf(path: Path, *, pages: int = 1) -> str:
    """Create a small structurally valid PDF without test-only runtime dependencies."""
    page_references = " ".join(f"{number} 0 R" for number in range(3, pages + 3))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{page_references}] /Count {pages} >>".encode(),
    ]
    objects.extend(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>" for _ in range(pages))
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode())
        payload.extend(body)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    payload.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
    payload.extend(f"{trailer}startxref\n{xref_offset}\n%%EOF\n".encode())
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _document(
    path: Path, sha256: str, *, document_id: str = "annual-report-2025"
) -> dict[str, object]:
    return {
        "document_id": document_id,
        "title": "2025 Annual Report",
        "organization": "Example Corporation",
        "year": 2025,
        "document_type": "annual_report",
        "language": "ko",
        "sector": "corporate",
        "content_stratum": "mixed",
        "template_family": "example-annual-report",
        "source_url": "https://example.test/reports/2025.pdf",
        "downloaded_at": str(date(2026, 8, 14)),
        "license": "CC-BY-4.0",
        "redistribution_status": "redistributable",
        "local_path": str(path),
        "sha256": sha256,
        "page_count": 1,
        "inclusion_rationale": "Korean report with narrative and tabular financial sections.",
    }


def _write_manifest(path: Path, documents: list[dict[str, object]], **targets: object) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "status": "draft",
                "targets": {
                    "minimum_documents": 1,
                    "maximum_documents": 30,
                    "minimum_pages": 1,
                    "maximum_pages": 2_000,
                    "maximum_organization_share": 1.0,
                    "maximum_template_family_share": 1.0,
                    **targets,
                },
                "documents": documents,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_snapshot_is_stable_across_document_order_and_local_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first_hash = _write_pdf(first)
    second_hash = _write_pdf(second, pages=2)
    first_document = _document(first, first_hash, document_id="first")
    second_document = _document(second, second_hash, document_id="second")

    first_manifest_path = tmp_path / "first.yaml"
    second_manifest_path = tmp_path / "second.yaml"
    _write_manifest(first_manifest_path, [first_document, second_document])
    alternate_first = {**first_document, "local_path": "/operator-private/first.pdf"}
    alternate_second = {**second_document, "local_path": "/operator-private/second.pdf"}
    _write_manifest(second_manifest_path, [alternate_second, alternate_first])

    assert (
        CorpusManifest.load(first_manifest_path).corpus_snapshot_id
        == CorpusManifest.load(second_manifest_path).corpus_snapshot_id
    )


def test_freeze_rejects_duplicate_document_content(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    sha256 = _write_pdf(pdf)
    manifest_path = tmp_path / "corpus.yaml"
    _write_manifest(
        manifest_path,
        [_document(pdf, sha256, document_id="one"), _document(pdf, sha256, document_id="two")],
        minimum_documents=2,
        minimum_pages=2,
    )

    with pytest.raises(CorpusManifestValidationError, match="duplicate SHA-256"):
        CorpusManifest.load(manifest_path).validate(freeze=True)


def test_freeze_requires_source_and_download_provenance(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    sha256 = _write_pdf(pdf)
    document = _document(pdf, sha256)
    document["source_url"] = ""
    document["downloaded_at"] = ""
    manifest_path = tmp_path / "corpus.yaml"
    _write_manifest(manifest_path, [document])

    with pytest.raises(CorpusManifestValidationError, match="source_url"):
        CorpusManifest.load(manifest_path).validate(freeze=True)


def test_freeze_rejects_unknown_or_unsupported_redistribution_status(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    sha256 = _write_pdf(pdf)
    unknown = _document(pdf, sha256)
    unknown["redistribution_status"] = "unknown"
    unknown_manifest = tmp_path / "unknown.yaml"
    _write_manifest(unknown_manifest, [unknown])

    with pytest.raises(CorpusManifestValidationError, match="unknown redistribution status"):
        CorpusManifest.load(unknown_manifest).validate(freeze=True)

    unsupported = _document(pdf, sha256)
    unsupported["redistribution_status"] = "publisher-approved"
    unsupported_manifest = tmp_path / "unsupported.yaml"
    _write_manifest(unsupported_manifest, [unsupported])

    with pytest.raises(CorpusManifestValidationError, match="redistributable"):
        CorpusManifest.load(unsupported_manifest)


def test_validation_rejects_mismatched_page_count_and_malformed_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    sha256 = _write_pdf(pdf)
    wrong_pages = _document(pdf, sha256)
    wrong_pages["page_count"] = 2
    wrong_manifest_path = tmp_path / "wrong-pages.yaml"
    _write_manifest(wrong_manifest_path, [wrong_pages])

    with pytest.raises(CorpusManifestValidationError, match="page count"):
        CorpusManifest.load(wrong_manifest_path).validate()

    malformed = tmp_path / "malformed.pdf"
    malformed.write_bytes(b"not a PDF")
    malformed_document = _document(malformed, hashlib.sha256(malformed.read_bytes()).hexdigest())
    malformed_manifest_path = tmp_path / "malformed.yaml"
    _write_manifest(malformed_manifest_path, [malformed_document])

    with pytest.raises(CorpusManifestValidationError, match="valid PDF"):
        CorpusManifest.load(malformed_manifest_path).validate()


def test_freeze_enforces_document_page_and_organization_targets(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    sha256 = _write_pdf(pdf)
    manifest_path = tmp_path / "corpus.yaml"
    _write_manifest(
        manifest_path,
        [_document(pdf, sha256)],
        minimum_documents=2,
        minimum_pages=3,
        maximum_organization_share=0.5,
    )

    with pytest.raises(CorpusManifestValidationError, match="minimum_documents"):
        CorpusManifest.load(manifest_path).validate(freeze=True)


def test_public_export_excludes_local_paths_and_document_bytes(tmp_path: Path) -> None:
    pdf = tmp_path / "restricted.pdf"
    sha256 = _write_pdf(pdf)
    document = _document(pdf, sha256)
    document["redistribution_status"] = "nonredistributable"
    manifest_path = tmp_path / "corpus.yaml"
    _write_manifest(manifest_path, [document])
    manifest = CorpusManifest.load(manifest_path)

    exported = manifest.public_export()

    assert exported["documents"][0]["source_url"] == document["source_url"]
    assert exported["documents"][0]["sha256"] == sha256
    assert "local_path" not in exported["documents"][0]
    assert "document_bytes" not in exported["documents"][0]


def test_approved_operational_manifest_is_populated_and_frozen() -> None:
    manifest_path = Path(__file__).parents[3] / "configs" / "corpus.yaml"
    manifest = CorpusManifest.load(manifest_path)

    assert len(manifest.documents) == 20
    assert sum(document.page_count for document in manifest.documents) == 1_981
    assert manifest.public_export()["status"] == "frozen"


def test_frozen_status_runs_freeze_validation_without_an_explicit_flag(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    sha256 = _write_pdf(pdf)
    document = _document(pdf, sha256)
    document["source_url"] = ""
    document["downloaded_at"] = ""
    manifest_path = tmp_path / "frozen.yaml"
    _write_manifest(manifest_path, [document])
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw["status"] = "frozen"
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(CorpusManifestValidationError, match="source_url"):
        CorpusManifest.load(manifest_path).validate()


def test_freeze_requires_sector_strata_and_template_family_distribution(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    sha256 = _write_pdf(pdf)
    manifest_path = tmp_path / "corpus.yaml"
    _write_manifest(
        manifest_path,
        [_document(pdf, sha256)],
        maximum_organization_share=1.0,
        maximum_template_family_share=0.5,
    )
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw["status"] = "frozen"
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(CorpusManifestValidationError, match="public sector"):
        CorpusManifest.load(manifest_path).validate()


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "duplicate.yaml"
    manifest_path.write_text(
        "schema_version: 1\nschema_version: 1\ndocuments: []\n", encoding="utf-8"
    )

    with pytest.raises(CorpusManifestValidationError, match="duplicate YAML key"):
        CorpusManifest.load(manifest_path)


def test_snapshot_order_is_deterministic_for_full_records_with_same_hash(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    sha256 = _write_pdf(pdf)
    first = _document(pdf, sha256, document_id="first")
    second = {**_document(pdf, sha256, document_id="second"), "title": "Other title"}
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    _write_manifest(first_path, [first, second])
    _write_manifest(second_path, [second, first])

    assert (
        CorpusManifest.load(first_path).corpus_snapshot_id
        == CorpusManifest.load(second_path).corpus_snapshot_id
    )
