"""Versioned corpus manifests, provenance checks, and the freeze gate."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator
from pypdf import PdfReader

from ragbench.core.hashing import canonical_json_hash

RedistributionStatus = Literal["redistributable", "nonredistributable", "unknown"]
Sector = Literal["corporate", "public"]
ContentStratum = Literal["table_heavy", "text_heavy", "mixed"]


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses ambiguous duplicate mapping keys."""


def _construct_mapping_with_unique_keys(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate YAML key: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_with_unique_keys,
)


class CorpusManifestValidationError(ValueError):
    """Raised when a manifest cannot meet the requested validation gate."""


class DocumentRecord(BaseModel):
    """One locally acquired PDF and the evidence supporting its use."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    title: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    year: int = Field(ge=1900, le=2100)
    document_type: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=16)
    sector: Sector
    content_stratum: ContentStratum
    template_family: str = Field(min_length=1)
    source_url: str = ""
    downloaded_at: str = ""
    license: str = Field(min_length=1)
    redistribution_status: RedistributionStatus
    local_path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: PositiveInt
    inclusion_rationale: str = Field(min_length=1)

    @field_validator("source_url")
    @classmethod
    def validate_source_url_if_provided(cls, value: str) -> str:
        if not value:
            return value
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            msg = "source_url must be an absolute HTTP(S) URL"
            raise ValueError(msg)
        return value

    @field_validator("downloaded_at")
    @classmethod
    def validate_downloaded_at_if_provided(cls, value: str) -> str:
        if not value:
            return value
        try:
            date.fromisoformat(value)
        except ValueError as error:
            msg = "downloaded_at must use ISO-8601 YYYY-MM-DD"
            raise ValueError(msg) from error
        return value

    def snapshot_metadata(self) -> dict[str, Any]:
        """Return metadata that identifies corpus content, excluding operator-local paths."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "organization": self.organization,
            "year": self.year,
            "document_type": self.document_type,
            "language": self.language,
            "sector": self.sector,
            "content_stratum": self.content_stratum,
            "template_family": self.template_family,
            "source_url": self.source_url,
            "downloaded_at": self.downloaded_at,
            "license": self.license,
            "redistribution_status": self.redistribution_status,
            "sha256": self.sha256,
            "page_count": self.page_count,
            "inclusion_rationale": self.inclusion_rationale,
        }


class CorpusTargets(BaseModel):
    """Measured corpus diversity and size requirements used by the freeze gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_documents: int = Field(default=20, ge=1)
    maximum_documents: int = Field(default=30, ge=1)
    minimum_pages: int = Field(default=1500, ge=1)
    maximum_pages: int = Field(default=2000, ge=1)
    maximum_organization_share: float = Field(default=0.35, gt=0, le=1)
    maximum_template_family_share: float = Field(default=0.35, gt=0, le=1)


class CorpusManifestFile(BaseModel):
    """On-disk YAML model, deliberately independent of local file availability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    status: Literal["draft", "frozen"] = "draft"
    targets: CorpusTargets = Field(default_factory=CorpusTargets)
    documents: tuple[DocumentRecord, ...] = ()


@dataclass(frozen=True)
class ManifestValidationResult:
    """A successful validation result with immutable aggregate facts."""

    corpus_snapshot_id: str
    document_count: int
    page_count: int


class CorpusManifest:
    """A loaded manifest whose explicit ``freeze=True`` validation is the freeze gate."""

    def __init__(self, file: CorpusManifestFile, *, source_path: Path) -> None:
        self._file = file
        self.source_path = source_path

    @classmethod
    def load(cls, path: Path) -> CorpusManifest:
        """Load a YAML manifest without touching local PDFs."""
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError as error:
            msg = f"cannot read corpus manifest {path}: {error}"
            raise CorpusManifestValidationError(msg) from error
        try:
            raw = yaml.load(contents, Loader=_UniqueKeyLoader)
        except yaml.YAMLError as error:
            msg = f"invalid corpus manifest {path}: {error}"
            raise CorpusManifestValidationError(msg) from error
        if raw is None:
            raw = {}
        try:
            file = CorpusManifestFile.model_validate(raw)
        except Exception as error:
            msg = f"invalid corpus manifest {path}: {error}"
            raise CorpusManifestValidationError(msg) from error
        return cls(file, source_path=path)

    @property
    def documents(self) -> tuple[DocumentRecord, ...]:
        """Return immutable document records in the supplied manifest order."""
        return self._file.documents

    @property
    def corpus_snapshot_id(self) -> str:
        """Hash stable metadata independently of input ordering and local paths."""
        documents = sorted(
            (item.snapshot_metadata() for item in self.documents),
            key=canonical_json_hash,
        )
        return canonical_json_hash(
            {"schema_version": self._file.schema_version, "documents": documents}
        )

    def validate(self, *, freeze: bool = False) -> ManifestValidationResult:
        """Validate local PDFs and, in freeze mode, required provenance and corpus targets."""
        freeze = freeze or self._file.status == "frozen"
        errors: list[str] = []
        ids = Counter(document.document_id for document in self.documents)
        hashes = Counter(document.sha256 for document in self.documents)
        errors.extend(
            f"duplicate document_id: {identifier}" for identifier, count in ids.items() if count > 1
        )
        errors.extend(
            f"duplicate SHA-256 content: {digest}" for digest, count in hashes.items() if count > 1
        )
        for document in self.documents:
            errors.extend(self._validate_local_pdf(document))
            if freeze:
                errors.extend(self._validate_freeze_provenance(document))
        if freeze:
            errors.extend(self._validate_freeze_targets())
            if self._file.status != "frozen":
                errors.append(
                    "manifest status is draft; set status to frozen only after human review"
                )
        if errors:
            raise CorpusManifestValidationError("; ".join(errors))
        return ManifestValidationResult(
            corpus_snapshot_id=self.corpus_snapshot_id,
            document_count=len(self.documents),
            page_count=sum(document.page_count for document in self.documents),
        )

    def public_export(self) -> dict[str, Any]:
        """Return public-safe provenance, omitting local filesystem locations and all bytes."""
        return {
            "schema_version": self._file.schema_version,
            "status": self._file.status,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "targets": self._file.targets.model_dump(mode="json"),
            "documents": [document.snapshot_metadata() for document in self.documents],
        }

    def export_public(self, destination: Path) -> None:
        """Atomically write a public-safe YAML manifest."""
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            yaml.safe_dump(self.public_export(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(destination)

    @staticmethod
    def _validate_local_pdf(document: DocumentRecord) -> list[str]:
        path = document.local_path
        if not path.is_file():
            return [f"{document.document_id}: local_path does not identify a regular file"]
        try:
            with path.open("rb") as stream:
                if stream.read(5) != b"%PDF-":
                    return [
                        f"{document.document_id}: local_path is not a valid PDF (missing PDF magic)"
                    ]
            with path.open("rb") as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_hash != document.sha256:
                return [f"{document.document_id}: SHA-256 does not match local PDF"]
            reader = PdfReader(path, strict=True)
            actual_pages = len(reader.pages)
        except Exception as error:
            return [f"{document.document_id}: local_path is not a valid PDF ({error})"]
        if actual_pages != document.page_count:
            return [
                f"{document.document_id}: page count {document.page_count} does not match "
                f"PDF page count {actual_pages}"
            ]
        return []

    @staticmethod
    def _validate_freeze_provenance(document: DocumentRecord) -> list[str]:
        errors: list[str] = []
        if not document.source_url:
            errors.append(f"{document.document_id}: source_url is required to freeze")
        if not document.downloaded_at:
            errors.append(f"{document.document_id}: downloaded_at is required to freeze")
        if document.redistribution_status == "unknown":
            errors.append(f"{document.document_id}: unknown redistribution status prevents freeze")
        return errors

    def _validate_freeze_targets(self) -> list[str]:
        targets = self._file.targets
        document_count = len(self.documents)
        page_count = sum(document.page_count for document in self.documents)
        errors: list[str] = []
        if document_count < targets.minimum_documents:
            errors.append(
                f"document count {document_count} is below minimum_documents "
                f"{targets.minimum_documents}"
            )
        if document_count > targets.maximum_documents:
            errors.append(
                f"document count {document_count} exceeds maximum_documents "
                f"{targets.maximum_documents}"
            )
        if page_count < targets.minimum_pages:
            errors.append(f"page count {page_count} is below minimum_pages {targets.minimum_pages}")
        if page_count > targets.maximum_pages:
            errors.append(f"page count {page_count} exceeds maximum_pages {targets.maximum_pages}")
        if document_count:
            organization_counts = Counter(document.organization for document in self.documents)
            largest = max(organization_counts.values()) / document_count
            if largest > targets.maximum_organization_share:
                errors.append(
                    "largest organization share "
                    f"{largest:.3f} exceeds maximum_organization_share "
                    f"{targets.maximum_organization_share:.3f}"
                )
            template_counts = Counter(document.template_family for document in self.documents)
            largest_template_share = max(template_counts.values()) / document_count
            if largest_template_share > targets.maximum_template_family_share:
                errors.append(
                    "largest template family share "
                    f"{largest_template_share:.3f} exceeds maximum_template_family_share "
                    f"{targets.maximum_template_family_share:.3f}"
                )
            sectors = {document.sector for document in self.documents}
            if "corporate" not in sectors:
                errors.append("freeze corpus requires at least one corporate sector document")
            if "public" not in sectors:
                errors.append("freeze corpus requires at least one public sector document")
            strata = {document.content_stratum for document in self.documents}
            if "table_heavy" not in strata:
                errors.append("freeze corpus requires at least one table_heavy document")
            if "text_heavy" not in strata:
                errors.append("freeze corpus requires at least one text_heavy document")
        return errors
