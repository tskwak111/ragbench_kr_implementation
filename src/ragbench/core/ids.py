"""Deterministic identifiers derived from immutable content."""

from typing import Any
from uuid import UUID, uuid5

from ragbench.core.hashing import canonical_json_hash

RAGBENCH_NAMESPACE = UUID("f3cc3380-aaf1-5c8e-8c6d-8f464c7ef0fb")


def deterministic_uuid(value: Any, *, namespace: UUID = RAGBENCH_NAMESPACE) -> UUID:
    """Derive a stable UUIDv5 from any canonically hashable value."""
    return uuid5(namespace, canonical_json_hash(value))
