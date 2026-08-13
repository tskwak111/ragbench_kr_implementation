"""Canonical hashing for reproducible identifiers and cache keys."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID


def _canonicalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(normalized, key=_canonical_json)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (UUID, Path, Enum)):
        return str(value.value if isinstance(value, Enum) else value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite floats cannot be hashed canonically")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    msg = f"unsupported canonical JSON type: {type(value).__name__}"
    raise TypeError(msg)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_hash(value: Any) -> str:
    """Return a SHA-256 digest of a stable, UTF-8 canonical JSON representation."""
    payload = _canonical_json(_canonicalize(value)).encode()
    return hashlib.sha256(payload).hexdigest()
