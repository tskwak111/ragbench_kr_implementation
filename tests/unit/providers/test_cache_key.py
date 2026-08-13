"""Deterministic provider request hashing contracts."""

from ragbench.core.hashing import canonical_json_hash
from ragbench.providers.upstage.client import CacheKeyParts


def test_canonical_hash_is_stable_across_mapping_key_order() -> None:
    """Catch serializing mappings in insertion order instead of canonical order."""
    left = {"model": "solar-pro4", "params": {"temperature": 0, "top_p": 0.9}}
    right = {"params": {"top_p": 0.9, "temperature": 0}, "model": "solar-pro4"}

    assert canonical_json_hash(left) == canonical_json_hash(right)


def test_cache_key_covers_every_billable_request_dimension_without_secret() -> None:
    """Catch cache aliases and accidental inclusion of provider credentials."""
    parts = CacheKeyParts(
        operation="generate",
        model_id="solar-pro4",
        provider_params={"temperature": 0},
        prompt_hash=canonical_json_hash("질문"),
        context_hash=canonical_json_hash(["근거"]),
        document_sha256=None,
        schema_version="provider-cache-v1",
    )
    key = parts.digest()

    assert len(key) == 64
    assert (
        key
        != CacheKeyParts(
            operation="generate",
            model_id="solar-pro3",
            provider_params={"temperature": 0},
            prompt_hash=parts.prompt_hash,
            context_hash=parts.context_hash,
            document_sha256=None,
            schema_version="provider-cache-v1",
        ).digest()
    )
    assert "secret-api-key" not in key
