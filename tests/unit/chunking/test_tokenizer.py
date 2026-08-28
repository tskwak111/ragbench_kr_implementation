import time

import pytest

import ragbench.chunking.tokenizer as tokenizer


def test_tokenizer_loads_without_registry_or_network(monkeypatch):
    monkeypatch.setattr(
        tokenizer.tiktoken, "get_encoding", lambda name: (_ for _ in ()).throw(AssertionError(name))
    )
    assert tokenizer.encoding().decode(tokenizer.encoding().encode("오프라인")) == "오프라인"


def test_tokenizer_asset_checksum_mismatch_fails_closed(tmp_path, monkeypatch):
    asset = tmp_path / "cl100k_base.tiktoken"
    asset.write_bytes(b"tampered")
    monkeypatch.setattr(tokenizer, "TOKENIZER_ASSET", asset)
    with pytest.raises(RuntimeError, match="checksum"):
        tokenizer._load_encoding()


def test_safe_token_boundaries_scale_to_long_documents():
    pieces = [b"a"] * 300_000

    started = time.perf_counter()
    offsets, safe = tokenizer._safe_token_boundaries(pieces)
    elapsed = time.perf_counter() - started

    assert offsets[-1] == 300_000
    assert safe[300_000] == 300_000
    assert elapsed < 0.75


def test_safe_token_windows_scale_to_long_documents():
    text = "a " * 80_000

    started = time.perf_counter()
    windows = tokenizer.safe_token_windows(text, 300, 100)
    elapsed = time.perf_counter() - started

    assert windows[0].token_start == 0
    assert windows[-1].token_end == 80_001
    assert elapsed < 0.75
