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
