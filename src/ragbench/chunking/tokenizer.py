"""Pinned, vendored tokenizer configuration and Unicode-safe token windows."""

from __future__ import annotations

import base64
import codecs
import hashlib
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import tiktoken

TOKENIZER_ENCODING = "cl100k_base"
TOKENIZER_LIBRARY = "tiktoken"
TOKENIZER_VERSION = version("tiktoken")
TOKENIZER_ASSET = Path(__file__).with_name("assets") / "cl100k_base.tiktoken"
TOKENIZER_ASSET_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
)
_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}


def tokenizer_snapshot() -> dict[str, str]:
    return {
        "library": TOKENIZER_LIBRARY,
        "version": TOKENIZER_VERSION,
        "encoding": TOKENIZER_ENCODING,
        "asset_sha256": TOKENIZER_ASSET_SHA256,
    }


@dataclass(frozen=True, slots=True)
class TokenWindow:
    content: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int


def _load_encoding() -> tiktoken.Encoding:
    payload = TOKENIZER_ASSET.read_bytes()
    if hashlib.sha256(payload).hexdigest() != TOKENIZER_ASSET_SHA256:
        raise RuntimeError("vendored tokenizer checksum mismatch")
    ranks: dict[bytes, int] = {}
    try:
        for line in payload.splitlines():
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    except (ValueError, TypeError) as error:
        raise RuntimeError("vendored tokenizer data is malformed") from error
    return tiktoken.Encoding(
        TOKENIZER_ENCODING,
        pat_str=_PATTERN,
        mergeable_ranks=ranks,
        special_tokens=_SPECIAL_TOKENS,
    )


@lru_cache(maxsize=1)
def encoding() -> tiktoken.Encoding:
    """Load the verified package asset without registry lookup or network access."""
    return _load_encoding()


def _safe_token_boundaries(token_bytes: list[bytes]) -> tuple[list[int], dict[int, int]]:
    offsets = [0]
    safe: dict[int, int] = {0: 0}
    decoder = codecs.getincrementaldecoder("utf-8")()
    characters = 0
    for index, piece in enumerate(token_bytes, start=1):
        offsets.append(offsets[-1] + len(piece))
        characters += len(decoder.decode(piece))
        if not decoder.getstate()[0]:
            safe[index] = characters
    decoder.decode(b"", final=True)
    return offsets, safe


def safe_token_windows(text: str, size: int, overlap: int) -> tuple[TokenWindow, ...]:
    """Split on whole-token positions that are also UTF-8 codepoint boundaries.

    A codepoint can span several tokenizer tokens. In that case a window may exceed ``size`` by
    at most the tokens required to reach its next valid UTF-8 boundary.
    """
    if size <= 0 or overlap < 0 or size <= overlap:
        raise ValueError("size must be positive and greater than non-negative overlap")
    tokenizer = encoding()
    tokens = tokenizer.encode(text)
    if not tokens:
        return ()
    pieces = [tokenizer.decode_single_token_bytes(token) for token in tokens]
    offsets, safe = _safe_token_boundaries(pieces)
    payload = text.encode("utf-8")
    if b"".join(pieces) != payload or len(tokens) not in safe:
        raise ValueError("tokenizer bytes do not reconstruct normalized text")
    safe_boundaries = tuple(safe)
    windows: list[TokenWindow] = []
    start = 0
    while start < len(tokens):
        nominal = min(start + size, len(tokens))
        if nominal == len(tokens):
            end = nominal
        else:
            next_boundary = bisect_right(safe_boundaries, start)
            end_index = max(next_boundary, bisect_right(safe_boundaries, nominal) - 1)
            end = safe_boundaries[end_index]
            while (
                end < len(tokens) and bisect_right(safe_boundaries, end - overlap) <= next_boundary
            ):
                end_index += 1
                end = safe_boundaries[end_index]
        content_bytes = payload[offsets[start] : offsets[end]]
        content = content_bytes.decode("utf-8")
        windows.append(TokenWindow(content, start, end, safe[start], safe[end]))
        if end == len(tokens):
            break
        next_start_index = bisect_right(safe_boundaries, end - overlap) - 1
        if safe_boundaries[next_start_index] <= start:
            raise RuntimeError("safe tokenizer window made no progress")
        start = safe_boundaries[next_start_index]
    return tuple(windows)
