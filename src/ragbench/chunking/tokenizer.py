"""Pinned, vendored tokenizer configuration and Unicode-safe token windows."""

from __future__ import annotations

import base64
import hashlib
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
    for piece in token_bytes:
        offsets.append(offsets[-1] + len(piece))
    safe: dict[int, int] = {0: 0}
    prefix = bytearray()
    for index, piece in enumerate(token_bytes, start=1):
        prefix.extend(piece)
        try:
            safe[index] = len(prefix.decode("utf-8"))
        except UnicodeDecodeError:
            continue
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
    windows: list[TokenWindow] = []
    start = 0
    while start < len(tokens):
        nominal = min(start + size, len(tokens))
        if nominal == len(tokens):
            end = nominal
        else:
            end_candidates = [boundary for boundary in safe if boundary > start]
            before = [boundary for boundary in end_candidates if boundary <= nominal]
            end = max(before) if before else min(end_candidates)
            while end < len(tokens):
                safe_starts = [boundary for boundary in safe if start < boundary <= end - overlap]
                if safe_starts:
                    break
                end = min(boundary for boundary in safe if boundary > end)
        content_bytes = payload[offsets[start] : offsets[end]]
        content = content_bytes.decode("utf-8")
        windows.append(TokenWindow(content, start, end, safe[start], safe[end]))
        if end == len(tokens):
            break
        safe_starts = [boundary for boundary in safe if start < boundary <= end - overlap]
        if not safe_starts:
            raise RuntimeError("safe tokenizer window made no progress")
        next_start = max(safe_starts)
        start = next_start
    return tuple(windows)
