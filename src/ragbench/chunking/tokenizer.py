"""Pinned tokenizer configuration and Unicode-safe token windows."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version

import tiktoken

TOKENIZER_ENCODING = "cl100k_base"
TOKENIZER_LIBRARY = "tiktoken"
TOKENIZER_VERSION = version("tiktoken")


def tokenizer_snapshot() -> dict[str, str]:
    return {
        "library": TOKENIZER_LIBRARY,
        "version": TOKENIZER_VERSION,
        "encoding": TOKENIZER_ENCODING,
    }


@dataclass(frozen=True, slots=True)
class TokenWindow:
    content: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int


def encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(TOKENIZER_ENCODING)


def safe_token_windows(text: str, size: int, overlap: int) -> tuple[TokenWindow, ...]:
    if size <= 0 or overlap < 0 or size <= overlap:
        raise ValueError("size must be positive and greater than non-negative overlap")
    tokens = encoding().encode(text)
    if not tokens:
        return ()
    decoded, offsets = encoding().decode_with_offsets(tokens)
    if decoded != text:
        raise ValueError("tokenizer round-trip changed normalized text")
    windows: list[TokenWindow] = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        char_start = offsets[start]
        char_end = offsets[end] if end < len(tokens) else len(text)
        if char_end <= char_start:
            end += 1
            while end < len(tokens) and offsets[end] <= char_start:
                end += 1
            char_end = offsets[end] if end < len(tokens) else len(text)
        content = text[char_start:char_end]
        while len(encoding().encode(content)) > size and char_end > char_start:
            char_end -= 1
            content = text[char_start:char_end]
        if content:
            windows.append(TokenWindow(content, start, end, char_start, char_end))
        if end == len(tokens):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return tuple(windows)
