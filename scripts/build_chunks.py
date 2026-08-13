#!/usr/bin/env python3
"""Build immutable JSONL chunk snapshots from complete offline parse checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ragbench.chunking.fixed import FixedChunker
from ragbench.chunking.heading import HeadingAwareChunker
from ragbench.chunking.tokenizer import tokenizer_snapshot
from ragbench.core.hashing import canonical_json_hash
from ragbench.ingestion.normalizer import normalize

FIXED_VARIANTS = ((300, 0), (300, 100), (600, 0), (600, 100), (1000, 0), (1000, 100))


class BuildIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    snapshot_id: str
    mode: str
    strategy: str
    path: Path
    chunks: int
    documents: int
    token_distribution: dict[str, float | int]


@dataclass(frozen=True, slots=True)
class BuildResult:
    datasets: tuple[DatasetArtifact, ...]
    metadata_path: Path


def _validate(
    checkpoints: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, list[Mapping[str, Any]]]]:
    if not checkpoints or any(item.get("status") != "succeeded" for item in checkpoints):
        raise BuildIntegrityError("all checkpoints must be successful")
    corpus = {str(item.get("snapshot_id")) for item in checkpoints}
    if len(corpus) != 1:
        raise BuildIntegrityError("mixed corpus snapshots are forbidden")
    by_mode: dict[str, list[Mapping[str, Any]]] = {"standard": [], "enhanced": []}
    for item in checkpoints:
        mode = str(item.get("mode"))
        if mode not in by_mode:
            raise BuildIntegrityError("unexpected parse mode")
        expected_pages = int(item.get("expected_pages") or 0)
        mappings = item.get("page_mappings")
        if not isinstance(mappings, Sequence) or isinstance(mappings, (str, bytes)):
            raise BuildIntegrityError("complete page mappings are required")
        mapped_pages = {
            int(mapping.get("source_page") or mapping.get("page") or 0)
            for mapping in mappings
            if isinstance(mapping, Mapping)
        }
        if expected_pages <= 0 or mapped_pages != set(range(1, expected_pages + 1)):
            raise BuildIntegrityError("page mappings are incomplete")
        by_mode[mode].append(item)
    source_sets = [
        {(str(item.get("document_id")), str(item.get("source_sha256"))) for item in by_mode[mode]}
        for mode in ("standard", "enhanced")
    ]
    if not source_sets[0] or source_sets[0] != source_sets[1]:
        raise BuildIntegrityError("standard and enhanced must be complete for the identical corpus")
    for mode, items in by_mode.items():
        identities = [
            (str(item.get("document_id")), str(item.get("source_sha256"))) for item in items
        ]
        if len(identities) != len(set(identities)):
            raise BuildIntegrityError(f"duplicate {mode} document checkpoint")
        parse_ids = {str(item.get("parse_snapshot_id")) for item in items}
        if len(parse_ids) != 1 or not next(iter(parse_ids)):
            raise BuildIntegrityError(f"mixed {mode} parse snapshots are forbidden")
    return next(iter(corpus)), by_mode


def _distribution(counts: list[int]) -> dict[str, float | int]:
    if not counts:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(counts), "max": max(counts), "mean": sum(counts) / len(counts)}


def _write_immutable(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise BuildIntegrityError(
                f"immutable snapshot already exists with different data: {path}"
            )
        return
    path.write_text(payload, encoding="utf-8")
    os.chmod(path, 0o600)


def build_chunk_snapshots(
    checkpoints: Sequence[Mapping[str, Any]], output_dir: Path
) -> BuildResult:
    corpus, by_mode = _validate(checkpoints)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    artifacts: list[DatasetArtifact] = []
    factories: list[Callable[[], FixedChunker | HeadingAwareChunker]] = [
        *(lambda s=s, o=o: FixedChunker(s, o) for s, o in FIXED_VARIANTS),
        lambda: HeadingAwareChunker(),
    ]
    for mode in ("standard", "enhanced"):
        for factory in factories:
            chunker = factory()
            records = []
            for checkpoint in sorted(by_mode[mode], key=lambda item: str(item["document_id"])):
                records.extend(chunker.split(normalize(checkpoint)))
            snapshot = canonical_json_hash(
                {
                    "corpus": corpus,
                    "parse_snapshot": by_mode[mode][0]["parse_snapshot_id"],
                    "mode": mode,
                    "strategy_hash": chunker.strategy_hash,
                    "tokenizer": tokenizer_snapshot(),
                }
            )
            path = output_dir / f"{snapshot}.jsonl"
            payload = "".join(
                json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            )
            _write_immutable(path, payload)
            artifacts.append(
                DatasetArtifact(
                    snapshot,
                    mode,
                    chunker.strategy,
                    path,
                    len(records),
                    len(by_mode[mode]),
                    _distribution([item.token_count for item in records]),
                )
            )
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "corpus_snapshot_id": corpus,
        "tokenizer": tokenizer_snapshot(),
        "datasets": [{**asdict(item), "path": item.path.name} for item in artifacts],
        "strategy_counts": dict(Counter(item.strategy for item in artifacts)),
    }
    _write_immutable(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2),
    )
    return BuildResult(tuple(artifacts), metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    checkpoints = [
        json.loads(line)
        for line in args.checkpoint_jsonl.read_text(encoding="utf-8").splitlines()
        if line
    ]
    result = build_chunk_snapshots(checkpoints, args.output_dir)
    print(
        json.dumps(
            {"metadata": str(result.metadata_path), "datasets": len(result.datasets)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
