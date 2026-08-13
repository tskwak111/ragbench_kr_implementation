#!/usr/bin/env python3
"""Plan the fixed retrieval-screen grid; execution is deliberately opt-in."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ragbench.core.hashing import canonical_json_hash
from ragbench.experiments.planner import generate_core_retrieval_configs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-snapshot-id", required=True)
    parser.add_argument("--question-snapshot-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute only when a concrete store and immutable snapshot loader are configured",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configs = generate_core_retrieval_configs(
        corpus_snapshot_id=args.corpus_snapshot_id,
        question_snapshot_id=args.question_snapshot_id,
        code_commit=args.code_commit,
        random_seed=args.random_seed,
    )
    if args.execute:
        raise SystemExit(
            "real retrieval screening is not available without an explicitly bound "
            "snapshot loader, retriever factory, and transactional store"
        )
    hashes = tuple(config.semantic_hash for config in configs)
    print(
        json.dumps(
            {
                "mode": "dry-run",
                "schema_version": "retrieval-screen-v1",
                "config_count": len(configs),
                "grid_hash": canonical_json_hash(hashes),
                "config_hashes": hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
