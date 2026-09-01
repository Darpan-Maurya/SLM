#!/usr/bin/env python
"""Tokenise a corpus into training shards."""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from slm.data.prepare import iter_source, tokenize_to_shards


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="file, directory, or hf://dataset:split")
    p.add_argument("-o", "--out", default="data/tokens/train")
    p.add_argument("-t", "--tokenizer", default="data/tokenizer.json")
    p.add_argument("--val-dir", default="")
    p.add_argument("--val-fraction", type=float, default=0.005)
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-docs", type=int, default=0)
    p.add_argument("--text-key", default="text")
    p.add_argument("--split-blank", action="store_true",
                   help="treat blank lines as document boundaries in plain text")
    args = p.parse_args()

    docs = iter_source(args.source, args.text_key, limit=args.max_docs)
    if args.split_blank:
        docs = (
            part + "\n\n"
            for doc in docs
            for part in doc.split("\n\n")
            if part.strip()
        )
    tokenize_to_shards(
        docs, args.tokenizer, args.out,
        shard_tokens=args.shard_tokens, workers=args.workers,
        val_fraction=args.val_fraction,
        val_dir=args.val_dir or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
