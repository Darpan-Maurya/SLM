#!/usr/bin/env python
"""Train a byte-level BPE tokenizer on a corpus."""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from slm.data.prepare import iter_source
from slm.tokenizer import Tokenizer

DEFAULT_SPECIALS = ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>", "<|pad|>")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="file, directory, or hf://dataset:split")
    p.add_argument("-o", "--out", default="data/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--max-docs", type=int, default=200_000,
                   help="documents sampled for training the merges")
    p.add_argument("--max-bytes", type=int, default=500_000_000)
    p.add_argument("--text-key", default="text")
    p.add_argument("--specials", nargs="*", default=list(DEFAULT_SPECIALS))
    args = p.parse_args()

    def limited():
        total = 0
        for doc in iter_source(args.source, args.text_key, limit=args.max_docs):
            total += len(doc)
            if total > args.max_bytes:
                return
            yield doc

    t0 = time.time()
    tok = Tokenizer.train(limited(), args.vocab_size,
                          special_tokens=tuple(args.specials), verbose=True)
    tok.save(args.out)
    print(f"[tokenizer] {tok} in {time.time()-t0:.1f}s -> {args.out}")

    sample = "".join(d for _, d in zip(range(20), iter_source(args.source, args.text_key)))
    if sample:
        print(f"[tokenizer] compression on a sample: {tok.compression(sample):.2f} bytes/token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
