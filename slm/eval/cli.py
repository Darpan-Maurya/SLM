"""`slm-eval`: validation perplexity and zero-shot benchmarks for a checkpoint."""
from __future__ import annotations

import argparse
import json
import os

from slm.data import ResumableLoader, TokenDataset, read_index
from slm.eval.harness import evaluate_docs, format_table, token_perplexity
from slm.eval.tasks import DEFAULT_SUITE, load_task
from slm.infer.load import load_model
from slm.tokenizer import Tokenizer
from slm.train.distributed import pick_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slm-eval", description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("-t", "--tokenizer", default="data/tokenizer.json")
    p.add_argument("--tasks", nargs="*", default=list(DEFAULT_SUITE))
    p.add_argument("--limit", type=int, default=1000, help="docs per task; 0 = all")
    p.add_argument("--val-dir", default="")
    p.add_argument("--val-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--metric", default="acc_norm")
    p.add_argument("-o", "--out", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = pick_device(args.device)
    model, cfg_dict = load_model(args.checkpoint, device)
    tok = Tokenizer.load(args.tokenizer)
    print(f"[eval] {model.num_params()/1e6:.1f}M params on {device}")

    results: dict[str, dict] = {}
    val_dir = args.val_dir or cfg_dict.get("data", {}).get("val_dir", "")
    if val_dir and os.path.isdir(val_dir):
        bpt = read_index(val_dir).get("bytes_per_token", 0.0)
        loader = ResumableLoader(TokenDataset(val_dir, model.cfg.max_seq_len), 8,
                                 num_workers=0, device=device, infinite=False)
        ppl = token_perplexity(model, loader, args.val_steps, bpt)
        loader.stop()
        if ppl:
            results["_perplexity"] = ppl
            print(f"[eval] val loss {ppl['loss']:.4f} | ppl {ppl['ppl']:.2f} "
                  f"| bpb {ppl.get('bpb', float('nan')):.4f}")

    for name in args.tasks:
        try:
            docs = load_task(name, args.limit)
        except KeyError as exc:
            print(f"[eval] {exc}")
            continue
        if not docs:
            print(f"[eval] {name}: unavailable, skipped")
            continue
        res = evaluate_docs(model, tok, docs, args.batch_size)
        results[name] = res
        print(f"[eval] {name:14s} acc {100*res['acc']:5.2f} | "
              f"acc_norm {100*res['acc_norm']:5.2f} | "
              f"random {100*res['random_baseline']:5.2f}")

    print("\n" + format_table(
        {k: v for k, v in results.items() if not k.startswith("_")}, args.metric))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[eval] written to {args.out}")
    return 0
