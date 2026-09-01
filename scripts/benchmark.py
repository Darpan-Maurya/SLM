#!/usr/bin/env python
"""Throughput and MFU for one or more configs on this machine."""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from slm.config import load_config
from slm.train.benchmark import benchmark, format_result
from slm.train.distributed import pick_device

DEFAULTS = [
    "configs/train/ablation.yaml",
    "configs/train/pretrain-300m.yaml",
    "configs/train/pretrain-moe-500m.yaml",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("configs", nargs="*", default=DEFAULTS)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args(argv)

    device = pick_device(args.device)
    print(f"[bench] device {device}\n")
    for path in args.configs:
        cfg = load_config([path], args.override)
        name = cfg.train.run_name
        try:
            res = benchmark(cfg, device, args.steps, args.warmup)
        except (RuntimeError, MemoryError) as exc:
            # an OOM here is a result, not a crash: it tells you this config does
            # not fit and micro_batch_size or grad_checkpoint needs adjusting
            print(f"{name:16s} failed: {str(exc).splitlines()[0]}")
            continue
        print(format_result(name, res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
