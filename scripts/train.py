#!/usr/bin/env python
"""Train a model. Usage: train.py [config.yaml ...] [key=value ...]"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from slm.config import load_config
from slm.train.trainer import Trainer


def main(argv: list[str]) -> int:
    configs = [a for a in argv if a.endswith((".yaml", ".yml"))]
    overrides = [a for a in argv if a not in configs]
    cfg = load_config(configs, overrides)
    return Trainer(cfg).train()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
