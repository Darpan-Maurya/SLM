"""Console entry points declared in pyproject."""
from __future__ import annotations

import sys


def train_main() -> int:
    from slm.config import load_config
    from slm.train.trainer import Trainer

    argv = sys.argv[1:]
    configs = [a for a in argv if a.endswith((".yaml", ".yml"))]
    overrides = [a for a in argv if a not in configs]
    return Trainer(load_config(configs, overrides)).train()


def eval_main() -> int:
    from slm.eval.cli import main

    return main(sys.argv[1:])


def generate_main() -> int:
    from slm.infer.cli import main

    return main(sys.argv[1:])


def chat_main() -> int:
    from slm.infer.cli import chat_main as run

    return run(sys.argv[1:])
