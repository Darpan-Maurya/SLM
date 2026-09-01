#!/usr/bin/env python
"""Thin wrapper; the implementation lives in slm.infer.cli."""
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from slm.infer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
