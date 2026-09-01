#!/usr/bin/env python
"""Run the ablation grid: one config key changed per experiment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

BASE = "configs/train/ablation.yaml"

# name -> (description, overrides). The baseline is the config as written.
ABLATIONS: dict[str, tuple[str, list[str]]] = {
    "baseline": ("as configured in ablation.yaml", []),
    "A1_adamw": ("AdamW instead of hybrid Muon", ["optim.kind=adamw"]),
    "A2_vocab_49k": ("49152 vocab instead of 32768", ["model.vocab_size=49152"]),
    "A3_mha": ("full multi-head attention", ["model.n_kv_head=12"]),
    "A3_mla": ("multi-head latent attention", ["model.attn_type=mla",
                                               "model.n_kv_head=12"]),
    "A4_mtp": ("multi-token prediction depth 1", ["model.mtp_depth=1"]),
    "A5_full_rope": ("RoPE on every layer (no NoPE)", ["model.nope_every=0"]),
    "A6_no_qk_norm": ("QK-norm disabled", ["model.qk_norm=false"]),
    "A7_moe": ("sparse MoE at matched active FLOPs",
               ["model.moe=true", "model.n_experts=16", "model.n_experts_active=2",
                "model.expert_hidden=256"]),
    "A8_no_doc_mask": ("cross-document attention allowed", ["model.doc_masking=false"]),
    "A9_cosine": ("cosine schedule instead of WSD", ["optim.schedule=cosine"]),
    "A10_no_zloss": ("logit z-loss disabled", ["optim.zloss=0.0"]),
}


def run_one(name: str, overrides: list[str], extra: list[str], dry: bool) -> dict:
    cmd = [sys.executable, "scripts/train.py", BASE,
           f"train.run_name=ablation-{name}", *overrides, *extra]
    print(f"\n=== {name}: {' '.join(overrides) or 'baseline'} ===")
    print("  " + " ".join(cmd))
    if dry:
        return {"name": name, "skipped": True}
    t0 = time.time()
    proc = subprocess.run(cmd)
    return {"name": name, "overrides": overrides, "returncode": proc.returncode,
            "seconds": round(time.time() - t0, 1)}


def collect(name: str) -> dict:
    """Final train and val loss from a finished ablation run."""
    path = f"runs/ablation-{name}/logs/metrics.jsonl"
    if not os.path.exists(path):
        return {}
    train_loss = val_loss = None
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "train/loss" in rec:
                train_loss = rec["train/loss"]
            if "val/loss" in rec:
                val_loss = rec["val/loss"]
    return {"final_train_loss": train_loss, "final_val_loss": val_loss}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="*", help="ablations to run; default: all")
    p.add_argument("--all", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", action="store_true", help="summarise finished runs only")
    p.add_argument("--override", nargs="*", default=[], help="extra key=value for every run")
    args = p.parse_args()

    if args.list:
        for name, (desc, ov) in ABLATIONS.items():
            print(f"{name:18s} {desc:45s} {' '.join(ov)}")
        return 0

    names = args.names or (list(ABLATIONS) if args.all else [])
    if not names and not args.report:
        print("nothing to do: pass names, --all, --list or --report")
        return 1

    if args.report:
        names = names or list(ABLATIONS)
        rows = ["| ablation | final train | final val | delta vs baseline |",
                "|---|---|---|---|"]
        base = collect("baseline").get("final_val_loss")
        for name in names:
            res = collect(name)
            if not res:
                continue
            v = res.get("final_val_loss")
            delta = f"{v - base:+.4f}" if (v is not None and base is not None) else "-"
            rows.append(f"| {name} | {res.get('final_train_loss', float('nan')):.4f} "
                        f"| {v if v is None else f'{v:.4f}'} | {delta} |")
        print("\n".join(rows))
        return 0

    results = []
    for name in names:
        if name not in ABLATIONS:
            print(f"unknown ablation {name!r}; --list to see them")
            return 1
        _, overrides = ABLATIONS[name]
        results.append(run_one(name, overrides, args.override, args.dry_run))

    failed = [r for r in results if r.get("returncode")]
    print(f"\n{len(results) - len(failed)}/{len(results)} ablations completed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
