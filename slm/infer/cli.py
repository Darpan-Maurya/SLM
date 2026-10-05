"""`slm-generate` and the interactive chat loop."""
from __future__ import annotations

import argparse
import time

import torch

from slm.infer.generate import SamplingConfig, generate_stream, speculative_generate
from slm.infer.load import load_model
from slm.tokenizer import Tokenizer
from slm.train.distributed import pick_device


def _common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("checkpoint")
    p.add_argument("-t", "--tokenizer", default="data/tokenizer.json")
    p.add_argument("-n", "--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default="auto")
    return p


def main(argv: list[str] | None = None) -> int:
    p = _common(argparse.ArgumentParser(prog="slm-generate", description=__doc__))
    p.add_argument("--prompt", default="The key insight is")
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--speculative", action="store_true",
                   help="use the MTP heads for self-speculative decoding")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    tok = Tokenizer.load(args.tokenizer)
    ids = torch.tensor([tok.encode_ordinary(args.prompt)], device=device)
    cfg = SamplingConfig(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, seed=args.seed,
        greedy=args.greedy, stop_tokens=(tok.eos_id,),
    )

    print(args.prompt, end="", flush=True)
    t0 = time.perf_counter()
    if args.speculative:
        seq, stats = speculative_generate(model, ids, cfg)
        n = seq.shape[1] - ids.shape[1]
        print(tok.decode(seq[0, ids.shape[1]:].tolist()))
        print(f"[gen] acceptance {100*stats['acceptance_rate']:.1f}%", end=" | ")
    else:
        n = 0
        for tokens in generate_stream(model, ids, cfg):
            tid = int(tokens[0, 0])
            if tid == tok.eos_id:
                break
            n += 1
            print(tok.decode([tid]), end="", flush=True)
        print()
    dt = time.perf_counter() - t0
    print(f"[gen] {n} tokens in {dt:.2f}s = {n/max(dt, 1e-9):.1f} tok/s")
    return 0


def chat_main(argv: list[str] | None = None) -> int:
    from slm.sft import ChatTemplate, Message

    p = _common(argparse.ArgumentParser(prog="slm-chat", description=__doc__))
    p.add_argument("--system", default="")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    tok = Tokenizer.load(args.tokenizer)
    template = ChatTemplate(tok)
    history: list[Message] = [Message("system", args.system)] if args.system else []
    cfg = SamplingConfig(max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_p=args.top_p,
                         stop_tokens=(tok.eos_id,))

    print("chat ready - empty line or ctrl-c to quit, /reset to clear history\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            return 0
        if user == "/reset":
            history = history[:1] if args.system else []
            print("[history cleared]\n")
            continue

        history.append(Message("user", user))
        ids = torch.tensor(
            [tok.encode(template.render_prompt(history), allowed_special="all")],
            device=device,
        )
        print("slm> ", end="", flush=True)
        pieces: list[int] = []
        for t in generate_stream(model, ids, cfg):
            tid = int(t[0, 0])
            if tid == tok.eos_id:
                break
            pieces.append(tid)
            print(tok.decode([tid]), end="", flush=True)
        print("\n")
        history.append(Message("assistant", tok.decode(pieces)))
