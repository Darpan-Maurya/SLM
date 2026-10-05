"""Measure step time, tokens/sec and MFU for a config on the current device."""
from __future__ import annotations

import time

import torch

from slm.config import Config
from slm.model import Transformer
from slm.train.optim import build_optimizer
from slm.train.telemetry import device_peak_tflops, format_params


def benchmark(
    cfg: Config,
    device: torch.device,
    steps: int = 20,
    warmup: int = 5,
    backward: bool = True,
) -> dict:
    """Time a synthetic training step; no data pipeline involved."""
    model = Transformer(cfg.model).to(device)
    model.set_grad_checkpoint(cfg.train.grad_checkpoint)
    if cfg.train.compile and device.type == "cuda":
        model = torch.compile(model)
    opt = build_optimizer(model, cfg.optim, device.type, verbose=False) if backward else None

    B, T = cfg.train.micro_batch_size, cfg.model.max_seq_len
    x = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)

    use_amp = device.type == "cuda" and cfg.train.dtype != "float32"
    amp_dtype = torch.bfloat16 if cfg.train.dtype == "bfloat16" else torch.float16

    def one_step():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(x, targets=y, return_logits=False, zloss=cfg.optim.zloss)
        if backward:
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        return out.loss

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = steps * B * T
    step_time = elapsed / steps
    tps = tokens / elapsed
    peak = device_peak_tflops(device)
    achieved = tps * cfg.model.flops_per_token()
    result = {
        "params": cfg.model.param_count()["total"],
        "active_params": cfg.model.param_count()["active"],
        "step_time_ms": step_time * 1000,
        "tokens_per_sec": tps,
        "tflops": achieved / 1e12,
        "mfu": (achieved / (peak * 1e12)) if peak else None,
        "peak_tflops": peak,
    }
    if device.type == "cuda":
        result["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return result


def format_result(name: str, r: dict) -> str:
    mfu = f"{100*r['mfu']:.1f}%" if r.get("mfu") else "n/a"
    mem = f" | {r['peak_memory_gb']:.1f} GB" if "peak_memory_gb" in r else ""
    return (f"{name:16s} {format_params(r['params']):>7} params "
            f"({format_params(r['active_params']):>7} active) | "
            f"{r['step_time_ms']:7.1f} ms/step | "
            f"{r['tokens_per_sec']/1e3:8.1f} ktok/s | "
            f"{r['tflops']:6.1f} TFLOP/s | MFU {mfu}{mem}")
