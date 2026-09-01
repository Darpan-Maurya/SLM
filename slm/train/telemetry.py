"""Throughput, MFU and metric logging."""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

# dense bf16/fp16 peak TFLOP/s, no sparsity
PEAK_TFLOPS = {
    "h200": 989.0, "h100": 989.0, "gh200": 989.0, "b200": 2250.0,
    "a100": 312.0, "a800": 312.0, "l40s": 362.0, "l40": 181.0, "a6000": 155.0,
    "4090": 165.2, "5090": 209.5, "3090": 71.0, "a10g": 125.0,
    "v100": 125.0, "t4": 65.0, "mi300x": 1307.0,
}


def device_peak_tflops(device: torch.device) -> float | None:
    """Peak dense bf16 throughput for the current accelerator, if we know it."""
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device).lower()
    for key, val in PEAK_TFLOPS.items():
        if key in name.replace(" ", ""):
            return val
    return None


@dataclass
class StepTimer:
    """Rolling window of step durations."""

    window: int = 50
    times: deque = field(default_factory=lambda: deque(maxlen=50))
    _t0: float = field(default_factory=time.perf_counter)

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._t0
        self._t0 = now
        self.times.append(dt)
        return dt

    @property
    def mean(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0


class Telemetry:
    """Console line, JSONL history, and optional Weights & Biases."""

    def __init__(
        self,
        run_dir: str,
        flops_per_token: float,
        device: torch.device,
        is_main: bool = True,
        wandb_project: str = "",
        wandb_entity: str = "",
        config: dict | None = None,
        run_name: str = "slm",
    ):
        self.is_main = is_main
        self.flops_per_token = flops_per_token
        self.peak = device_peak_tflops(device)
        self.timer = StepTimer()
        self.start = time.time()
        self.path = os.path.join(run_dir, "logs", "metrics.jsonl")
        self.wandb = None
        if is_main:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            if wandb_project:
                self.wandb = self._init_wandb(wandb_project, wandb_entity, config, run_name)

    @staticmethod
    def _init_wandb(project, entity, config, name):
        try:
            import wandb

            return wandb.init(project=project, entity=entity or None,
                              config=config, name=name, resume="allow")
        except Exception as exc:
            print(f"[telemetry] wandb disabled: {exc!r}")
            return None

    def mfu(self, tokens: int, seconds: float) -> float | None:
        """Fraction of the device's peak FLOPs actually used."""
        if not self.peak or seconds <= 0:
            return None
        achieved = tokens * self.flops_per_token / seconds
        return achieved / (self.peak * 1e12)

    def log(self, step: int, metrics: dict[str, Any], prefix: str = "train") -> dict:
        record = {"step": step, "time": time.time() - self.start,
                  **{f"{prefix}/{k}": _scalar(v) for k, v in metrics.items()}}
        if not self.is_main:
            return record
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.wandb is not None:
            self.wandb.log(record, step=step)
        return record

    def console(self, step: int, max_steps: int, metrics: dict[str, Any]) -> str:
        loss = _scalar(metrics.get("loss", float("nan")))
        parts = [
            f"step {step:>7}/{max_steps}",
            f"loss {loss:7.4f}",
            f"ppl {math.exp(min(loss, 20)):8.2f}",
            f"lr {_scalar(metrics.get('lr', 0)):.2e}",
        ]
        if "grad_norm" in metrics:
            parts.append(f"|g| {_scalar(metrics['grad_norm']):6.3f}")
        if "tokens_per_sec" in metrics:
            parts.append(f"{_scalar(metrics['tokens_per_sec'])/1e3:7.1f} ktok/s")
        if metrics.get("mfu"):
            parts.append(f"mfu {100*_scalar(metrics['mfu']):4.1f}%")
        if "step_time" in metrics:
            parts.append(f"{1000*_scalar(metrics['step_time']):6.0f} ms")
        if "eta_hours" in metrics:
            parts.append(f"eta {_scalar(metrics['eta_hours']):.1f}h")
        line = " | ".join(parts)
        if self.is_main:
            print(line, flush=True)
        return line

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


def _scalar(v: Any) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.detach().float().item())
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def format_params(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)
