"""Muon: orthogonalised-momentum optimiser for 2D hidden weights (see docs/RESEARCH.md R4)."""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

# quintic Newton-Schulz coefficients tuned for fast convergence of the singular values
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def ns_dtype(device: torch.device) -> torch.dtype:
    """bf16 where tensor cores make it fast, fp32 elsewhere.

    CPU and MPS have no fast bf16 matmul path, and the iteration is five matmuls
    per parameter per step - running it in bf16 there is orders of magnitude
    slower than fp32, not faster.
    """
    return torch.bfloat16 if device.type == "cuda" else torch.float32


@torch.no_grad()
def newton_schulz(
    G: torch.Tensor, steps: int = 5, eps: float = 1e-7, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Approximate the orthogonal factor of G by a quintic Newton-Schulz iteration."""
    assert G.ndim >= 2
    X = G.to(dtype or ns_dtype(G.device))
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = _NS_B * A + _NS_C * (A @ A)
        X = _NS_A * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """SGD-momentum whose update is orthogonalised before it is applied."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.1,
        rms_scale: float = 0.2,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        super().__init__(
            list(params),
            dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                 weight_decay=weight_decay, rms_scale=rms_scale, ns_dtype=None),
        )
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon takes 2D parameters only; got shape {tuple(p.shape)}. "
                        "Route embeddings, norms and the LM head to AdamW."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            wd, scale = group["weight_decay"], group["rms_scale"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1.0 - mom)
                g = p.grad.lerp(buf, mom) if group["nesterov"] else buf
                update = newton_schulz(g, group["ns_steps"], dtype=group.get("ns_dtype"))
                # match AdamW's update RMS so a shared learning rate transfers
                adjusted = scale * math.sqrt(max(p.size(0), p.size(1)))
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr * adjusted)
        return loss


class CombinedOptimizer(torch.optim.Optimizer):
    """Presents several optimisers as one, so the training loop stays unaware."""

    def __init__(self, optimizers: dict[str, torch.optim.Optimizer]):
        self.optimizers = optimizers
        self._names = list(optimizers)
        # deliberately not calling super().__init__: state lives in the children
        self.defaults: dict[str, Any] = {}

    @property
    def param_groups(self) -> list[dict]:
        groups = []
        for name in self._names:
            for g in self.optimizers[name].param_groups:
                g.setdefault("owner", name)
                groups.append(g)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for opt in self.optimizers.values():
            opt.step()
        return closure() if closure is not None else None

    def state_dict(self) -> dict:
        return {n: self.optimizers[n].state_dict() for n in self._names}

    def load_state_dict(self, sd: dict) -> None:
        for n, s in sd.items():
            if n in self.optimizers:
                self.optimizers[n].load_state_dict(s)

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{n}({sum(len(g['params']) for g in o.param_groups)} tensors)"
            for n, o in self.optimizers.items()
        )
        return f"CombinedOptimizer({parts})"
