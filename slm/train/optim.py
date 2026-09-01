"""Optimiser construction and learning-rate schedules."""
from __future__ import annotations

import inspect
import math

import torch

from slm.config import OptimConfig


# --- schedules -----------------------------------------------------------------
def schedule_lr(
    step: int,
    *,
    base_lr: float,
    max_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    schedule: str = "wsd",
    decay_steps: int = 0,
    decay_shape: str = "1-sqrt",
) -> float:
    """Learning rate at ``step`` (0-indexed). Pure function, no state."""
    min_lr = base_lr * min_lr_ratio

    if warmup_steps > 0 and step < warmup_steps:
        # +1 so step 0 is not a dead step at lr=0
        return base_lr * (step + 1) / warmup_steps

    if schedule == "constant":
        return base_lr

    progress_step = step - warmup_steps
    total = max(1, max_steps - warmup_steps)

    if schedule == "cosine":
        t = min(1.0, progress_step / total)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))

    if schedule == "linear":
        t = min(1.0, progress_step / total)
        return base_lr + (min_lr - base_lr) * t

    if schedule == "wsd":
        d = decay_steps if decay_steps > 0 else max(1, int(0.2 * max_steps))
        decay_start = max_steps - d
        if step < decay_start:
            return base_lr
        t = min(1.0, (step - decay_start) / d)
        if decay_shape == "linear":
            frac = 1.0 - t
        elif decay_shape == "cosine":
            frac = 0.5 * (1 + math.cos(math.pi * t))
        else:                       # "1-sqrt": empirically the strongest of the three
            frac = 1.0 - math.sqrt(t)
        return min_lr + (base_lr - min_lr) * frac

    raise ValueError(f"unknown schedule {schedule!r}")


def lr_at(step: int, cfg: OptimConfig, max_steps: int) -> float:
    return schedule_lr(
        step,
        base_lr=cfg.lr, max_steps=max_steps, warmup_steps=cfg.warmup_steps,
        min_lr_ratio=cfg.min_lr_ratio, schedule=cfg.schedule,
        decay_steps=cfg.decay_steps, decay_shape=cfg.decay_shape,
    )


def set_lr(optimizer: torch.optim.Optimizer, lr: float, muon_lr: float | None = None) -> float:
    """Set the LR on every group, honouring per-group scales and Muon's own base."""
    for group in optimizer.param_groups:
        base = muon_lr if (muon_lr is not None and group.get("owner") == "muon") else lr
        group["lr"] = base * group.get("lr_scale", 1.0)
    return lr


# --- optimiser -----------------------------------------------------------------
def build_optimizer(
    model: torch.nn.Module, cfg: OptimConfig, device_type: str = "cuda", verbose: bool = True
) -> torch.optim.Optimizer:
    """AdamW, or hybrid Muon+AdamW when cfg.kind == 'muon'."""
    if cfg.kind == "muon":
        return _build_muon(model, cfg, device_type, verbose)
    if cfg.kind != "adamw":
        raise ValueError(f"unknown optim kind {cfg.kind!r}")
    return _build_adamw(model, cfg, device_type, verbose)


def _build_muon(model, cfg: OptimConfig, device_type: str, verbose: bool):
    """Muon on hidden matrices, AdamW on embeddings, norms, head and routers."""
    from slm.train.muon import CombinedOptimizer, Muon

    muon_params, other = model.muon_split()
    adam_groups = [
        {"params": [p for p in other if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in other if p.dim() < 2], "weight_decay": 0.0},
    ]
    opts = {
        "muon": Muon(
            muon_params, lr=cfg.muon_lr, momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps, weight_decay=cfg.weight_decay,
            rms_scale=cfg.muon_rms_scale,
        ),
        "adamw": torch.optim.AdamW(
            adam_groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps,
            **({"fused": True} if cfg.fused and device_type == "cuda" else {}),
        ),
    }
    if verbose:
        n_muon = sum(p.numel() for p in muon_params)
        n_adam = sum(p.numel() for p in other)
        print(
            f"[optim] Muon {n_muon:,} params in {len(muon_params)} matrices | "
            f"AdamW {n_adam:,} params in {len(other)} tensors"
        )
    return CombinedOptimizer(opts)


def _build_adamw(model, cfg: OptimConfig, device_type: str, verbose: bool):
    """AdamW with decay on matrices only, fused when the backend supports it."""
    groups = model.param_groups(cfg.weight_decay)
    n_decay = sum(p.numel() for p in groups[0]["params"])
    n_plain = sum(p.numel() for p in groups[1]["params"])

    kwargs: dict = dict(lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps)
    supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = cfg.fused and supports_fused and device_type == "cuda"
    if use_fused:
        kwargs["fused"] = True

    opt = torch.optim.AdamW(groups, **kwargs)
    if verbose:
        print(
            f"[optim] AdamW fused={use_fused} | decayed {n_decay:,} params "
            f"in {len(groups[0]['params'])} tensors | "
            f"undecayed {n_plain:,} in {len(groups[1]['params'])}"
        )
    return opt


def clip_grad_norm(
    model: torch.nn.Module, max_norm: float, foreach: bool | None = None
) -> torch.Tensor:
    """Global-norm clip; foreach=None lets torch pick a backend the device supports."""
    if max_norm <= 0:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, foreach=foreach)


def scale_lr_for_batch(base_lr: float, base_tokens: int, actual_tokens: int) -> float:
    """Square-root batch scaling."""
    return base_lr * math.sqrt(actual_tokens / base_tokens)


# --- diagnostics ---------------------------------------------------------------
@torch.no_grad()
def grad_stats(model: torch.nn.Module) -> dict[str, float]:
    """Cheap health signals: a run usually dies in the grad norms first."""
    total_sq, max_abs, n = 0.0, 0.0, 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total_sq += float(g.pow(2).sum())
        max_abs = max(max_abs, float(g.abs().max()))
        n += g.numel()
    return {
        "grad_norm": math.sqrt(total_sq),
        "grad_max": max_abs,
        "grad_rms": math.sqrt(total_sq / max(n, 1)),
    }


@torch.no_grad()
def param_stats(model: torch.nn.Module) -> dict[str, float]:
    total_sq, n = 0.0, 0
    for p in model.parameters():
        total_sq += float(p.detach().float().pow(2).sum())
        n += p.numel()
    return {"param_norm": math.sqrt(total_sq), "param_rms": math.sqrt(total_sq / max(n, 1))}
