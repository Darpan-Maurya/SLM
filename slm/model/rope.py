"""Rotary position embeddings, with linear / NTK / YaRN context extension."""
from __future__ import annotations

import math

import torch

from slm.config import ModelConfig


def _yarn_correction_dim(rotations: float, dim: int, base: float, orig_len: int) -> float:
    """Head dimension whose wavelength completes `rotations` turns over the context."""
    return (dim * math.log(orig_len / (rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_ramp(low: float, high: float, dim: int, device) -> torch.Tensor:
    """Smooth 0->1 ramp between two head-dimension indices."""
    if high - low < 1e-3:
        high = low + 1e-3
    x = (torch.arange(dim, dtype=torch.float64, device=device) - low) / (high - low)
    return x.clamp(0.0, 1.0)


def build_inv_freq(cfg: ModelConfig, head_dim: int, device=None) -> tuple[torch.Tensor, float]:
    """Inverse frequencies plus the attention-temperature multiplier."""
    base, scaling = cfg.rope_theta, max(1.0, cfg.rope_scaling)
    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float64)
    kind = cfg.rope_scaling_type

    if kind == "none" or scaling == 1.0:
        return 1.0 / (base ** (idx / head_dim)), 1.0

    if kind == "linear":
        return 1.0 / (base ** (idx / head_dim)), 1.0  # positions divided in build_rope

    if kind == "ntk":
        # stretch the base instead of the positions: keeps high frequencies intact
        adjusted = base * (scaling ** (head_dim / (head_dim - 2)))
        return 1.0 / (adjusted ** (idx / head_dim)), 1.0

    if kind == "yarn":
        orig = cfg.rope_original_len or int(cfg.max_seq_len / scaling)
        extrapolation = 1.0 / (base ** (idx / head_dim))
        interpolation = extrapolation / scaling
        low = math.floor(_yarn_correction_dim(cfg.yarn_beta_fast, head_dim, base, orig))
        high = math.ceil(_yarn_correction_dim(cfg.yarn_beta_slow, head_dim, base, orig))
        low, high = max(low, 0), min(high, head_dim // 2 - 1)
        # high-frequency dims extrapolate, low-frequency dims interpolate
        mask = 1.0 - _yarn_ramp(low, high, head_dim // 2, device)
        inv_freq = interpolation * (1 - mask) + extrapolation * mask
        mscale = 0.1 * math.log(scaling) + 1.0
        return inv_freq, mscale

    raise ValueError(f"unknown rope_scaling_type {kind!r}")


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10_000.0,
    scaling: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    cfg: ModelConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)` of shape `(seq_len, head_dim)`, phases accumulated in fp64."""
    assert head_dim % 2 == 0, "rotary dimension must be even"
    if cfg is not None:
        # the config owns the scaling factor; the argument is only for cfg-less callers
        scaling = cfg.rope_scaling
        inv_freq, mscale = build_inv_freq(cfg, head_dim, device)
        divisor = scaling if cfg.rope_scaling_type == "linear" else 1.0
    else:
        idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float64)
        inv_freq, mscale, divisor = 1.0 / (theta ** (idx / head_dim)), 1.0, scaling
    t = torch.arange(seq_len, device=device, dtype=torch.float64) / max(divisor, 1e-9)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * mscale).to(dtype), (emb.sin() * mscale).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Pair dimension i with i+d/2 and rotate."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate `(B, H, T, hd)` by the position phases in `cos`/`sin` of `(T, hd)`."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)
