"""Attention: grouped-query (default) and multi-head latent (DeepSeek-V2 MLA)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.config import ModelConfig
from slm.model.cache import Cache
from slm.model.masking import HAS_FLEX, AttnMask, flex_attention
from slm.model.norm import RMSNorm
from slm.model.rope import apply_rope


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (B, n_kv, T, hd) to (B, n_kv*n_rep, T, hd), groups contiguous."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def dispatch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: AttnMask | None,
    dropout: float = 0.0,
    scale: float | None = None,
) -> torch.Tensor:
    """Run SDPA or flex_attention depending on which mask form was built."""
    if mask is None or mask.plain:
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout,
            is_causal=(mask is None or mask.is_causal), scale=scale,
        )
    if mask.block is not None and HAS_FLEX:
        return flex_attention(q, k, v, block_mask=mask.block, scale=scale)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask.dense.to(q.dtype), dropout_p=dropout, scale=scale,
    )


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with n_head queries sharing n_kv_head key/value heads."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head, self.n_kv_head = cfg.n_head, cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.d_head = cfg.d_head
        self.use_rope = not cfg.is_nope_layer(layer_idx)

        q_dim, kv_dim = cfg.n_head * cfg.d_head, cfg.n_kv_head * cfg.d_head
        self.wqkv = nn.Linear(cfg.d_model, q_dim + 2 * kv_dim, bias=cfg.bias)
        self.wo = nn.Linear(q_dim, cfg.d_model, bias=cfg.bias)
        self.q_norm = RMSNorm(cfg.d_head, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.d_head, cfg.norm_eps) if cfg.qk_norm else None
        self.split = (q_dim, kv_dim, kv_dim)
        self.attn_dropout = cfg.attn_dropout
        # muP replaces 1/sqrt(d_head) with 1/d_head so attention logits stay
        # width-independent as the model grows
        self.scale = (1.0 / cfg.d_head) if cfg.use_mup else None

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        mask: AttnMask | None = None,
        cache: Cache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.wqkv(x).split(self.split, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)

        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope and rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)

        if cache is not None:
            k, v = cache.update(self.layer_idx, (k, v), start_pos)

        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        y = dispatch_attention(
            q, k, v, mask, self.attn_dropout if self.training else 0.0, scale=self.scale
        )
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class LatentAttention(nn.Module):
    """MLA: keys and values are cached as one low-rank latent plus a shared RoPE key."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = cfg.n_head
        self.qk_nope, self.qk_rope = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim
        self.v_dim, self.kv_lora = cfg.v_head_dim, cfg.kv_lora_rank
        self.use_rope = not cfg.is_nope_layer(layer_idx)
        qk = self.qk_nope + self.qk_rope
        self.qk = qk
        self.scale = qk ** -0.5

        self.q_lora = cfg.q_lora_rank
        if self.q_lora:
            self.wq_a = nn.Linear(cfg.d_model, self.q_lora, bias=False)
            self.q_norm = RMSNorm(self.q_lora, cfg.norm_eps)
            self.wq_b = nn.Linear(self.q_lora, cfg.n_head * qk, bias=False)
        else:
            self.wq = nn.Linear(cfg.d_model, cfg.n_head * qk, bias=False)

        self.wkv_a = nn.Linear(cfg.d_model, self.kv_lora + self.qk_rope, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora, cfg.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora, cfg.n_head * (self.qk_nope + self.v_dim), bias=False
        )
        self.wo = nn.Linear(cfg.n_head * self.v_dim, cfg.d_model, bias=False)
        self.attn_dropout = cfg.attn_dropout

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        mask: AttnMask | None = None,
        cache: Cache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq_b(self.q_norm(self.wq_a(x))) if self.q_lora else self.wq(x)
        q = q.view(B, T, self.n_head, self.qk).transpose(1, 2)
        q_nope, q_rope = q.split([self.qk_nope, self.qk_rope], dim=-1)

        kv = self.wkv_a(x)
        c_kv, k_rope = kv.split([self.kv_lora, self.qk_rope], dim=-1)
        c_kv = c_kv.unsqueeze(1)                      # (B, 1, T, kv_lora)
        k_rope = k_rope.unsqueeze(1)                  # (B, 1, T, qk_rope), head-shared

        if self.use_rope and rope is not None:
            q_rope = apply_rope(q_rope, *rope)
            k_rope = apply_rope(k_rope, *rope)

        if cache is not None:
            c_kv, k_rope = cache.update(self.layer_idx, (c_kv, k_rope), start_pos)

        kv_up = self.wkv_b(self.kv_norm(c_kv.squeeze(1)))
        kv_up = kv_up.view(B, -1, self.n_head, self.qk_nope + self.v_dim).transpose(1, 2)
        k_nope, v = kv_up.split([self.qk_nope, self.v_dim], dim=-1)

        k = torch.cat([k_nope, k_rope.expand(-1, self.n_head, -1, -1)], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)
        y = dispatch_attention(
            q, k, v, mask, self.attn_dropout if self.training else 0.0, scale=self.scale
        )
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


def build_attention(cfg: ModelConfig, layer_idx: int) -> nn.Module:
    """Instantiate the attention variant named by the config."""
    if cfg.attn_type == "mla":
        return LatentAttention(cfg, layer_idx)
    if cfg.attn_type == "gqa":
        return GroupedQueryAttention(cfg, layer_idx)
    raise ValueError(f"unknown attn_type {cfg.attn_type!r}")
