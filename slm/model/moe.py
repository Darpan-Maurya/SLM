"""Sparse mixture-of-experts with auxiliary-loss-free load balancing (DeepSeek-V3)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.config import ModelConfig
from slm.model.mlp import SwiGLU


class Router(nn.Module):
    """Top-k gate whose per-expert bias steers load without touching the gradient."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_experts_active
        self.norm_topk = cfg.norm_topk_prob
        self.aux_weight = cfg.router_aux_loss
        self.zloss_weight = cfg.router_zloss
        self.bias_update = cfg.router_bias_update
        self.weight = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_model))
        nn.init.normal_(self.weight, std=cfg.init_std)
        # bias is a buffer, not a parameter: it is nudged by a rule, never by SGD
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts))
        self.register_buffer("load", torch.zeros(cfg.n_experts), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        logits = F.linear(x.float(), self.weight.float())
        scores = logits.sigmoid()
        _, idx = torch.topk(scores + self.expert_bias, self.top_k, dim=-1)
        weights = scores.gather(-1, idx)
        if self.norm_topk:
            weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-9)

        aux: dict[str, torch.Tensor] = {}
        counts = torch.bincount(idx.flatten(), minlength=self.n_experts).float()
        if self.training:
            self.load += counts
        aux["expert_counts"] = counts.detach()
        # perfectly balanced routing sends N*k/E tokens to each expert
        ideal = idx.numel() / self.n_experts
        aux["load_cv"] = (counts.std() / max(ideal, 1e-9)).detach()

        if self.zloss_weight > 0:
            aux["router_zloss"] = logits.logsumexp(-1).pow(2).mean()
        if self.aux_weight > 0:
            frac = counts / counts.sum().clamp(min=1)
            prob = scores.mean(0)
            aux["router_aux"] = self.n_experts * (frac * prob).sum()
        return idx, weights.to(x.dtype), aux

    @torch.no_grad()
    def update_bias(self) -> None:
        """Nudge under-used experts up and over-used ones down, then reset the tally."""
        if self.bias_update <= 0 or self.load.sum() == 0:
            return
        err = self.load.mean() - self.load
        self.expert_bias += self.bias_update * torch.sign(err)
        self.load.zero_()


class MoE(nn.Module):
    """Routed experts plus always-on shared experts."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_experts_active
        self.router = Router(cfg)
        self.experts = nn.ModuleList(
            SwiGLU(cfg.d_model, cfg.expert_hidden, cfg.bias, cfg.dropout)
            for _ in range(cfg.n_experts)
        )
        self.shared = (
            SwiGLU(cfg.d_model, cfg.expert_hidden * cfg.n_shared_experts,
                   cfg.bias, cfg.dropout)
            if cfg.n_shared_experts > 0 else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        idx, weights, aux = self.router(flat)

        out = torch.zeros_like(flat)
        # group token-expert pairs by expert so each expert sees one contiguous batch
        flat_expert = idx.reshape(-1)
        order = torch.argsort(flat_expert)
        sorted_expert = flat_expert[order]
        token_of = order // self.top_k
        weight_of = weights.reshape(-1)[order]
        boundaries = torch.searchsorted(
            sorted_expert, torch.arange(self.n_experts + 1, device=x.device)
        )
        starts = boundaries[:-1].tolist()
        ends = boundaries[1:].tolist()

        for e in range(self.n_experts):
            lo, hi = starts[e], ends[e]
            if hi <= lo:
                continue
            sel = token_of[lo:hi]
            y = self.experts[e](flat[sel])
            out.index_add_(0, sel, y * weight_of[lo:hi, None].to(y.dtype))

        out = out.view(B, T, d)
        if self.shared is not None:
            out = out + self.shared(x)
        return out, aux


def build_ffn(cfg: ModelConfig, layer_idx: int) -> nn.Module:
    """Dense SwiGLU or a routed MoE block, per the layer schedule."""
    if cfg.is_moe_layer(layer_idx):
        return MoE(cfg)
    return SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.bias, cfg.dropout)
