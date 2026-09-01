"""Decoder-only transformer; see docs/ARCHITECTURE.md for why each part is here."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from slm.config import ModelConfig
from slm.model.attention import build_attention
from slm.model.cache import Cache
from slm.model.masking import AttnMask, build_mask, dense_mask, doc_ids_from_eos
from slm.model.moe import MoE, build_ffn
from slm.model.norm import RMSNorm
from slm.model.rope import build_rope_cache


@dataclass
class ModelOutput:
    """What a forward pass returns."""

    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    hidden: torch.Tensor | None = None      # pre-final-norm stream, for MTP drafting
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


class Block(nn.Module):
    """Pre-norm transformer block; its FFN is dense or a routed MoE."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = build_attention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = build_ffn(cfg, layer_idx)
        self.is_moe = isinstance(self.ffn, MoE)
        self.resid_dropout = cfg.dropout

    def forward(self, x, rope=None, mask=None, cache=None, start_pos=0):
        h = self.attn(self.attn_norm(x), rope, mask, cache, start_pos)
        if self.resid_dropout and self.training:
            h = F.dropout(h, self.resid_dropout)
        x = x + h
        if self.is_moe:
            h, aux = self.ffn(self.ffn_norm(x))
        else:
            h, aux = self.ffn(self.ffn_norm(x)), None
        if self.resid_dropout and self.training:
            h = F.dropout(h, self.resid_dropout)
        return x + h, aux


class MTPHead(nn.Module):
    """One extra prediction depth: mixes the previous hidden state with a future token."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm_h = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_e = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.block = Block(cfg, layer_idx)

    def forward(self, h, emb, rope=None, mask=None):
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(emb)], dim=-1))
        return self.block(x, rope, mask)


class Transformer(nn.Module):
    """The model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.mtp = nn.ModuleList(
            MTPHead(cfg, cfg.n_layer + i) for i in range(cfg.mtp_depth)
        )

        rope_dim = cfg.qk_rope_head_dim if cfg.attn_type == "mla" else cfg.d_head
        cos, sin = build_rope_cache(
            cfg.max_seq_len, rope_dim, cfg.rope_theta, cfg.rope_scaling, cfg=cfg
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.grad_checkpoint = False
        # muP: readout is scaled down by the width multiplier at the forward pass
        self.output_mult = 1.0 / cfg.width_mult
        self.apply(self._init_weights)
        if cfg.scale_residual_init:
            self._scale_residual_branches()
        if cfg.use_mup:
            self._apply_mup()

    # --- initialisation ------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scale_residual_branches(self) -> None:
        """Shrink every projection that writes into the residual stream by 1/sqrt(2L)."""
        scale = self.cfg.init_std / math.sqrt(2 * self.cfg.n_layer)
        for block in list(self.blocks) + [h.block for h in self.mtp]:
            nn.init.normal_(block.attn.wo.weight, 0.0, scale)
            if isinstance(block.ffn, MoE):
                for e in block.ffn.experts:
                    nn.init.normal_(e.w2.weight, 0.0, scale)
                if block.ffn.shared is not None:
                    nn.init.normal_(block.ffn.shared.w2.weight, 0.0, scale)
            else:
                nn.init.normal_(block.ffn.w2.weight, 0.0, scale)

    def _apply_mup(self) -> None:
        """Hidden matrices get init std / sqrt(width_mult); the readout gets 1/width_mult.

        Combined with the 1/d_head attention scale and the per-group LR scaling in
        `param_groups`, this is the muP recipe: hyperparameters tuned on a narrow
        proxy transfer to the full width. Verified by `test_mup_coordinate_check`.
        """
        mult = self.cfg.width_mult
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.dim() != 2 or name.startswith("tok_emb"):
                    continue
                if name.startswith("lm_head"):
                    if not self.cfg.tie_embeddings:
                        p.mul_(1.0 / mult)
                else:
                    p.mul_(1.0 / math.sqrt(mult))

    def mup_lr_scale(self, name: str, param: torch.Tensor) -> float:
        """Adam LR multiplier for this parameter under muP (1/width for hidden)."""
        if not self.cfg.use_mup:
            return 1.0
        if param.dim() < 2 or name.startswith(("tok_emb", "lm_head")):
            return 1.0
        return 1.0 / self.cfg.width_mult

    # --- introspection -------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def set_grad_checkpoint(self, enabled: bool) -> None:
        self.grad_checkpoint = enabled

    def param_groups(self, weight_decay: float) -> list[dict]:
        """AdamW groups: decay matrices, leave gains and biases alone.

        Under muP the hidden matrices are split into their own group with an LR
        multiplier of 1/width_mult, which `set_lr` applies each step.
        """
        buckets: dict[tuple[float, bool], list] = {}
        seen = set()
        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            key = (self.mup_lr_scale(name, p), p.dim() >= 2)
            buckets.setdefault(key, []).append(p)
        groups = []
        for (scale, is_matrix), params in sorted(buckets.items()):
            groups.append({
                "params": params,
                "weight_decay": weight_decay if is_matrix else 0.0,
                "lr_scale": scale,
                "name": ("decay" if is_matrix else "no_decay")
                        + (f"_mup{scale:g}" if scale != 1.0 else ""),
            })
        return groups

    def muon_split(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """(hidden matrices for Muon, everything else for AdamW)."""
        muon, other, seen = [], [], set()
        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            excluded = name.startswith(("tok_emb", "lm_head")) or "router" in name
            (muon if p.dim() == 2 and not excluded else other).append(p)
        return muon, other

    def update_router_biases(self) -> None:
        """Apply loss-free load balancing; call once after each optimiser step."""
        for m in self.modules():
            if isinstance(m, MoE):
                m.router.update_bias()

    # --- forward -------------------------------------------------------------
    def _rope(self, start: int, length: int, dtype):
        return (
            self.rope_cos[start : start + length].to(dtype),
            self.rope_sin[start : start + length].to(dtype),
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        cache: Cache | None = None,
        start_pos: int = 0,
        loss_mask: torch.Tensor | None = None,
        return_logits: bool = True,
        return_hidden: bool = False,
        zloss: float | None = None,
    ) -> ModelOutput:
        cfg = self.cfg
        B, T = idx.shape
        assert start_pos + T <= cfg.max_seq_len, (
            f"position {start_pos + T} exceeds max_seq_len {cfg.max_seq_len}"
        )
        zloss = cfg_zloss if (cfg_zloss := zloss) is not None else 0.0

        emb = self.tok_emb(idx)
        if cfg.embedding_scale:
            emb = emb * math.sqrt(cfg.d_model)
        x = emb
        rope = self._rope(start_pos, T, x.dtype)
        kv_len = start_pos + T

        masks = self._build_masks(B, T, kv_len, start_pos, doc_ids, x)
        metrics: dict[str, torch.Tensor] = {}
        aux_total = x.new_zeros(())

        for i, block in enumerate(self.blocks):
            mask = masks[cfg.is_global_layer(i)]
            if self.grad_checkpoint and self.training:
                x, aux = _ckpt(block, x, rope, mask, cache, start_pos,
                               use_reentrant=False)
            else:
                x, aux = block(x, rope, mask, cache, start_pos)
            if aux:
                aux_total = aux_total + self._aux_loss(aux, metrics, i)
        h = self.norm(x)

        if targets is None:
            logits = self.lm_head(h if return_logits else h[:, -1:]) * self.output_mult
            return ModelOutput(logits=logits, metrics=metrics,
                               hidden=x if return_hidden else None)

        logits = self.lm_head(h) * self.output_mult
        loss, m = cross_entropy_loss(
            logits, targets, loss_mask=loss_mask, zloss=zloss,
            softcap=cfg.final_logit_softcap,
        )
        metrics.update(m)
        total = loss + aux_total

        if self.mtp and self.training:
            mtp_loss = self._mtp_loss(x, idx, targets, rope, masks, loss_mask, metrics)
            total = total + cfg.mtp_weight * mtp_loss
        metrics["loss"] = total.detach()
        return ModelOutput(logits=logits if return_logits else None,
                           loss=total, metrics=metrics,
                           hidden=x if return_hidden else None)

    def _build_masks(self, B, T, kv_len, start_pos, doc_ids, x) -> dict[bool, AttnMask]:
        """One mask for global layers, one for windowed layers."""
        cfg = self.cfg
        ids = doc_ids if (cfg.doc_masking and doc_ids is not None) else None
        out = {}
        for is_global in (True, False):
            window = 0 if is_global else cfg.sliding_window
            if ids is None and window <= 0:
                # SDPA's is_causal aligns top-left, which is wrong once a cache
                # holds earlier tokens; a single query needs no mask at all
                if start_pos == 0:
                    out[is_global] = AttnMask(is_causal=True)
                elif T == 1:
                    out[is_global] = AttnMask(is_causal=False)
                else:
                    out[is_global] = AttnMask(
                        is_causal=False,
                        dense=dense_mask(T, kv_len, x.device, x.dtype,
                                         offset=start_pos),
                    )
            else:
                out[is_global] = build_mask(
                    T, kv_len, x.device, x.dtype, ids, window, start_pos,
                    impl=cfg.attn_impl, batch=B,
                )
            if cfg.sliding_window <= 0:
                out[False] = out[True]
                break
        return out

    def _aux_loss(self, aux: dict, metrics: dict, layer: int) -> torch.Tensor:
        """Accumulate router regularisers and record balance diagnostics."""
        total = torch.zeros((), device=aux["expert_counts"].device)
        if "router_zloss" in aux:
            total = total + self.cfg.router_zloss * aux["router_zloss"]
        if "router_aux" in aux:
            total = total + self.cfg.router_aux_loss * aux["router_aux"]
        metrics[f"load_cv/layer{layer}"] = aux["load_cv"]
        prev = metrics.get("load_cv")
        metrics["load_cv"] = aux["load_cv"] if prev is None else (prev + aux["load_cv"]) / 2
        return total

    def _mtp_loss(self, h, idx, targets, rope, masks, loss_mask, metrics):
        """Predict tokens t+2..t+1+depth from lightweight extra blocks."""
        total = h.new_zeros(())
        T = idx.shape[1]
        prev = h
        for k, head in enumerate(self.mtp, start=1):
            span = T - k
            if span <= 1:
                break
            emb_future = self.tok_emb(idx[:, k:])
            sub_rope = (rope[0][:span], rope[1][:span])
            hk, _ = head(prev[:, :span], emb_future, sub_rope, masks[True])
            logits_k = self.lm_head(self.norm(hk)) * self.output_mult
            mask_k = loss_mask[:, :span] if loss_mask is not None else None
            lk, _ = cross_entropy_loss(logits_k, targets[:, k:], loss_mask=mask_k)
            metrics[f"mtp{k}_loss"] = lk.detach()
            total = total + lk
            prev = hk
        return total / max(len(self.mtp), 1)

    # --- inference helpers ---------------------------------------------------
    @torch.no_grad()
    def resize_context(self, new_len: int, scaling: float | None = None,
                       scaling_type: str | None = None) -> None:
        """Rebuild rotary tables for a longer context (linear / NTK / YaRN)."""
        cfg = self.cfg
        cfg.rope_original_len = cfg.rope_original_len or cfg.max_seq_len
        cfg.max_seq_len = new_len
        if scaling is not None:
            cfg.rope_scaling = scaling
        if scaling_type is not None:
            cfg.rope_scaling_type = scaling_type
        rope_dim = cfg.qk_rope_head_dim if cfg.attn_type == "mla" else cfg.d_head
        cos, sin = build_rope_cache(
            new_len, rope_dim, cfg.rope_theta, cfg.rope_scaling,
            device=self.rope_cos.device, cfg=cfg,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def alloc_cache(self, batch: int, max_len: int | None = None) -> Cache:
        """Empty decoding cache sized for this model."""
        return Cache(self.cfg.n_layer, max_len or self.cfg.max_seq_len)

    def doc_ids(self, idx: torch.Tensor, eos_id: int) -> torch.Tensor:
        """Document segmentation for intra-document masking."""
        return doc_ids_from_eos(idx, eos_id)

    @torch.no_grad()
    def mtp_step(
        self, hidden: torch.Tensor, token: torch.Tensor, depth: int, position: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run MTP head `depth` on one position; returns (logits, next hidden)."""
        head = self.mtp[depth]
        emb = self.tok_emb(token)
        rope = self._rope(position, hidden.shape[1], hidden.dtype)
        hk, _ = head(hidden, emb, rope, AttnMask(is_causal=True))
        return self.lm_head(self.norm(hk)) * self.output_mult, hk


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    zloss: float = 0.0,
    ignore_index: int = -100,
    softcap: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Token-mean cross entropy in fp32, plus the optional logit z-loss."""
    logits = logits.float()
    if softcap > 0.0:
        logits = torch.tanh(logits / softcap) * softcap
    V = logits.shape[-1]
    flat_logits = logits.view(-1, V)
    flat_targets = targets.reshape(-1)

    losses = F.cross_entropy(
        flat_logits, flat_targets, ignore_index=ignore_index, reduction="none"
    )
    valid = (flat_targets != ignore_index).float()
    if loss_mask is not None:
        valid = valid * loss_mask.reshape(-1).float()
    n_valid = valid.sum().clamp(min=1.0)
    ce = (losses * valid).sum() / n_valid

    metrics = {"ce": ce.detach(), "n_tokens": n_valid.detach()}
    loss = ce
    if zloss > 0.0:
        z = ((torch.logsumexp(flat_logits, dim=-1) ** 2) * valid).sum() / n_valid
        loss = loss + zloss * z
        metrics["zloss"] = z.detach()
    return loss, metrics
