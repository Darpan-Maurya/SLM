"""Attention masks for causal, sliding-window, and intra-document attention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    HAS_FLEX = True
except ImportError:  # pragma: no cover
    HAS_FLEX = False
    flex_attention = create_block_mask = None  # type: ignore


@dataclass
class AttnMask:
    """Whichever mask representation the chosen attention backend consumes."""

    is_causal: bool = True          # plain causal: use the SDPA fast path
    dense: torch.Tensor | None = None    # additive float mask for SDPA
    block: Any | None = None        # flex_attention BlockMask
    window: int = 0

    @property
    def plain(self) -> bool:
        return self.dense is None and self.block is None


def _mask_mod(doc_ids: torch.Tensor | None, window: int, offset: int = 0):
    """Build a flex_attention predicate over (batch, head, q_pos, kv_pos)."""

    def mod(b, h, q, kv):
        q_abs = q + offset
        ok = q_abs >= kv
        if window > 0:
            ok = ok & (q_abs - kv < window)
        if doc_ids is not None:
            ok = ok & (doc_ids[b, q_abs] == doc_ids[b, kv])
        return ok

    return mod


def dense_mask(
    q_len: int,
    kv_len: int,
    device,
    dtype,
    doc_ids: torch.Tensor | None = None,
    window: int = 0,
    offset: int = 0,
) -> torch.Tensor:
    """Additive `-inf` mask of shape (1 or B, 1, q_len, kv_len)."""
    q_pos = torch.arange(q_len, device=device)[:, None] + offset
    kv_pos = torch.arange(kv_len, device=device)[None, :]
    ok = q_pos >= kv_pos
    if window > 0:
        ok = ok & (q_pos - kv_pos < window)
    ok = ok[None, None]
    if doc_ids is not None:
        same = doc_ids[:, offset : offset + q_len, None] == doc_ids[:, None, :kv_len]
        ok = ok & same[:, None]
    return torch.zeros_like(ok, dtype=dtype).masked_fill(~ok, float("-inf"))


def build_mask(
    q_len: int,
    kv_len: int,
    device,
    dtype,
    doc_ids: torch.Tensor | None = None,
    window: int = 0,
    offset: int = 0,
    impl: str = "auto",
    batch: int = 1,
) -> AttnMask:
    """Pick the cheapest correct mask for this layer's attention pattern."""
    if doc_ids is None and window <= 0:
        return AttnMask(is_causal=True)

    # flex has no MPS backward and no CPU fused kernel, so "auto" only picks it
    # on CUDA; an explicit impl="flex" is honoured wherever the caller asks
    dev = torch.device(device).type if not isinstance(device, torch.device) else device.type
    use_flex = HAS_FLEX and (impl == "flex" or (impl == "auto" and dev == "cuda"))
    if use_flex and q_len == kv_len:
        mod = _mask_mod(doc_ids, window, offset)
        block = create_block_mask(
            mod, batch if doc_ids is not None else None, None, q_len, kv_len,
            device=device,
        )
        return AttnMask(is_causal=False, block=block, window=window)

    return AttnMask(
        is_causal=False,
        dense=dense_mask(q_len, kv_len, device, dtype, doc_ids, window, offset),
        window=window,
    )


def doc_ids_from_eos(tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Segment a packed batch into documents by cumulative EOS count."""
    is_eos = (tokens == eos_id).to(torch.int32)
    # a document owns its own terminating EOS, so shift the count by one position
    return torch.cumsum(is_eos, dim=-1) - is_eos
