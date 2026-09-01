"""Incremental-decoding cache, shaped by whatever the attention layer stores."""
from __future__ import annotations

import torch


class Cache:
    """Per-layer ring of tensors whose time axis is dim -2, allocated on first use."""

    def __init__(self, n_layer: int, max_len: int):
        self.n_layer = n_layer
        self.max_len = max_len
        self.slots: list[tuple[torch.Tensor, ...] | None] = [None] * n_layer
        self.length = 0

    def update(
        self, layer: int, tensors: tuple[torch.Tensor, ...], start: int
    ) -> tuple[torch.Tensor, ...]:
        """Write `tensors` at position `start`, return everything cached so far."""
        buf = self.slots[layer]
        if buf is None:
            buf = tuple(
                torch.zeros(
                    (*t.shape[:-2], self.max_len, t.shape[-1]),
                    device=t.device, dtype=t.dtype,
                )
                for t in tensors
            )
            self.slots[layer] = buf
        end = start + tensors[0].shape[-2]
        out = []
        for dst, src in zip(buf, tensors):
            dst[..., start:end, :] = src
            out.append(dst[..., :end, :])
        if layer == self.n_layer - 1:
            self.length = end
        return tuple(out)

    def reorder(self, index: torch.Tensor) -> None:
        """Permute the batch dimension (beam search, rejection sampling)."""
        self.slots = [
            None if s is None else tuple(t.index_select(0, index) for t in s)
            for s in self.slots
        ]

    def reset(self) -> None:
        self.slots = [None] * self.n_layer
        self.length = 0

    def bytes(self) -> int:
        """Total cache footprint - the real limit on serving batch size."""
        return sum(
            t.numel() * t.element_size()
            for s in self.slots if s is not None for t in s
        )
