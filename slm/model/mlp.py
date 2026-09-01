"""Feed-forward blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Gated feed-forward network with gate and up projections fused into one GEMM."""

    def __init__(self, d_model: int, hidden: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.w13 = nn.Linear(d_model, 2 * hidden, bias=bias)
        self.w2 = nn.Linear(hidden, d_model, bias=bias)
        self.dropout = dropout
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        if self.dropout and self.training:
            x = F.dropout(x, self.dropout)
        return self.w2(x)
