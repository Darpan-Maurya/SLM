"""Model components."""
from slm.model.attention import GroupedQueryAttention, LatentAttention, build_attention
from slm.model.cache import Cache
from slm.model.masking import AttnMask, build_mask, doc_ids_from_eos
from slm.model.mlp import SwiGLU
from slm.model.moe import MoE, Router, build_ffn
from slm.model.norm import RMSNorm
from slm.model.rope import apply_rope, build_rope_cache
from slm.model.transformer import (
    Block,
    ModelOutput,
    MTPHead,
    Transformer,
    cross_entropy_loss,
)

__all__ = [
    "AttnMask",
    "Block",
    "Cache",
    "GroupedQueryAttention",
    "LatentAttention",
    "MTPHead",
    "MoE",
    "ModelOutput",
    "RMSNorm",
    "Router",
    "SwiGLU",
    "Transformer",
    "apply_rope",
    "build_attention",
    "build_ffn",
    "build_mask",
    "build_rope_cache",
    "cross_entropy_loss",
    "doc_ids_from_eos",
]
