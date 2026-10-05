"""Inference."""
from slm.infer.generate import (
    SamplingConfig,
    generate,
    generate_stream,
    speculative_generate,
)
from slm.infer.load import load_model

__all__ = [
    "SamplingConfig",
    "generate",
    "generate_stream",
    "load_model",
    "speculative_generate",
]
