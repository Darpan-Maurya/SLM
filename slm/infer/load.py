"""Load a model for inference from a checkpoint directory or an export file."""
from __future__ import annotations

import os

import torch

from slm.config import ModelConfig, from_dict
from slm.model import Transformer


def load_model(path: str, device=None, dtype=None) -> tuple[Transformer, dict]:
    """Returns (model in eval mode, the config dict it was trained with)."""
    if os.path.isdir(path):
        from slm.train.checkpoint import CheckpointManager

        ck = CheckpointManager.load(path, map_location="cpu")
        state, cfg_dict = ck["model"], ck["config"]
    else:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state, cfg_dict = blob["model"], blob["config"]
    model = Transformer(from_dict(ModelConfig, cfg_dict.get("model", cfg_dict)))
    model.load_state_dict(state, strict=True)
    if dtype is not None:
        model = model.to(dtype)
    return model.to(device or "cpu").eval(), cfg_dict
