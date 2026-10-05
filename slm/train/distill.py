"""Logit knowledge distillation from a same-tokenizer teacher."""
from __future__ import annotations

import os

import torch

from slm.config import ModelConfig
from slm.model import Transformer


def load_teacher(path: str, device, dtype=torch.bfloat16) -> Transformer:
    """Load a frozen teacher from a checkpoint directory or an inference export."""
    if os.path.isdir(path):
        import json

        with open(os.path.join(path, "config.json")) as f:
            cfg_dict = json.load(f)
        state = torch.load(os.path.join(path, "model.pt"), map_location="cpu",
                           weights_only=False)
        model_cfg = cfg_dict.get("model", cfg_dict)
    else:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state, model_cfg = blob["model"], blob["config"].get("model", blob["config"])

    from slm.config import from_dict

    teacher = Transformer(from_dict(ModelConfig, model_cfg))
    teacher.load_state_dict(state, strict=True)
    teacher.to(device=device, dtype=dtype if device.type == "cuda" else torch.float32)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    loss_mask: torch.Tensor | None = None,
    top_k: int = 0,
) -> torch.Tensor:
    """Temperature-scaled KL(teacher || student), averaged over live tokens."""
    t = max(temperature, 1e-3)
    s = student_logits.float() / t
    q = teacher_logits.float() / t
    if top_k and top_k < q.shape[-1]:
        # keep only the teacher's top-k and renormalise; the tail carries no signal
        vals, idx = q.topk(top_k, dim=-1)
        q_probs = vals.softmax(-1)
        s_logprobs = s.log_softmax(-1).gather(-1, idx)
        kl = (q_probs * (q_probs.clamp_min(1e-9).log() - s_logprobs)).sum(-1)
    else:
        q_probs = q.softmax(-1)
        kl = (q_probs * (q.log_softmax(-1) - s.log_softmax(-1))).sum(-1)

    if loss_mask is not None:
        m = loss_mask.float()
        kl = (kl * m).sum() / m.sum().clamp(min=1.0)
    else:
        kl = kl.mean()
    # T^2 keeps the gradient magnitude comparable across temperatures (Hinton 2015)
    return kl * (t ** 2)
