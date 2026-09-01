"""Process-group setup and model wrapping for single-GPU, DDP and FSDP."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    """Where this process sits in the job."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    backend: str = ""

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    def __repr__(self) -> str:
        return (f"DistInfo(rank={self.rank}/{self.world_size}, "
                f"device={self.device}, backend={self.backend or 'none'})")


def pick_device(prefer: str = "auto") -> torch.device:
    """Best available accelerator."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_distributed(prefer_device: str = "auto") -> DistInfo:
    """Join the process group if torchrun launched us, else run standalone."""
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return DistInfo(device=pick_device(prefer_device))

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = pick_device(prefer_device)
    return DistInfo(rank, local_rank, world_size, device, backend)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: torch.Tensor | float, device=None) -> float:
    """Average a scalar across ranks."""
    if not dist.is_initialized():
        return float(value)
    t = value.detach().clone() if isinstance(value, torch.Tensor) else torch.tensor(
        float(value), device=device
    )
    t = t.to(device or t.device, torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


def wrap_model(model: torch.nn.Module, strategy: str, info: DistInfo, cfg=None):
    """Apply DDP or FSDP; returns the model unchanged when running standalone."""
    if not info.enabled or strategy == "single":
        return model
    if strategy == "auto":
        strategy = "ddp"
    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        return DDP(
            model,
            device_ids=[info.local_rank] if info.device.type == "cuda" else None,
            # MoE routing sends different experts different work each step
            find_unused_parameters=bool(cfg and cfg.model.moe),
            gradient_as_bucket_view=True,
        )
    if strategy == "fsdp":
        return _wrap_fsdp(model, info, cfg)
    raise ValueError(f"unknown strategy {strategy!r}")


def _wrap_fsdp(model: torch.nn.Module, info: DistInfo, cfg=None):
    """Per-block sharding via FSDP2 when available, else FSDP1."""
    from slm.model.transformer import Block

    reshard = bool(cfg is None or cfg.train.fsdp_reshard_after_forward)
    try:
        from torch.distributed.fsdp import fully_shard  # FSDP2

        for block in model.blocks:
            fully_shard(block, reshard_after_forward=reshard)
        fully_shard(model, reshard_after_forward=reshard)
        return model
    except ImportError:
        pass

    from functools import partial

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    return FSDP(
        model,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
        device_id=info.local_rank if info.device.type == "cuda" else None,
        use_orig_params=True,
    )
