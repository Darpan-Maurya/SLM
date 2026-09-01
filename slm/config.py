"""Typed, composable configuration."""
from __future__ import annotations

import ast
import dataclasses
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


# --- Model ---------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Architecture. Defaults describe the 300M flagship."""

    vocab_size: int = 32768
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 4          # GQA: n_head % n_kv_head == 0; == n_head is MHA
    d_model: int = 1024
    d_head: int | None = None   # default d_model // n_head
    ffn_hidden: int | None = None   # default: swiglu-adjusted 8/3*d, rounded
    ffn_multiple_of: int = 128
    max_seq_len: int = 2048

    # positional
    rope_theta: float = 10_000.0
    rope_scaling: float = 1.0   # >1 extends context by frequency interpolation

    # normalisation / stability
    norm_eps: float = 1e-5
    qk_norm: bool = True        # RMSNorm on q,k before attention (OLMo2/Chameleon)
    final_logit_softcap: float = 0.0

    # parameterisation
    tie_embeddings: bool = True
    init_std: float = 0.02
    scale_residual_init: bool = True   # 1/sqrt(2*n_layer) on out-projections
    embedding_scale: bool = False      # multiply embeddings by sqrt(d_model)

    # regularisation
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # runtime
    use_flash: bool = True      # torch SDPA fused kernels
    bias: bool = False

    # --- attention variant ---
    attn_type: str = "gqa"          # gqa | mla
    attn_impl: str = "auto"         # auto | sdpa | flex | eager
    nope_every: int = 4             # every Nth layer drops RoPE (0 disables)
    doc_masking: bool = True        # never attend across a document boundary
    sliding_window: int = 0         # 0 = full causal attention
    global_attn_every: int = 4      # with a window set, every Nth layer stays global

    # --- MLA (multi-head latent attention, DeepSeek-V2) ---
    kv_lora_rank: int = 0           # 0 -> d_model // 4
    q_lora_rank: int = 0            # 0 disables query compression
    qk_rope_head_dim: int = 0       # 0 -> d_head // 2
    qk_nope_head_dim: int = 0       # 0 -> d_head
    v_head_dim: int = 0             # 0 -> d_head

    # --- rope scaling ---
    rope_scaling_type: str = "none"     # none | linear | ntk | yarn
    rope_original_len: int = 0          # context the model was pretrained at
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    # --- mixture of experts ---
    moe: bool = False
    n_experts: int = 0              # total routed experts
    n_experts_active: int = 0       # experts evaluated per token (top-k)
    n_shared_experts: int = 1       # always-on experts
    expert_hidden: int | None = None    # per-expert width; None -> ffn_hidden // k
    moe_layer_freq: int = 1         # every Nth eligible layer is MoE
    moe_first_dense: int = 1        # leading layers kept dense
    router_bias_update: float = 1e-3    # loss-free load balancing rate (DeepSeek-V3)
    router_aux_loss: float = 0.0        # classic aux loss; 0 = rely on the bias
    router_zloss: float = 1e-3
    norm_topk_prob: bool = True

    # --- multi-token prediction (DeepSeek-V3) ---
    mtp_depth: int = 0              # extra future tokens predicted while training
    mtp_weight: float = 0.3

    # --- muP (hyperparameter transfer from a narrow proxy) ---
    use_mup: bool = False
    mup_base_d_model: int = 256

    def __post_init__(self) -> None:
        if self.d_head is None:
            assert self.d_model % self.n_head == 0, "d_model must divide by n_head"
            self.d_head = self.d_model // self.n_head
        if self.ffn_hidden is None:
            # SwiGLU has 3 matrices instead of 2, so shrink by 2/3 to keep the parameter count comparable...
            hidden = int(8 * self.d_model / 3)
            m = self.ffn_multiple_of
            self.ffn_hidden = m * ((hidden + m - 1) // m)
        assert self.n_head % self.n_kv_head == 0, "n_head must divide by n_kv_head"

        if self.attn_type == "mla":
            self.kv_lora_rank = self.kv_lora_rank or max(64, self.d_model // 4)
            self.qk_rope_head_dim = self.qk_rope_head_dim or max(16, self.d_head // 2)
            self.qk_nope_head_dim = self.qk_nope_head_dim or self.d_head
            self.v_head_dim = self.v_head_dim or self.d_head
            assert self.qk_rope_head_dim % 2 == 0, "rope dim must be even"

        if self.moe:
            assert self.n_experts > 0, "moe=True needs n_experts"
            self.n_experts_active = self.n_experts_active or max(1, self.n_experts // 8)
            assert self.n_experts_active <= self.n_experts
            if self.expert_hidden is None:
                # fine-grained experts: total active width matches the dense block
                total = self.ffn_hidden * (self.n_experts_active + self.n_shared_experts)
                self.expert_hidden = max(
                    self.ffn_multiple_of,
                    self.ffn_multiple_of * round(
                        total / (self.n_experts_active + self.n_shared_experts)
                        / self.ffn_multiple_of / 2
                    ),
                )

    @property
    def width_mult(self) -> float:
        """How much wider this model is than the muP base width."""
        return self.d_model / self.mup_base_d_model if self.use_mup else 1.0

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Layer uses experts (leading layers stay dense for stability)."""
        if not self.moe or layer_idx < self.moe_first_dense:
            return False
        return (layer_idx - self.moe_first_dense) % max(1, self.moe_layer_freq) == 0

    def is_nope_layer(self, layer_idx: int) -> bool:
        """Layer runs without rotary embeddings (SmolLM3 NoPE hybrid)."""
        return self.nope_every > 0 and (layer_idx + 1) % self.nope_every == 0

    def is_global_layer(self, layer_idx: int) -> bool:
        """Layer attends over the full context rather than a sliding window."""
        if self.sliding_window <= 0:
            return True
        if self.global_attn_every <= 0:
            return False
        return (layer_idx + 1) % self.global_attn_every == 0

    # -- parameter accounting (exact; verified against the built module) ----- #
    def attn_params(self) -> int:
        """Parameters in one attention block."""
        d, hd, nh, nkv = self.d_model, self.d_head, self.n_head, self.n_kv_head
        if self.attn_type == "mla":
            qk = self.qk_nope_head_dim + self.qk_rope_head_dim
            if self.q_lora_rank:
                q = d * self.q_lora_rank + self.q_lora_rank * nh * qk + self.q_lora_rank
            else:
                q = d * nh * qk
            kv_down = d * (self.kv_lora_rank + self.qk_rope_head_dim)
            kv_up = self.kv_lora_rank * nh * (self.qk_nope_head_dim + self.v_head_dim)
            o = nh * self.v_head_dim * d
            n = q + kv_down + kv_up + o + self.kv_lora_rank
        else:
            n = d * nh * hd + 2 * d * nkv * hd + nh * hd * d
        if self.qk_norm and self.attn_type != "mla":
            n += hd * 2
        return n

    def ffn_params(self, layer_idx: int) -> tuple[int, int]:
        """(total, active) FFN parameters for one layer."""
        d = self.d_model
        if not self.is_moe_layer(layer_idx):
            n = 3 * d * self.ffn_hidden
            return n, n
        eh = self.expert_hidden
        per_expert = 3 * d * eh
        total = per_expert * (self.n_experts + self.n_shared_experts) + d * self.n_experts
        active = per_expert * (self.n_experts_active + self.n_shared_experts)
        return total, active

    def param_count(self) -> dict[str, int]:
        d, L = self.d_model, self.n_layer
        emb = self.vocab_size * d
        attn = self.attn_params()
        norms = 2 * d
        total_blocks = active_blocks = 0
        for i in range(L):
            ffn_total, ffn_active = self.ffn_params(i)
            total_blocks += attn + ffn_total + norms
            active_blocks += attn + ffn_active + norms
        head = 0 if self.tie_embeddings else emb
        # an MTP head is a full block (2 norms) plus its own norm_h/norm_e pair
        mtp = self.mtp_depth * (attn + self.ffn_params(0)[0] + 2 * norms + 2 * d * d)
        non_emb = total_blocks + d
        return {
            "embedding": emb,
            "per_block": total_blocks // max(L, 1),
            "blocks": total_blocks,
            "head": head,
            "mtp": mtp,
            "non_embedding": non_emb,
            "total": emb + non_emb + head + mtp,
            # what a forward pass actually touches - the number that sets speed
            "active": emb + active_blocks + d + head,
        }

    def flops_per_token(self) -> float:
        """Forward+backward FLOPs per token; uses active params, so MoE is costed right."""
        counts = self.param_count()
        n = counts["active"] - counts["embedding"]
        return 6 * n + 12 * self.n_layer * self.n_head * self.d_head * self.max_seq_len


# --- Data ----------------------------------------------------------------------
@dataclass
class DataConfig:
    train_dir: str = "data/tokens/train"
    val_dir: str = "data/tokens/val"
    seq_len: int = 2048
    seed: int = 1337
    # 1 worker keeps resume bit-exact. >1 raises throughput on slow storage but makes the saved...
    num_workers: int = 1
    prefetch: int = 4
    # Sampling weights over sub-corpora, e.g. {"web": 0.8, "code": 0.2}. Empty means uniform over...
    # sampling weights over sub-corpora; keys are subdirectories of train_dir
    mixture: dict[str, float] = field(default_factory=dict)
    # staged mixtures: [{"until": <fraction of the run>, "mixture": {...}}, ...]
    curriculum: list[dict] = field(default_factory=list)
    eos_id: int = -1                # read from the dataset index when available


# --- Optimiser and schedule ----------------------------------------------------
@dataclass
class OptimConfig:
    kind: str = "muon"             # muon | adamw  (A1 decides; see docs/ABLATIONS.md)
    lr: float = 6e-4
    min_lr_ratio: float = 0.1      # final lr = lr * min_lr_ratio
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # "wsd" = warmup-stable-decay. Preferred here: the decay phase can start at *any* step, so a...
    schedule: str = "wsd"          # wsd | cosine | linear | constant
    warmup_steps: int = 2000
    decay_steps: int = 0           # wsd: 0 -> auto (20% of max_steps)
    decay_shape: str = "1-sqrt"    # wsd decay curve: 1-sqrt | linear | cosine
    fused: bool = True
    zloss: float = 1e-4            # logit z-loss; 0 disables
    muon_lr: float = 0.02          # Muon operates on a different scale to AdamW
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_rms_scale: float = 0.2
    independent_wd: bool = True    # decouple wd from lr (AdamW-as-published)


# --- Checkpointing -------------------------------------------------------------
@dataclass
class CheckpointConfig:
    interval: int = 1000           # steps between rolling checkpoints
    keep_last: int = 2             # rolling checkpoints retained
    permanent_every: int = 10_000  # never-deleted milestones
    save_optimizer: bool = True
    save_on_exit: bool = True      # SIGTERM/SIGINT -> flush a checkpoint
    async_save: bool = True        # copy state to CPU, write from a worker thread
    # Remote mirror: "" | "s3://bucket/prefix" | "hf://user/repo" | "file:///path" |...
    remote: str = ""
    remote_every: int = 1          # upload every Nth checkpoint
    verify_hash: bool = True


# --- Fine-tuning and distillation ----------------------------------------------
@dataclass
class FinetuneConfig:
    mode: str = "pretrain"          # pretrain | sft
    data: str = ""                  # jsonl file or directory of conversations
    val_data: str = ""
    tokenizer: str = "data/tokenizer.json"
    drop_long: bool = False
    init_from: str = ""             # checkpoint to start the fine-tune from


@dataclass
class DistillConfig:
    teacher: str = ""               # checkpoint dir or inference export; "" disables
    alpha: float = 0.5              # weight on KD relative to cross entropy
    temperature: float = 2.0
    top_k: int = 0                  # 0 uses the full teacher distribution


# --- Training ------------------------------------------------------------------
@dataclass
class TrainConfig:
    run_name: str = "slm"
    out_dir: str = "runs"
    seed: int = 1337

    # token budget drives everything; max_steps is derived unless set directly
    total_tokens: int = 6_000_000_000
    max_steps: int = 0             # 0 -> derived from total_tokens
    global_batch_tokens: int = 524_288   # ~0.5M tokens/step (GPT-3 small regime)
    micro_batch_size: int = 8      # sequences per fwd/bwd per device

    # precision / speed
    dtype: str = "bfloat16"        # bfloat16 | float16 | float32
    compile: bool = True
    matmul_precision: str = "high" # torch.set_float32_matmul_precision
    grad_checkpoint: bool = False
    # distributed strategy: auto | ddp | fsdp | single
    strategy: str = "auto"
    fsdp_reshard_after_forward: bool = True

    # cadence
    log_every: int = 10
    eval_every: int = 500
    eval_steps: int = 100
    sample_every: int = 1000
    sample_prompt: str = "The key insight is"

    # observability
    wandb_project: str = ""
    wandb_entity: str = ""

    # resume behaviour: auto | none | <path to checkpoint dir>
    resume: str = "auto"
    # exit code used when the job stops early because of preemption, so a supervisor loop knows...
    preempt_exit_code: int = 0
    # stop after N seconds (spot-instance friendly); 0 disables
    max_runtime_sec: int = 0
    # poll the cloud metadata endpoint for a spot-reclaim notice; on for spot runs
    poll_cloud_preemption: bool = False
    # dropping a FINISH file into the run dir re-points max_steps so the WSD
    # decay starts immediately and the run ends properly annealed
    watch_finish_file: bool = True

    def derive(self, model: ModelConfig) -> None:
        if self.max_steps == 0:
            self.max_steps = max(1, self.total_tokens // self.global_batch_tokens)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ckpt: CheckpointConfig = field(default_factory=CheckpointConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)

    def __post_init__(self) -> None:
        # Keep the two sequence lengths in lockstep: the model's positional tables and the loader's...
        if self.data.seq_len != self.model.max_seq_len:
            self.data.seq_len = self.model.max_seq_len
        self.train.derive(self.model)

    @property
    def tokens_per_step(self) -> int:
        return self.train.global_batch_tokens

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# --- (de)serialisation ---------------------------------------------------------
def _coerce(value: Any, typ: Any) -> Any:
    """Best-effort cast of a YAML/CLI scalar into the annotated dataclass type."""
    origin = get_origin(typ)
    if origin is not None:
        args = get_args(typ)
        if origin is tuple:
            return tuple(_coerce(v, args[0] if args else Any) for v in value)
        if origin is list:
            return [_coerce(v, args[0] if args else Any) for v in value]
        if origin is dict:
            return dict(value)
        # Optional[X] / X | None
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            return None
        if len(non_none) == 1:
            return _coerce(value, non_none[0])
        return value
    if typ is Any or typ is None or isinstance(typ, str):
        return value
    if typ is bool and isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    try:
        return typ(value)
    except (TypeError, ValueError):
        return value


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build a (possibly nested) dataclass from a plain dict, ignoring extras."""
    kwargs: dict[str, Any] = {}
    # `from __future__ import annotations` leaves f.type as a string, so nested
    # dataclasses would never be recognised without resolving the hints first
    hints = get_type_hints(cls)
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints[f.name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = from_dict(ftype, value)  # type: ignore[arg-type]
        else:
            kwargs[f.name] = _coerce(value, ftype)
    return cls(**kwargs)  # type: ignore[return-value]


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_scalar(s: str) -> Any:
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def parse_overrides(pairs: Iterable[str]) -> dict[str, Any]:
    """``--optim.lr=3e-4`` / ``optim.lr=3e-4`` -> ``{'optim': {'lr': 3e-4}}``."""
    out: dict[str, Any] = {}
    for raw in pairs:
        item = raw[2:] if raw.startswith("--") else raw
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {raw!r}")
        key, value = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(value)
    return out


def load_config(paths: Iterable[str] = (), overrides: Iterable[str] = ()) -> Config:
    """Merge YAML files (in order) and ``key=value`` overrides into a Config."""
    merged: dict[str, Any] = {}
    for p in paths:
        merged = deep_merge(merged, _load_yaml_with_base(p))
    merged = deep_merge(merged, parse_overrides(overrides))
    return from_dict(Config, merged)


def _load_yaml_with_base(path: str, _seen: set[str] | None = None) -> dict:
    path = os.path.abspath(path)
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular _base_ include at {path}")
    _seen.add(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("_base_", None)
    if base:
        bases = [base] if isinstance(base, str) else list(base)
        acc: dict = {}
        for b in bases:
            bp = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
            acc = deep_merge(acc, _load_yaml_with_base(bp, _seen))
        data = deep_merge(acc, data)
    return data
