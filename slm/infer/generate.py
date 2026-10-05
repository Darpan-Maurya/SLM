"""Sampling and generation with a KV cache."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch

from slm.model import Transformer


@dataclass
class SamplingConfig:
    """Decoding knobs."""

    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    stop_tokens: tuple[int, ...] = ()
    seed: int | None = None
    greedy: bool = False
    extra: dict = field(default_factory=dict)


def apply_penalties(
    logits: torch.Tensor, generated: torch.Tensor, cfg: SamplingConfig
) -> torch.Tensor:
    """Discourage repetition using the tokens produced so far."""
    if cfg.repetition_penalty == 1.0 and cfg.frequency_penalty == 0.0:
        return logits
    for b in range(logits.shape[0]):
        seen, counts = torch.unique(generated[b], return_counts=True)
        if cfg.repetition_penalty != 1.0:
            vals = logits[b, seen]
            # penalise by division for positive logits, multiplication for negative
            logits[b, seen] = torch.where(
                vals > 0, vals / cfg.repetition_penalty, vals * cfg.repetition_penalty
            )
        if cfg.frequency_penalty:
            logits[b, seen] -= cfg.frequency_penalty * counts.to(logits.dtype)
    return logits


def filter_logits(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """Apply temperature and the top-k / top-p / min-p truncations."""
    if cfg.temperature != 1.0:
        logits = logits / max(cfg.temperature, 1e-6)

    if cfg.top_k and cfg.top_k < logits.shape[-1]:
        kth = torch.topk(logits, cfg.top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if cfg.min_p > 0:
        probs = logits.softmax(-1)
        threshold = cfg.min_p * probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < threshold, float("-inf"))

    if cfg.top_p and cfg.top_p < 1.0:
        ordered, index = torch.sort(logits, descending=True, dim=-1)
        cumulative = ordered.softmax(-1).cumsum(-1)
        drop = cumulative - ordered.softmax(-1) > cfg.top_p
        ordered = ordered.masked_fill(drop, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, index, ordered)
    return logits


def sample_from(logits: torch.Tensor, cfg: SamplingConfig, gen=None) -> torch.Tensor:
    """One token per row."""
    if cfg.greedy or cfg.temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    probs = filter_logits(logits, cfg).softmax(-1)
    return torch.multinomial(probs, num_samples=1, generator=gen)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    cfg: SamplingConfig | None = None,
) -> torch.Tensor:
    """Continue `prompt` (B, T) and return the full sequence including the prompt."""
    if prompt.dim() == 1:
        prompt = prompt[None]
    pieces = [prompt.to(next(model.parameters()).device)]
    pieces.extend(_generate_tokens(model, prompt, cfg))
    return torch.cat(pieces, dim=1)


@torch.inference_mode()
def generate_stream(
    model: Transformer,
    prompt: torch.Tensor,
    cfg: SamplingConfig | None = None,
) -> Iterator[torch.Tensor]:
    """Yield one (B, 1) token tensor at a time."""
    yield from _generate_tokens(model, prompt, cfg)


def _generate_tokens(
    model: Transformer, prompt: torch.Tensor, cfg: SamplingConfig | None
) -> Iterator[torch.Tensor]:
    cfg = cfg or SamplingConfig()
    model.eval()
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    if prompt.dim() == 1:
        prompt = prompt[None]
    B, T = prompt.shape

    gen = None
    if cfg.seed is not None:
        gen = torch.Generator(device=device).manual_seed(cfg.seed)

    budget = min(cfg.max_new_tokens, model.cfg.max_seq_len - T)
    if budget <= 0:
        return
    cache = model.alloc_cache(B, max_len=T + budget)

    out = model(prompt, cache=cache, start_pos=0, return_logits=False)
    logits = out.logits[:, -1].float()
    produced = prompt
    done = torch.zeros(B, dtype=torch.bool, device=device)
    stops = torch.tensor(cfg.stop_tokens, device=device) if cfg.stop_tokens else None

    pad = int(stops[0]) if stops is not None else 0
    for i in range(budget):
        logits = apply_penalties(logits, produced, cfg)
        nxt = sample_from(logits, cfg, gen)
        if stops is not None:
            # rows that already stopped keep emitting the stop token as padding,
            # and the token that *causes* the stop is still handed to the caller
            nxt = torch.where(done[:, None], torch.full_like(nxt, pad), nxt)
            done = done | (nxt.squeeze(1)[:, None] == stops[None]).any(-1)
        yield nxt
        if stops is not None and bool(done.all()):
            return
        produced = torch.cat([produced, nxt], dim=1)
        if i + 1 >= budget:
            return
        out = model(nxt, cache=cache, start_pos=T + i, return_logits=False)
        logits = out.logits[:, -1].float()


@torch.inference_mode()
def speculative_generate(
    model: Transformer,
    prompt: torch.Tensor,
    cfg: SamplingConfig | None = None,
    draft: Transformer | None = None,
    gamma: int = 3,
) -> tuple[torch.Tensor, dict]:
    """Speculative decoding: a draft proposes `gamma` tokens, the target verifies them.

    Uses `draft` if given, otherwise the model's own MTP heads. Returns the
    sequence and acceptance statistics.
    """
    cfg = cfg or SamplingConfig()
    if draft is None and not len(model.mtp):
        raise ValueError("no draft model and no MTP heads: nothing to speculate with")
    model.eval()
    device = next(model.parameters()).device
    seq = prompt.to(device)
    if seq.dim() == 1:
        seq = seq[None]
    assert seq.shape[0] == 1, "speculative decoding here handles one sequence"

    gen = torch.Generator(device=device).manual_seed(cfg.seed) if cfg.seed else None
    gamma = min(gamma, len(model.mtp)) if draft is None else gamma
    budget = min(cfg.max_new_tokens, model.cfg.max_seq_len - seq.shape[1])
    proposed = accepted = 0

    while seq.shape[1] < prompt.shape[-1] + budget:
        drafts, draft_probs = _propose(model, draft, seq, cfg, gamma, gen)
        if not drafts:
            break
        candidate = torch.cat([seq, torch.tensor([drafts], device=device)], dim=1)
        logits = model(candidate, return_logits=True).logits[0].float()

        n_new = 0
        for j, tok in enumerate(drafts):
            proposed += 1
            target = filter_logits(logits[seq.shape[1] - 1 + j][None], cfg).softmax(-1)[0]
            p_t = float(target[tok])
            p_d = float(draft_probs[j][tok])
            if p_d <= 0 or torch.rand((), generator=gen, device=device).item() < min(1.0, p_t / p_d):
                n_new += 1
                accepted += 1
                continue
            residual = (target - draft_probs[j].to(target.device)).clamp(min=0)
            if float(residual.sum()) <= 0:
                residual = target
            fixed = int(torch.multinomial(residual / residual.sum(), 1, generator=gen))
            seq = torch.cat([candidate[:, : seq.shape[1] + n_new],
                             torch.tensor([[fixed]], device=device)], dim=1)
            n_new = -1
            break
        if n_new >= 0:
            seq = candidate[:, : seq.shape[1] + n_new]
            bonus = sample_from(filter_logits(
                logits[seq.shape[1] - 1][None], cfg), cfg, gen)
            seq = torch.cat([seq, bonus], dim=1)
        if cfg.stop_tokens and int(seq[0, -1]) in cfg.stop_tokens:
            break

    stats = {"proposed": proposed, "accepted": accepted,
             "acceptance_rate": accepted / max(proposed, 1)}
    return seq, stats


def _propose(model, draft, seq, cfg, gamma, gen):
    """Draft up to `gamma` tokens and the distributions they came from."""
    tokens: list[int] = []
    probs: list[torch.Tensor] = []
    device = seq.device

    def take(logits):
        p = filter_logits(logits.float(), cfg).softmax(-1)[0]
        tok = int(torch.multinomial(p, 1, generator=gen))
        tokens.append(tok)
        probs.append(p)
        return tok

    if draft is not None:
        work = seq
        for _ in range(gamma):
            tok = take(draft(work, return_logits=True).logits[:, -1])
            work = torch.cat([work, torch.tensor([[tok]], device=device)], dim=1)
        return tokens, probs

    # self-speculation: the main model gives token t+1, each MTP head gives one more
    out = model(seq, return_logits=True, return_hidden=True)
    hidden = out.hidden[:, -1:]
    tok = take(out.logits[:, -1])
    position = seq.shape[1]
    for depth in range(min(gamma - 1, len(model.mtp))):
        logits, hidden = model.mtp_step(
            hidden, torch.tensor([[tok]], device=device), depth, position
        )
        tok = take(logits[:, -1])
        position += 1
    return tokens, probs


def decode_text(tokenizer, ids: Sequence[int] | torch.Tensor) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return tokenizer.decode(ids)
