"""Zero-shot scoring by continuation log-likelihood."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch


@dataclass
class Doc:
    """One multiple-choice item."""

    context: str
    choices: list[str]
    gold: int
    meta: dict = field(default_factory=dict)


@dataclass
class Request:
    """A (context, continuation) pair whose log-probability we need."""

    context: str
    continuation: str
    doc_index: int
    choice_index: int


@dataclass
class Score:
    logprob: float
    n_tokens: int
    n_chars: int
    is_greedy: bool

    @property
    def per_token(self) -> float:
        return self.logprob / max(self.n_tokens, 1)

    @property
    def per_char(self) -> float:
        return self.logprob / max(self.n_chars, 1)


@torch.inference_mode()
def score_requests(
    model,
    tokenizer,
    requests: Sequence[Request],
    batch_size: int = 8,
    max_len: int | None = None,
    device: torch.device | None = None,
) -> list[Score]:
    """Log P(continuation | context) for each request."""
    model.eval()
    device = device or next(model.parameters()).device
    max_len = max_len or model.cfg.max_seq_len
    encoded = []
    for r in requests:
        ctx = tokenizer.encode_ordinary(r.context) or [tokenizer.eos_id]
        cont = tokenizer.encode_ordinary(r.continuation)
        if not cont:
            cont = [tokenizer.eos_id]
        ids = (ctx + cont)[-max_len:]
        n_cont = min(len(cont), len(ids) - 1)
        encoded.append((ids, n_cont, len(r.continuation)))

    # longest first, so consecutive batches share a shape and the device allocator reuses buffers
    order = sorted(range(len(encoded)), key=lambda k: -len(encoded[k][0]))
    out: list[Score | None] = [None] * len(encoded)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        chunk = [encoded[k] for k in idx]
        width = max(len(ids) for ids, _, _ in chunk)
        # right padding is safe under causal attention: padded positions can only
        # influence tokens after them, and we never read those
        batch = torch.zeros(len(chunk), width, dtype=torch.long)
        for i, (ids, _, _) in enumerate(chunk):
            batch[i, : len(ids)] = torch.tensor(ids)
        batch = batch.to(device)
        logits = model(batch[:, :-1]).logits.float()
        targets = batch[:, 1:]
        # one read-back per batch, and no log-softmax materialised over the vocabulary
        token_lp = (logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    - torch.logsumexp(logits, dim=-1)).cpu().double()
        hit = (logits.argmax(-1) == targets).cpu()
        del logits

        for i, (k, (ids, n_cont, n_chars)) in enumerate(zip(idx, chunk)):
            end_pos = len(ids) - 1
            span = slice(end_pos - n_cont, end_pos)
            out[k] = Score(float(token_lp[i, span].sum()), n_cont, n_chars,
                           bool(hit[i, span].all()))
    return out


def build_requests(docs: Sequence[Doc]) -> list[Request]:
    return [
        Request(d.context, c, i, j)
        for i, d in enumerate(docs)
        for j, c in enumerate(d.choices)
    ]


def evaluate_docs(
    model, tokenizer, docs: Sequence[Doc], batch_size: int = 8
) -> dict[str, float]:
    """Accuracy under raw, length-normalised and per-token scoring."""
    if not docs:
        return {}
    requests = build_requests(docs)
    scores = score_requests(model, tokenizer, requests, batch_size)

    by_doc: dict[int, list[tuple[int, Score]]] = {}
    for r, s in zip(requests, scores):
        by_doc.setdefault(r.doc_index, []).append((r.choice_index, s))

    hits = {"acc": 0, "acc_norm": 0, "acc_token": 0}
    for i, entries in by_doc.items():
        entries.sort()
        if len(entries) == 1:               # one continuation (LAMBADA): right only if greedy
            for k in hits:
                hits[k] += int(entries[0][1].is_greedy)
            continue
        gold = docs[i].gold
        raw = [s.logprob for _, s in entries]
        norm = [s.per_char for _, s in entries]
        tok = [s.per_token for _, s in entries]
        hits["acc"] += int(_argmax(raw) == gold)
        hits["acc_norm"] += int(_argmax(norm) == gold)
        hits["acc_token"] += int(_argmax(tok) == gold)

    n = len(by_doc)
    result = {k: v / n for k, v in hits.items()}
    result["n"] = n
    result["random_baseline"] = sum(1 / len(d.choices) if len(d.choices) > 1 else 0.0
                                    for d in docs) / n
    return result


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


@torch.inference_mode()
def token_perplexity(
    model, loader, steps: int = 100, bytes_per_token: float = 0.0
) -> dict[str, float]:
    """Mean NLL over `steps` batches, plus perplexity and bits-per-byte."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for _ in range(steps):
        batch = loader.next_batch()
        if batch is None:
            break
        x, y = batch
        out = model(x, targets=y, return_logits=False)
        n = float(out.metrics["n_tokens"])
        total_nll += float(out.metrics["ce"]) * n
        total_tokens += n
    if total_tokens == 0:
        return {}
    nll = total_nll / total_tokens
    res = {"loss": nll, "ppl": math.exp(min(nll, 20)), "n_tokens": total_tokens}
    if bytes_per_token > 0:
        # bits-per-byte is the only cross-tokenizer-comparable quality number
        res["bpb"] = nll / math.log(2) / bytes_per_token
    return res


def format_table(results: dict[str, dict], metric: str = "acc_norm") -> str:
    """Render a results dict as a markdown table."""
    rows = ["| task | n | " + metric + " | random |", "|---|---|---|---|"]
    total, count = 0.0, 0
    for name, res in sorted(results.items()):
        if metric not in res:
            continue
        rows.append(
            f"| {name} | {int(res['n'])} | {100*res[metric]:.2f} | "
            f"{100*res.get('random_baseline', 0):.2f} |"
        )
        total += res[metric]
        count += 1
    if count:
        rows.append(f"| **average** | | **{100*total/count:.2f}** | |")
    return "\n".join(rows)
