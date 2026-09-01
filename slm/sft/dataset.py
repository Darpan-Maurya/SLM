"""In-memory SFT dataset and a loader matching the pretraining loader's interface."""
from __future__ import annotations

import numpy as np
import torch

from slm.data.loader import LoaderState
from slm.data.permute import epoch_seed, permute
from slm.sft.chat import ChatTemplate, pack_examples, read_conversations


class SFTDataset:
    """Packed instruction data with a per-token loss mask."""

    def __init__(self, path: str, tokenizer, seq_len: int, drop_long: bool = False):
        template = ChatTemplate(tokenizer)
        pad = tokenizer.special_tokens.get("<|pad|>", tokenizer.eos_id)
        examples = [template.encode(m) for m in read_conversations(path)]
        if not examples:
            raise ValueError(f"no usable conversations in {path!r}")
        self.tokens, self.masks = pack_examples(examples, seq_len, pad, drop_long)
        self.seq_len = seq_len
        self.n_examples = len(examples)
        self.trainable_fraction = float(self.masks.mean())

    def __len__(self) -> int:
        return len(self.tokens)

    def get(self, idx: int):
        return self.tokens[idx], self.masks[idx]

    def __repr__(self) -> str:
        return (f"SFTDataset({self.n_examples} conversations -> {len(self)} windows "
                f"of {self.seq_len}, {100*self.trainable_fraction:.1f}% trainable)")


class SFTLoader:
    """Same deterministic, resumable contract as the pretraining loader."""

    def __init__(self, dataset: SFTDataset, micro_batch_size: int, rank: int = 0,
                 world_size: int = 1, seed: int = 1337, device="cpu", **_):
        self.ds = dataset
        self.B = micro_batch_size
        self.rank, self.world_size = rank, world_size
        self.seed = seed
        self.device = torch.device(device)
        self.per_step = micro_batch_size * world_size
        self.state = LoaderState(seed=seed)

    def next_batch(self):
        n = len(self.ds)
        if self.state.cursor + self.per_step > n:
            self.state.epoch += 1
            self.state.cursor = 0
        base = self.state.cursor + self.rank * self.B
        idx = permute(np.arange(base, base + self.B, dtype=np.uint64), n,
                      epoch_seed(self.seed, self.state.epoch))
        toks = torch.from_numpy(np.stack([self.ds.tokens[i] for i in idx]))
        mask = torch.from_numpy(np.stack([self.ds.masks[i] for i in idx]))
        self.state.cursor += self.per_step
        self.state.tokens_seen += self.per_step * self.ds.seq_len
        x = toks[:, :-1].to(self.device)
        y = toks[:, 1:].to(self.device)
        m = mask[:, 1:].to(self.device)
        return x, y, m

    def __iter__(self):
        while True:
            yield self.next_batch()

    @property
    def steps_per_epoch(self) -> int:
        return len(self.ds) // self.per_step

    def state_dict(self) -> dict:
        return {**self.state.to_dict(), "per_step": self.per_step, "exact": True}

    def load_state_dict(self, sd: dict, strict: bool = False) -> None:
        self.state = LoaderState.from_dict(sd)

    def stop(self) -> None:
        pass
