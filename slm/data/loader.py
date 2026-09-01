"""Deterministic, resumable data loading."""
from __future__ import annotations

import os
import queue
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from slm.data.permute import epoch_seed, permute
from slm.data.shards import Shard, discover_shards, read_index


class TokenDataset:
    """A directory of shards viewed as one contiguous token stream."""

    def __init__(self, directory: str, seq_len: int):
        self.directory = directory
        self.seq_len = seq_len
        paths = discover_shards(directory)
        if not paths:
            raise FileNotFoundError(f"no token shards under {directory!r}")
        self.paths = paths
        self._shards: list[Shard] | None = None
        self._pid: int | None = None

        try:
            meta = read_index(directory)
            self.vocab_size = meta.get("vocab_size", 0)
            sizes = [s["n_tokens"] for s in meta["shards"]]
        except FileNotFoundError:
            self.vocab_size = 0
            sizes = [Shard(p).header.n_tokens for p in paths]

        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.n_tokens = int(self.offsets[-1])
        self.n_sequences = max(0, (self.n_tokens - 1) // seq_len)
        if self.n_sequences == 0:
            raise ValueError(
                f"{directory!r} holds {self.n_tokens} tokens, too few for "
                f"even one sequence of length {seq_len}"
            )

    # memmaps must be opened in the process that reads them
    @property
    def shards(self) -> list[Shard]:
        if self._shards is None or self._pid != os.getpid():
            self._shards = [Shard(p) for p in self.paths]
            self._pid = os.getpid()
        return self._shards

    def get(self, seq_idx: int, out: np.ndarray | None = None) -> np.ndarray:
        start = int(seq_idx) * self.seq_len
        need = self.seq_len + 1
        if out is None:
            out = np.empty(need, dtype=np.int64)
        s = int(np.searchsorted(self.offsets, start, side="right") - 1)
        filled = 0
        while filled < need:
            shard = self.shards[s]
            local = start + filled - int(self.offsets[s])
            take = min(need - filled, len(shard) - local)
            out[filled : filled + take] = shard.tokens[local : local + take]
            filled += take
            s += 1
            if s >= len(self.shards):        # wrap the very last partial window
                start = -filled
                s = 0
        return out

    def get_batch(self, indices: Sequence[int]) -> np.ndarray:
        out = np.empty((len(indices), self.seq_len + 1), dtype=np.int64)
        for i, idx in enumerate(indices):
            self.get(int(idx), out[i])
        return out

    def close(self) -> None:
        if self._shards:
            for s in self._shards:
                s.close()
        self._shards = None

    def __len__(self) -> int:
        return self.n_sequences

    def __repr__(self) -> str:
        return (
            f"TokenDataset({self.directory!r}, {self.n_tokens:,} tokens, "
            f"{self.n_sequences:,} sequences of {self.seq_len})"
        )


@dataclass
class LoaderState:
    """Everything needed to resume: two integers and the seed they came from."""

    epoch: int = 0
    cursor: int = 0          # sequences consumed globally within this epoch
    tokens_seen: int = 0     # cumulative, across epochs - for logging/schedules
    seed: int = 1337

    def to_dict(self) -> dict:
        return dict(epoch=self.epoch, cursor=self.cursor,
                    tokens_seen=self.tokens_seen, seed=self.seed)

    @classmethod
    def from_dict(cls, d: dict) -> LoaderState:
        return cls(**{k: d[k] for k in ("epoch", "cursor", "tokens_seen", "seed") if k in d})


@dataclass
class _Item:
    x: torch.Tensor
    y: torch.Tensor
    epoch: int
    cursor: int          # cursor value *after* this batch
    tokens_seen: int


class ResumableLoader:
    """Yields ``(x, y)`` micro-batches and can be restored to any past position."""

    def __init__(
        self,
        dataset: TokenDataset,
        micro_batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1337,
        num_workers: int = 2,
        prefetch: int = 4,
        device: str | torch.device = "cpu",
        pin_memory: bool | None = None,
        infinite: bool = True,
    ):
        self.ds = dataset
        self.B = micro_batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.infinite = infinite
        self.device = torch.device(device)
        self.per_step = world_size * micro_batch_size
        if self.per_step > dataset.n_sequences:
            raise ValueError(
                f"global micro-batch {self.per_step} exceeds the {dataset.n_sequences} "
                f"sequences in {dataset.directory!r}"
            )
        self.pin_memory = (
            pin_memory if pin_memory is not None else self.device.type == "cuda"
        )
        self.num_workers = max(0, num_workers)
        self.prefetch = max(1, prefetch)

        self.state = LoaderState(seed=seed)
        self._q: queue.Queue[_Item | None] | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._fetch_lock = threading.Lock()
        # cursor the *producer* has reached; may run ahead of self.state.cursor
        self._fetch_epoch = 0
        self._fetch_cursor = 0
        self._exhausted = False

    # -- the pure ordering function ----------------------------------------- #
    def sequence_indices(self, epoch: int, start: int, count: int) -> np.ndarray:
        n = self.ds.n_sequences
        s = epoch_seed(self.seed, epoch)
        return permute(np.arange(start, start + count, dtype=np.uint64), n, s)

    # -- producer ------------------------------------------------------------ #
    def _next_slot(self) -> tuple[int, int] | None:
        """Reserve the next global window. Returns ``(epoch, cursor_before)``."""
        with self._fetch_lock:
            if self._fetch_cursor + self.per_step > self.ds.n_sequences:
                if not self.infinite:
                    self._exhausted = True
                    return None
                self._fetch_epoch += 1
                self._fetch_cursor = 0
            slot = (self._fetch_epoch, self._fetch_cursor)
            self._fetch_cursor += self.per_step
            return slot

    def _build(self, epoch: int, cursor: int) -> _Item:
        base = cursor + self.rank * self.B
        idx = self.sequence_indices(epoch, base, self.B)
        raw = self.ds.get_batch(idx)
        buf = torch.from_numpy(raw)
        if self.pin_memory:
            buf = buf.pin_memory()
        x, y = buf[:, :-1], buf[:, 1:]
        return _Item(
            x=x.contiguous(), y=y.contiguous(), epoch=epoch,
            cursor=cursor + self.per_step,
            tokens_seen=self.per_step * self.ds.seq_len,
        )

    def _worker(self) -> None:
        while not self._stop.is_set():
            slot = self._next_slot()
            if slot is None:
                self._q.put(None)
                return
            try:
                item = self._build(*slot)
            except Exception as exc:                     # surface, don't hang
                self._q.put(exc)                          # type: ignore[arg-type]
                return
            while not self._stop.is_set():
                try:
                    self._q.put(item, timeout=0.25)
                    break
                except queue.Full:
                    continue

    def _start(self) -> None:
        if self._threads:
            return
        self._fetch_epoch = self.state.epoch
        self._fetch_cursor = self.state.cursor
        self._exhausted = False
        self._stop.clear()
        if self.num_workers == 0:
            return
        self._q = queue.Queue(maxsize=self.prefetch * max(1, self.num_workers))
        for _ in range(self.num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)

    def _restart(self) -> None:
        self.stop()
        self._start()

    def stop(self) -> None:
        self._stop.set()
        if self._q is not None:
            try:
                while True:
                    self._q.get_nowait()
            except queue.Empty:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        self._q = None

    # -- consumer ------------------------------------------------------------ #
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        self._start()
        while True:
            batch = self.next_batch()
            if batch is None:
                return
            yield batch

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self._threads and self.num_workers > 0:
            self._start()

        if self.num_workers == 0:
            slot = self._next_slot()
            if slot is None:
                return None
            item = self._build(*slot)
        else:
            item = self._q.get()                          # type: ignore[union-attr]
            if item is None:
                return None
            if isinstance(item, Exception):
                raise item

        # With >1 worker, batches can arrive out of order. Order does not affect correctness (they...
        if (item.epoch, item.cursor) > (self.state.epoch, self.state.cursor):
            self.state.epoch = item.epoch
            self.state.cursor = item.cursor
        self.state.tokens_seen += item.tokens_seen

        x = item.x.to(self.device, non_blocking=self.pin_memory)
        y = item.y.to(self.device, non_blocking=self.pin_memory)
        return x, y

    # -- state --------------------------------------------------------------- #
    def state_dict(self) -> dict:
        # Multiple in-flight workers make the reported cursor a high-water mark; with more than one...
        return {
            **self.state.to_dict(),
            "per_step": self.per_step,
            "seq_len": self.ds.seq_len,
            "n_sequences": self.ds.n_sequences,
            "exact": self.num_workers <= 1,
        }

    def load_state_dict(self, sd: dict, strict: bool = False) -> None:
        if sd.get("seed", self.seed) != self.seed:
            raise ValueError(
                f"checkpoint data seed {sd.get('seed')} != configured {self.seed}; "
                "resuming would reshuffle the corpus"
            )
        if strict and sd.get("n_sequences") not in (None, self.ds.n_sequences):
            raise ValueError(
                f"dataset changed size ({sd['n_sequences']} -> {self.ds.n_sequences}); "
                "the ordering is defined relative to it. Pass strict=False to "
                "continue anyway (data order after this point will differ)."
            )
        self.state = LoaderState.from_dict(sd)
        # Align the cursor to this job's global micro-batch so ranks stay in lockstep after a...
        if self.state.cursor % self.per_step:
            self.state.cursor -= self.state.cursor % self.per_step
        self._restart()

    def skip(self, n_steps: int) -> None:
        self.state.cursor += n_steps * self.per_step
        while self.state.cursor >= self.ds.n_sequences:
            self.state.cursor -= self.ds.n_sequences
            self.state.epoch += 1
        self._restart()

    @property
    def steps_per_epoch(self) -> int:
        return self.ds.n_sequences // self.per_step

    def __repr__(self) -> str:
        return (
            f"ResumableLoader(B={self.B}, world={self.world_size}, "
            f"steps/epoch={self.steps_per_epoch:,}, epoch={self.state.epoch}, "
            f"cursor={self.state.cursor:,})"
        )


class MixtureLoader:
    """Interleaves several corpora at fixed token ratios, deterministically."""

    PHI = 0.6180339887498949

    def __init__(self, loaders: dict[str, ResumableLoader], weights: dict[str, float]):
        names = sorted(loaders)
        total = sum(weights.get(n, 0.0) for n in names)
        if total <= 0:
            raise ValueError("mixture weights must sum to something positive")
        self.names = names
        self.loaders = loaders
        self.weights = np.array([weights.get(n, 0.0) / total for n in names])
        self.edges = np.cumsum(self.weights)
        self.k = 0

    def _domain(self, k: int) -> str:
        u = (k * self.PHI) % 1.0
        return self.names[int(np.searchsorted(self.edges, u, side="right"))]

    def next_batch(self):
        name = self._domain(self.k)
        self.k += 1
        return self.loaders[name].next_batch()

    def __iter__(self):
        while True:
            b = self.next_batch()
            if b is None:
                return
            yield b

    def realised_weights(self, n: int = 100_000) -> dict[str, float]:
        counts = {n_: 0 for n_ in self.names}
        for k in range(n):
            counts[self._domain(k)] += 1
        return {k: v / n for k, v in counts.items()}

    def set_weights(self, weights: dict[str, float]) -> None:
        """Change the mixture mid-run (curriculum stages)."""
        total = sum(weights.get(n, 0.0) for n in self.names)
        if total <= 0:
            raise ValueError("mixture weights must sum to something positive")
        self.weights = np.array([weights.get(n, 0.0) / total for n in self.names])
        self.edges = np.cumsum(self.weights)

    def state_dict(self) -> dict:
        return {"k": self.k,
                "loaders": {n: ld.state_dict() for n, ld in self.loaders.items()}}

    def load_state_dict(self, sd: dict, strict: bool = False) -> None:
        self.k = sd["k"]
        for n, s in sd["loaders"].items():
            if n in self.loaders:
                self.loaders[n].load_state_dict(s, strict=strict)

    def stop(self) -> None:
        for ld in self.loaders.values():
            ld.stop()

    @property
    def state(self):
        agg = sum(ld.state.tokens_seen for ld in self.loaders.values())
        return LoaderState(epoch=0, cursor=self.k, tokens_seen=agg)
