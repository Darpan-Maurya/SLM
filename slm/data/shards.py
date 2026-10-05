"""On-disk token shard format."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

MAGIC = b"SLMTOK\x00\x00"
VERSION = 1
HEADER_BYTES = 1024
_DTYPES = {0: np.uint16, 1: np.uint32}
_CODES = {np.dtype(np.uint16): 0, np.dtype(np.uint32): 1}


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    """uint16 buys a 2x smaller dataset; use it whenever the vocab allows."""
    return np.dtype(np.uint16) if vocab_size <= 65536 else np.dtype(np.uint32)


@dataclass
class ShardHeader:
    version: int
    dtype_code: int
    n_tokens: int
    vocab_size: int
    n_docs: int = 0

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(_DTYPES[self.dtype_code])

    def pack(self) -> bytes:
        buf = bytearray(HEADER_BYTES)
        buf[0:8] = MAGIC
        meta = np.array(
            [self.version, self.dtype_code, self.n_tokens, self.vocab_size, self.n_docs],
            dtype=np.uint64,
        )
        buf[8 : 8 + meta.nbytes] = meta.tobytes()
        return bytes(buf)

    @classmethod
    def unpack(cls, raw: bytes) -> ShardHeader:
        if raw[0:8] != MAGIC:
            raise ValueError("not an SLM token shard (bad magic)")
        meta = np.frombuffer(raw[8 : 8 + 5 * 8], dtype=np.uint64)
        version = int(meta[0])
        if version != VERSION:
            raise ValueError(f"shard version {version} != supported {VERSION}")
        return cls(version, int(meta[1]), int(meta[2]), int(meta[3]), int(meta[4]))


class ShardWriter:
    """Append token batches, then ``close()`` to stamp the final count."""

    def __init__(self, path: str, vocab_size: int, track_docs: bool = True):
        self.path = path
        self.tmp = path + ".tmp"
        self.dtype = dtype_for_vocab(vocab_size)
        self.vocab_size = vocab_size
        self.n_tokens = 0
        self.track_docs = track_docs
        self.doc_starts: list[int] = []
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._f = open(self.tmp, "wb")  # noqa: SIM115 - owned until close()
        self._f.write(b"\0" * HEADER_BYTES)   # reserve; rewritten on close

    def write(self, tokens: Sequence[int] | np.ndarray, is_document: bool = True) -> None:
        arr = np.asarray(tokens, dtype=self.dtype)
        if arr.size == 0:
            return
        if arr.max(initial=0) >= self.vocab_size:
            raise ValueError(
                f"token id {int(arr.max())} >= vocab_size {self.vocab_size}"
            )
        if self.track_docs and is_document:
            self.doc_starts.append(self.n_tokens)
        self._f.write(arr.tobytes())
        self.n_tokens += arr.size

    def close(self) -> dict:
        self._f.flush()
        self._f.seek(0)
        header = ShardHeader(
            VERSION, _CODES[self.dtype], self.n_tokens, self.vocab_size,
            len(self.doc_starts),
        )
        self._f.write(header.pack())
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        os.replace(self.tmp, self.path)

        if self.track_docs and self.doc_starts:
            docs = np.asarray(self.doc_starts, dtype=np.uint64)
            dtmp = self.path + ".docs.tmp"
            with open(dtmp, "wb") as f:
                f.write(docs.tobytes())
                f.flush()
                os.fsync(f.fileno())
            os.replace(dtmp, self.path + ".docs")
        return {
            "name": os.path.basename(self.path),
            "n_tokens": self.n_tokens,
            "n_docs": len(self.doc_starts),
            "dtype": str(self.dtype),
        }

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *exc) -> None:
        if not self._f.closed:
            self.close()


class Shard:
    """Read-only memory-mapped view of one shard."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            self.header = ShardHeader.unpack(f.read(HEADER_BYTES))
        self._tokens: np.memmap | None = None
        self._docs: np.ndarray | None = None

    @property
    def tokens(self) -> np.memmap:
        # Opened lazily and per-process: a memmap must never be inherited across a fork, or two...
        if self._tokens is None:
            self._tokens = np.memmap(
                self.path, dtype=self.header.dtype, mode="r",
                offset=HEADER_BYTES, shape=(self.header.n_tokens,),
            )
        return self._tokens

    @property
    def doc_starts(self) -> np.ndarray:
        if self._docs is None:
            p = self.path + ".docs"
            self._docs = (
                np.fromfile(p, dtype=np.uint64) if os.path.exists(p)
                else np.zeros(0, dtype=np.uint64)
            )
        return self._docs

    def __len__(self) -> int:
        return self.header.n_tokens

    def sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        return h.hexdigest()

    def close(self) -> None:
        self._tokens = None

    def __repr__(self) -> str:
        return f"Shard({os.path.basename(self.path)}, {self.header.n_tokens:,} tokens)"


def write_index(
    directory: str, shards: list[dict], vocab_size: int, extra: dict | None = None
) -> str:
    payload = {
        "version": VERSION,
        "vocab_size": vocab_size,
        "n_tokens": sum(s["n_tokens"] for s in shards),
        "n_shards": len(shards),
        "shards": shards,
        **(extra or {}),
    }
    path = os.path.join(directory, "index.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def read_index(directory: str) -> dict:
    with open(os.path.join(directory, "index.json")) as f:
        return json.load(f)


def discover_shards(directory: str) -> list[str]:
    """Shard paths in manifest order, falling back to a sorted glob."""
    idx = os.path.join(directory, "index.json")
    if os.path.exists(idx):
        meta = read_index(directory)
        return [os.path.join(directory, s["name"]) for s in meta["shards"]]
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".bin")
    )


def iter_tokens(directory: str, chunk: int = 1 << 20) -> Iterator[np.ndarray]:
    for path in discover_shards(directory):
        shard = Shard(path)
        toks = shard.tokens
        for start in range(0, len(toks), chunk):
            yield np.asarray(toks[start : start + chunk])
        shard.close()
