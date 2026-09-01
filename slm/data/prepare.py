"""Turn a text corpus into tokenised shards."""
from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
import time
from collections.abc import Iterable, Iterator

import numpy as np

from slm.data.shards import ShardWriter, write_index
from slm.tokenizer import Tokenizer

_WORKER_TOK: Tokenizer | None = None


def _init_worker(tokenizer_path: str) -> None:
    global _WORKER_TOK
    _WORKER_TOK = Tokenizer.load(tokenizer_path)


def _encode_chunk(docs: list[str]) -> tuple[list[list[int]], int]:
    assert _WORKER_TOK is not None
    eos = _WORKER_TOK.eos_id
    out, nbytes = [], 0
    for d in docs:
        nbytes += len(d.encode("utf-8"))
        out.append([*_WORKER_TOK.encode_ordinary(d), eos])
    return out, nbytes


# --- sources -------------------------------------------------------------------
def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def iter_file(path: str, text_key: str = "text", split_blank: bool = False) -> Iterator[str]:
    """Yield documents from .jsonl(.gz) or plain text."""
    if ".jsonl" in path or path.endswith(".json"):
        with _open_text(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get(text_key) if isinstance(obj, dict) else str(obj)
                if text:
                    yield text
    else:
        with _open_text(path) as f:
            content = f.read()
        if split_blank:
            for part in content.split("\n\n"):
                if part.strip():
                    yield part + "\n\n"
        else:
            yield content


def iter_source(source: str, text_key: str = "text", limit: int = 0) -> Iterator[str]:
    """Documents from a file, a directory of files, or `hf://dataset[:split]`."""
    if source.startswith("hf://"):
        it = _iter_hf(source, text_key)
    elif os.path.isdir(source):
        it = (
            doc
            for fn in sorted(os.listdir(source))
            if not fn.endswith((".bin", ".json", ".docs"))
            for doc in iter_file(os.path.join(source, fn), text_key)
        )
    else:
        it = iter_file(source, text_key)
    for n, doc in enumerate(it, start=1):
        yield doc
        if limit and n >= limit:
            return


def _iter_hf(source: str, text_key: str) -> Iterator[str]:
    """Stream a Hugging Face dataset without downloading it whole."""
    from datasets import load_dataset

    spec = source.removeprefix("hf://")
    name, _, split = spec.partition(":")
    name, _, config = name.partition("@")
    ds = load_dataset(name, config or None, split=split or "train", streaming=True)
    for row in ds:
        text = row.get(text_key)
        if text:
            yield text


# --- driver --------------------------------------------------------------------
def tokenize_to_shards(
    documents: Iterable[str],
    tokenizer_path: str,
    out_dir: str,
    shard_tokens: int = 100_000_000,
    workers: int = 0,
    chunk_docs: int = 256,
    val_fraction: float = 0.0,
    val_dir: str | None = None,
    verbose: bool = True,
) -> dict:
    """Encode documents in parallel and write uint16/uint32 shards plus an index."""
    tok = Tokenizer.load(tokenizer_path)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    os.makedirs(out_dir, exist_ok=True)
    val_dir = val_dir or (out_dir.rstrip("/") + "_val" if val_fraction > 0 else None)
    if val_dir:
        os.makedirs(val_dir, exist_ok=True)

    writers = _WriterPool(out_dir, tok.vocab_size, shard_tokens)
    val_writers = _WriterPool(val_dir, tok.vocab_size, shard_tokens) if val_dir else None

    total_tokens = total_bytes = total_docs = 0
    t0 = time.time()
    pool = None
    try:
        if workers > 1:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(workers, initializer=_init_worker, initargs=(tokenizer_path,))
            results = pool.imap(_encode_chunk, _chunks(documents, chunk_docs), chunksize=1)
        else:
            _init_worker(tokenizer_path)
            results = (_encode_chunk(c) for c in _chunks(documents, chunk_docs))

        for encoded, nbytes in results:
            total_bytes += nbytes
            for ids in encoded:
                total_docs += 1
                target = writers
                if val_writers is not None and (total_docs % int(1 / val_fraction) == 0):
                    target = val_writers
                target.write(ids)
                total_tokens += len(ids)
            if verbose and total_docs % 10_000 < chunk_docs:
                rate = total_tokens / max(time.time() - t0, 1e-9)
                print(f"[prepare] {total_docs:,} docs | {total_tokens:,} tokens "
                      f"| {rate/1e6:.2f}M tok/s", flush=True)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    extra = {
        "eos_id": tok.eos_id,
        "tokenizer": os.path.basename(tokenizer_path),
        "bytes_per_token": total_bytes / max(total_tokens, 1),
        "n_documents": total_docs,
    }
    meta = writers.close(tok.vocab_size, extra)
    if val_writers is not None:
        val_writers.close(tok.vocab_size, extra)
    if verbose:
        print(f"[prepare] done: {total_tokens:,} tokens in {meta['n_shards']} shards "
              f"| {extra['bytes_per_token']:.2f} bytes/token "
              f"| {time.time()-t0:.1f}s")
    return meta


class _WriterPool:
    """Rolls to a new shard once the current one reaches the size target."""

    def __init__(self, out_dir: str, vocab_size: int, shard_tokens: int):
        self.out_dir = out_dir
        self.vocab_size = vocab_size
        self.shard_tokens = shard_tokens
        self.shards: list[dict] = []
        self.writer: ShardWriter | None = None

    def write(self, ids: list[int]) -> None:
        if self.writer is None:
            self.writer = ShardWriter(
                os.path.join(self.out_dir, f"shard_{len(self.shards):05d}.bin"),
                self.vocab_size,
            )
        self.writer.write(np.asarray(ids, dtype=np.int64))
        if self.writer.n_tokens >= self.shard_tokens:
            self.shards.append(self.writer.close())
            self.writer = None

    def close(self, vocab_size: int, extra: dict) -> dict:
        if self.writer is not None:
            self.shards.append(self.writer.close())
            self.writer = None
        write_index(self.out_dir, self.shards, vocab_size, extra)
        return {"n_shards": len(self.shards),
                "n_tokens": sum(s["n_tokens"] for s in self.shards)}


def _chunks(it: Iterable[str], n: int) -> Iterator[list[str]]:
    buf: list[str] = []
    for item in it:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
