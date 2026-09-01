"""Byte-level BPE, trained from scratch."""
from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from heapq import heapify, heappop, heappush
from itertools import pairwise

import regex as re

# GPT-4's split pattern (cl100k_base). Keeps contractions, caps runs of letters and digits,...
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

BYTE_TOKENS = 256


class Tokenizer:
    """A trained byte-level BPE tokenizer."""

    def __init__(
        self,
        merges: dict[tuple[int, int], int] | None = None,
        special_tokens: dict[str, int] | None = None,
        pattern: str = SPLIT_PATTERN,
    ):
        self.pattern = pattern
        self._re = re.compile(pattern)
        self.merges = merges or {}
        self.special_tokens = special_tokens or {}
        self._rebuild()

    # -- derived tables ------------------------------------------------------ #
    def _rebuild(self) -> None:
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(BYTE_TOKENS)}
        for (a, b), new_id in self.merges.items():
            vocab[new_id] = vocab[a] + vocab[b]
        self.vocab = vocab
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.inv_special = {v: k for k, v in self.special_tokens.items()}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(k) for k in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        n = BYTE_TOKENS + len(self.merges)
        if self.special_tokens:
            n = max(n, max(self.special_tokens.values()) + 1)
        return n

    @property
    def eos_id(self) -> int:
        for name in ("<|endoftext|>", "<|eot|>", "</s>"):
            if name in self.special_tokens:
                return self.special_tokens[name]
        raise KeyError("no end-of-text special token in this tokenizer")

    # -- training ------------------------------------------------------------ #
    @classmethod
    def train(
        cls,
        text_iter: Iterable[str],
        vocab_size: int,
        special_tokens: Sequence[str] = ("<|endoftext|>",),
        pattern: str = SPLIT_PATTERN,
        verbose: bool = False,
        min_frequency: int = 2,
    ) -> Tokenizer:
        """Learn merges until ``vocab_size`` is reached."""
        n_special = len(special_tokens)
        n_merges = vocab_size - BYTE_TOKENS - n_special
        if n_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} is below the floor of "
                f"{BYTE_TOKENS + n_special} (bytes + specials)"
            )

        splitter = re.compile(pattern)
        word_freq: Counter[bytes] = Counter()
        for chunk in text_iter:
            for piece in splitter.findall(chunk):
                word_freq[piece.encode("utf-8")] += 1
        if verbose:
            total = sum(word_freq.values())
            print(f"[bpe] {len(word_freq):,} unique words / {total:,} total")

        words: list[list[int]] = []
        freqs: list[int] = []
        for w, c in word_freq.items():
            if c < min_frequency and len(word_freq) > 100_000:
                continue
            words.append(list(w))
            freqs.append(c)

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_where: dict[tuple[int, int], set[int]] = {}
        for i, w in enumerate(words):
            for pair in pairwise(w):
                pair_counts[pair] += freqs[i]
                pair_where.setdefault(pair, set()).add(i)

        heap = [(-c, p) for p, c in pair_counts.items()]
        heapify(heap)

        merges: dict[tuple[int, int], int] = {}
        next_id = BYTE_TOKENS
        while len(merges) < n_merges:
            best = None
            while heap:
                neg, pair = heappop(heap)
                if pair_counts.get(pair, 0) == -neg and -neg > 0:
                    best = pair
                    break
            if best is None:
                if verbose:
                    print(f"[bpe] corpus exhausted after {len(merges)} merges")
                break

            new_id = next_id
            next_id += 1
            merges[best] = new_id
            touched = pair_where.pop(best, set())
            dirty: set[tuple[int, int]] = set()

            for wi in touched:
                w = words[wi]
                f = freqs[wi]
                # withdraw this word's contribution
                for p in pairwise(w):
                    pair_counts[p] -= f
                    dirty.add(p)
                    s = pair_where.get(p)
                    if s is not None:
                        s.discard(wi)
                merged = _merge_symbols(w, best, new_id)
                words[wi] = merged
                for p in pairwise(merged):
                    pair_counts[p] += f
                    dirty.add(p)
                    pair_where.setdefault(p, set()).add(wi)

            for p in dirty:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
                    pair_where.pop(p, None)

            if verbose and len(merges) % 1000 == 0:
                print(f"[bpe] merge {len(merges):>6}/{n_merges}  {best} -> {new_id}")

        specials = {tok: vocab_size - n_special + i for i, tok in enumerate(special_tokens)}
        return cls(merges=merges, special_tokens=specials, pattern=pattern)

    # -- encoding ------------------------------------------------------------ #
    def _encode_word(self, piece: bytes) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = list(piece)
        if len(ids) > 1:
            while True:
                best_rank = None
                best_i = -1
                for i in range(len(ids) - 1):
                    r = self.ranks.get((ids[i], ids[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank, best_i = r, i
                if best_rank is None:
                    break
                new_id = self.merges[(ids[best_i], ids[best_i + 1])]
                ids[best_i : best_i + 2] = [new_id]
        if len(self._cache) < 500_000:
            self._cache[piece] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode without interpreting special tokens (safe for untrusted text)."""
        out: list[int] = []
        for piece in self._re.findall(text):
            out.extend(self._encode_word(piece.encode("utf-8")))
        return out

    def encode(self, text: str, allowed_special: str | set[str] = "all") -> list[int]:
        """Encode, honouring special tokens."""
        if not self._special_re or allowed_special == "none":
            return self.encode_ordinary(text)
        allowed = (
            set(self.special_tokens) if allowed_special == "all" else set(allowed_special)
        )
        out: list[int] = []
        for part in self._special_re.split(text):
            if part in allowed:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def encode_batch(self, texts: Sequence[str], **kw) -> list[list[int]]:
        return [self.encode(t, **kw) for t in texts]

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        parts: list[bytes] = []
        for i in ids:
            i = int(i)
            if i in self.inv_special:
                parts.append(self.inv_special[i].encode("utf-8"))
            else:
                tok = self.vocab.get(i)
                if tok is None:
                    raise ValueError(f"id {i} is outside this vocabulary")
                parts.append(tok)
        return b"".join(parts).decode("utf-8", errors=errors)

    def decode_stream(self, ids: Iterable[int]) -> Iterator[str]:
        """Yield text incrementally, holding back partial UTF-8 sequences."""
        buf = b""
        for i in ids:
            i = int(i)
            buf += (
                self.inv_special[i].encode("utf-8")
                if i in self.inv_special
                else self.vocab[i]
            )
            try:
                text = buf.decode("utf-8")
            except UnicodeDecodeError:
                continue
            buf = b""
            yield text
        if buf:
            yield buf.decode("utf-8", errors="replace")

    # -- persistence --------------------------------------------------------- #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            # list-of-triples keeps merge order explicit and JSON-safe
            "merges": [[a, b, i] for (a, b), i in self.merges.items()],
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        merges = {(a, b): i for a, b, i in payload["merges"]}
        return cls(
            merges=merges,
            special_tokens=payload.get("special_tokens", {}),
            pattern=payload.get("pattern", SPLIT_PATTERN),
        )

    # -- diagnostics --------------------------------------------------------- #
    def compression(self, text: str) -> float:
        """Bytes per token - the number that decides how far a token budget goes."""
        n = len(self.encode_ordinary(text))
        return len(text.encode("utf-8")) / max(n, 1)

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)}, "
            f"special={list(self.special_tokens)})"
        )


def _merge_symbols(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    a, b = pair
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
