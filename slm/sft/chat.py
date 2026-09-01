"""Chat templating and prompt-masked SFT examples."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

ROLE_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Example:
    """Token ids plus a mask marking the positions that contribute to the loss."""

    ids: list[int]
    mask: list[int]

    def __len__(self) -> int:
        return len(self.ids)


class ChatTemplate:
    """Renders messages into tokens, training only on assistant turns."""

    def __init__(self, tokenizer, train_on_all_assistant_turns: bool = True):
        self.tok = tokenizer
        self.all_turns = train_on_all_assistant_turns
        missing = [t for t in ROLE_TOKENS.values() if t not in tokenizer.special_tokens]
        if missing:
            raise ValueError(f"tokenizer lacks chat special tokens: {missing}")
        self.eos = tokenizer.eos_id

    def render_prompt(self, messages: Sequence[Message]) -> str:
        """Text form of a conversation, ending ready for the assistant to speak."""
        parts = [f"{ROLE_TOKENS[m.role]}{m.content}" for m in messages]
        parts.append(ROLE_TOKENS["assistant"])
        return "".join(parts)

    def encode(self, messages: Sequence[Message]) -> Example:
        ids: list[int] = []
        mask: list[int] = []
        for m in messages:
            head = self.tok.encode(ROLE_TOKENS[m.role], allowed_special="all")
            body = self.tok.encode_ordinary(m.content)
            ids += head
            mask += [0] * len(head)
            if m.role == "assistant":
                # loss on the reply and on the EOS that ends it, never on the prompt
                ids += [*body, self.eos]
                mask += [1] * (len(body) + 1)
            else:
                ids += body
                mask += [0] * len(body)
        return Example(ids, mask)


def parse_record(record: dict) -> list[Message] | None:
    """Accept OpenAI `messages`, Alpaca `instruction/input/output`, or prompt/response."""
    if "messages" in record:
        msgs = [Message(m["role"], m["content"]) for m in record["messages"]
                if m.get("role") in ROLE_TOKENS and m.get("content")]
        return msgs or None
    if "instruction" in record:
        user = record["instruction"]
        if record.get("input"):
            user = f"{user}\n\n{record['input']}"
        out = record.get("output") or record.get("response")
        return [Message("user", user), Message("assistant", out)] if out else None
    prompt = record.get("prompt") or record.get("question")
    reply = record.get("response") or record.get("completion") or record.get("answer")
    if prompt and reply:
        return [Message("user", prompt), Message("assistant", reply)]
    return None


def read_conversations(path: str) -> Iterator[list[Message]]:
    """Conversations from a .jsonl file or a directory of them."""
    paths = (
        [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".jsonl")]
        if os.path.isdir(path) else [path]
    )
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = parse_record(record)
                if msgs:
                    yield msgs


def pack_examples(
    examples: Sequence[Example], seq_len: int, pad_id: int, drop_long: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Pack examples into fixed windows without letting one example bleed into the next."""
    xs, ms = [], []
    cur_ids: list[int] = []
    cur_mask: list[int] = []
    for ex in examples:
        ids, mask = ex.ids, ex.mask
        if len(ids) > seq_len + 1:
            if drop_long:
                continue
            ids, mask = ids[: seq_len + 1], mask[: seq_len + 1]
        if len(cur_ids) + len(ids) > seq_len + 1:
            pad = seq_len + 1 - len(cur_ids)
            xs.append(cur_ids + [pad_id] * pad)
            ms.append(cur_mask + [0] * pad)
            cur_ids, cur_mask = [], []
        cur_ids += ids
        cur_mask += mask
    if cur_ids:
        pad = seq_len + 1 - len(cur_ids)
        xs.append(cur_ids + [pad_id] * pad)
        ms.append(cur_mask + [0] * pad)
    return np.asarray(xs, dtype=np.int64), np.asarray(ms, dtype=np.int64)
