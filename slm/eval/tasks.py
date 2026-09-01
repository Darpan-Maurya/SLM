"""Benchmark task loaders in cloze (continuation) form."""
from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable, Iterator

from slm.eval.harness import Doc

CACHE_DIR = os.environ.get("SLM_EVAL_CACHE", "data/eval")

# direct downloads so a run does not depend on the `datasets` package
URLS = {
    "hellaswag": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "lambada": "https://raw.githubusercontent.com/openai/gpt-2/master/data/lambada_test.jsonl",
}

REGISTRY: dict[str, Callable[[int], list[Doc]]] = {}


def register(name: str):
    def wrap(fn):
        REGISTRY[name] = fn
        return fn
    return wrap


def _cached(name: str, url: str) -> str | None:
    """Download once into CACHE_DIR; return the path or None if unreachable."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.jsonl")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(path, "wb") as f:
            f.write(resp.read())
        return path
    except Exception as exc:
        print(f"[eval] could not fetch {name}: {exc!r}")
        if os.path.exists(path):
            os.remove(path)
        return None


def _read_jsonl(path: str, limit: int = 0) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def _hf(name: str, config: str | None, split: str, limit: int) -> list[dict] | None:
    try:
        from datasets import load_dataset

        ds = load_dataset(name, config, split=split)
        return list(ds.select(range(min(limit or len(ds), len(ds)))))
    except Exception as exc:
        print(f"[eval] datasets unavailable for {name}: {exc!r}")
        return None


@register("hellaswag")
def hellaswag(limit: int = 0) -> list[Doc]:
    path = _cached("hellaswag", URLS["hellaswag"])
    if not path:
        return []
    docs = []
    for row in _read_jsonl(path, limit):
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize() if row.get("ctx_b") else row["ctx"]
        docs.append(Doc(
            context=row.get("activity_label", "") + ": " + ctx,
            choices=[" " + e.strip() for e in row["endings"]],
            gold=int(row["label"]),
        ))
    return docs


@register("lambada")
def lambada(limit: int = 0) -> list[Doc]:
    """Last-word prediction: one choice, scored as exact greedy continuation."""
    path = _cached("lambada", URLS["lambada"])
    if not path:
        return []
    docs = []
    for row in _read_jsonl(path, limit):
        text = row["text"] if isinstance(row, dict) else str(row)
        context, _, last = text.rstrip().rpartition(" ")
        docs.append(Doc(context=context, choices=[" " + last], gold=0))
    return docs


def _mc_from_hf(name, config, split, limit, ctx_key, choices_key, label_key):
    rows = _hf(name, config, split, limit)
    if rows is None:
        return []
    docs = []
    for row in rows:
        choices = row[choices_key]
        if isinstance(choices, dict):
            labels, texts = choices.get("label", []), choices.get("text", [])
            gold = labels.index(row[label_key]) if row[label_key] in labels else 0
            choices = texts
        else:
            gold = int(row[label_key])
        docs.append(Doc(context=row[ctx_key],
                        choices=[" " + str(c).strip() for c in choices], gold=gold))
    return docs


@register("arc_easy")
def arc_easy(limit: int = 0) -> list[Doc]:
    return _mc_from_hf("allenai/ai2_arc", "ARC-Easy", "test", limit,
                       "question", "choices", "answerKey")


@register("arc_challenge")
def arc_challenge(limit: int = 0) -> list[Doc]:
    return _mc_from_hf("allenai/ai2_arc", "ARC-Challenge", "test", limit,
                       "question", "choices", "answerKey")


@register("piqa")
def piqa(limit: int = 0) -> list[Doc]:
    rows = _hf("ybisk/piqa", None, "validation", limit)
    if rows is None:
        return []
    return [Doc(context=r["goal"], choices=[" " + r["sol1"], " " + r["sol2"]],
                gold=int(r["label"])) for r in rows]


@register("winogrande")
def winogrande(limit: int = 0) -> list[Doc]:
    rows = _hf("allenai/winogrande", "winogrande_xl", "validation", limit)
    if rows is None:
        return []
    docs = []
    for r in rows:
        prefix, _, suffix = r["sentence"].partition("_")
        docs.append(Doc(context=prefix.rstrip(),
                        choices=[" " + r["option1"] + suffix, " " + r["option2"] + suffix],
                        gold=int(r["answer"]) - 1))
    return docs


@register("openbookqa")
def openbookqa(limit: int = 0) -> list[Doc]:
    return _mc_from_hf("allenai/openbookqa", "main", "test", limit,
                       "question_stem", "choices", "answerKey")


@register("sanity")
def sanity(limit: int = 0) -> list[Doc]:
    """A trivially learnable task: any harness that fails this is broken."""
    pairs = [("The capital of France is", [" Paris", " banana", " seventeen"], 0),
             ("Two plus two equals", [" four", " purple", " Tuesday"], 0),
             ("The sky is", [" blue", " loud", " Thursday"], 0)]
    docs = [Doc(c, ch, g) for c, ch, g in pairs]
    return docs[:limit] if limit else docs


def load_task(name: str, limit: int = 0) -> list[Doc]:
    if name not in REGISTRY:
        raise KeyError(f"unknown task {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](limit)


DEFAULT_SUITE = ("hellaswag", "arc_easy", "arc_challenge", "piqa",
                 "winogrande", "openbookqa", "lambada")
