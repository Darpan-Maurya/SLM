"""BPE tests, including a differential test against a deliberately naive trainer."""
import json
from collections import Counter
from itertools import pairwise

import pytest
import regex as re

from slm.tokenizer import SPLIT_PATTERN, Tokenizer
from slm.tokenizer.bpe import BYTE_TOKENS

CORPUS = (
    "the quick brown fox jumps over the lazy dog. " * 40
    + "the theory of the thermometer is theoretical. " * 25
    + "naive:  café naïve résumé 你好世界 \U0001f600\U0001f680 " * 12
    + "def f(x):\n    return x ** 2  # comment\n" * 18
)


# --------------------------------------------------------------------------- #
# Reference implementation: obviously correct, hopelessly slow.
# --------------------------------------------------------------------------- #
def naive_train(text: str, vocab_size: int, n_special: int = 1):
    """Rescan-everything BPE. Same tie-break rule as the fast trainer:
    highest count wins; ties go to the lexicographically smaller pair."""
    words = [list(w.encode("utf-8")) for w in re.findall(SPLIT_PATTERN, text)]
    merges: dict[tuple[int, int], int] = {}
    n_merges = vocab_size - BYTE_TOKENS - n_special
    next_id = BYTE_TOKENS
    for _ in range(n_merges):
        counts: Counter = Counter()
        for w in words:
            for p in pairwise(w):
                counts[p] += 1
        if not counts:
            break
        _, best = min((-c, p) for p, c in counts.items())
        merges[best] = next_id
        new_words = []
        for w in words:
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    out.append(next_id)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new_words.append(out)
        words = new_words
        next_id += 1
    return merges


def test_fast_trainer_matches_naive_trainer():
    """The incremental trainer must learn byte-identical merges to the naive one."""
    fast = Tokenizer.train([CORPUS], vocab_size=400, min_frequency=1)
    ref = naive_train(CORPUS, vocab_size=400)
    assert list(fast.merges.items()) == list(ref.items())


# --------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.train([CORPUS], vocab_size=512, min_frequency=1)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        "the quick brown fox",
        "café naïve résumé",
        "你好世界",
        "\U0001f600\U0001f680\U0001f9e0",
        "def f(x):\n\treturn x ** 2\n",
        "   leading and trailing   ",
        "mixed 中文 and English \U0001f600 and \n\n newlines",
        "\x00\x01\x02 control bytes \x7f",
        "a" * 500,
    ],
)
def test_roundtrip_is_lossless(tok, text):
    assert tok.decode(tok.encode_ordinary(text)) == text


def test_roundtrip_on_arbitrary_bytes(tok):
    """Byte-level BPE must survive text that is not valid on the first try."""
    raw = bytes(range(256)).decode("latin-1")
    assert tok.decode(tok.encode_ordinary(raw)) == raw


def test_ids_stay_in_vocab(tok):
    ids = tok.encode(CORPUS)
    assert ids and all(0 <= i < tok.vocab_size for i in ids)


# --------------------------------------------------------------------------- #
# Merge behaviour
# --------------------------------------------------------------------------- #
def test_merges_actually_compress():
    text = CORPUS
    small = Tokenizer.train([text], vocab_size=BYTE_TOKENS + 1 + 0, min_frequency=1)
    big = Tokenizer.train([text], vocab_size=1000, min_frequency=1)
    assert len(big.encode_ordinary(text)) < 0.5 * len(small.encode_ordinary(text))
    assert big.compression(text) > 3.0


def test_encoding_applies_lowest_rank_merge_first(tok):
    """Greedy-by-rank must equal replaying the merge list in training order."""
    for word in ["theoretical", "thermometer", "jumps", "résumé"]:
        ids = list(word.encode("utf-8"))
        changed = True
        while changed:                      # replay merges in learned order
            changed = False
            for (a, b), new in tok.merges.items():
                out, i = [], 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                        out.append(new)
                        i += 2
                        changed = True
                    else:
                        out.append(ids[i])
                        i += 1
                ids = out
        assert tok.encode_ordinary(word) == ids, word


def test_encoding_cache_is_transparent(tok):
    a = tok.encode_ordinary("theoretical thermometer")
    b = tok.encode_ordinary("theoretical thermometer")
    assert a == b


def test_pretoken_boundaries_are_never_crossed(tok):
    """No token may contain a letter and a following space-word (GPT-4 pattern).

    Checked only on tokens that are complete UTF-8: a merge may legitimately
    cover part of a multi-byte character, and decoding those lossily would
    manufacture a replacement char that splits into two pretokens.
    """
    checked = 0
    for tid, piece in tok.vocab.items():
        if tid < BYTE_TOKENS:
            continue
        try:
            s = piece.decode("utf-8")
        except UnicodeDecodeError:
            continue
        checked += 1
        assert len(re.findall(SPLIT_PATTERN, s)) <= 1, (tid, s)
    assert checked > 10, "test degenerate: almost nothing was checked"


# --------------------------------------------------------------------------- #
# Special tokens
# --------------------------------------------------------------------------- #
def test_special_tokens_are_atomic():
    t = Tokenizer.train(
        [CORPUS], vocab_size=512,
        special_tokens=("<|endoftext|>", "<|user|>", "<|assistant|>"),
        min_frequency=1,
    )
    ids = t.encode("hello<|endoftext|>world<|user|>hi")
    assert t.special_tokens["<|endoftext|>"] in ids
    assert t.special_tokens["<|user|>"] in ids
    assert t.decode(ids) == "hello<|endoftext|>world<|user|>hi"


def test_special_tokens_sit_at_the_top_of_the_id_space():
    t = Tokenizer.train([CORPUS], vocab_size=512,
                        special_tokens=("<|endoftext|>", "<|eot|>"), min_frequency=1)
    assert t.vocab_size == 512
    assert sorted(t.special_tokens.values()) == [510, 511]
    assert max(t.merges.values()) < 510


def test_encode_ordinary_does_not_honour_specials():
    t = Tokenizer.train([CORPUS], vocab_size=400, min_frequency=1)
    ids = t.encode_ordinary("hello<|endoftext|>")
    assert t.special_tokens["<|endoftext|>"] not in ids
    assert t.decode(ids) == "hello<|endoftext|>"


def test_eos_id_lookup():
    t = Tokenizer.train([CORPUS], vocab_size=400, min_frequency=1)
    assert t.eos_id == t.special_tokens["<|endoftext|>"]


def test_vocab_size_floor_is_enforced():
    with pytest.raises(ValueError):
        Tokenizer.train([CORPUS], vocab_size=100)


# --------------------------------------------------------------------------- #
# Persistence & streaming
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip(tok, tmp_path):
    p = tmp_path / "tok.json"
    tok.save(str(p))
    other = Tokenizer.load(str(p))
    assert other.merges == tok.merges
    assert other.special_tokens == tok.special_tokens
    assert other.vocab_size == tok.vocab_size
    text = "the theoretical thermometer \U0001f600"
    assert other.encode(text) == tok.encode(text)
    assert json.loads(p.read_text())["version"] == 1


def test_decode_stream_holds_back_partial_utf8(tok):
    text = "你好\U0001f600 hello"
    ids = tok.encode_ordinary(text)
    assert "".join(tok.decode_stream(ids)) == text


def test_decode_rejects_unknown_id(tok):
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size + 99])


def test_training_is_deterministic():
    a = Tokenizer.train([CORPUS], vocab_size=400, min_frequency=1)
    b = Tokenizer.train([CORPUS], vocab_size=400, min_frequency=1)
    assert a.merges == b.merges
