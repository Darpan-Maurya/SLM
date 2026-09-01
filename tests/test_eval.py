"""Evaluation harness tests: the harness itself must be verified before it judges models."""
import math

import pytest
import torch
import torch.nn.functional as F

from slm.config import ModelConfig
from slm.eval import Doc, evaluate_docs, format_table, load_task, score_requests
from slm.eval.harness import Request, token_perplexity
from slm.eval.tasks import REGISTRY
from slm.model import Transformer
from slm.tokenizer import Tokenizer

CORPUS = "the cat sat on the mat . the dog ran in the park . " * 60


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.train([CORPUS], vocab_size=320, min_frequency=1)


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    return Transformer(ModelConfig(
        vocab_size=tok.vocab_size, n_layer=2, n_head=2, n_kv_head=1, d_model=64,
        max_seq_len=128, ffn_hidden=96, nope_every=0, doc_masking=False,
    )).eval()


# --- the scorer must be exactly right ------------------------------------------
def test_score_matches_a_hand_computed_logprob(model, tok):
    """Compare against log-softmax computed directly, with no harness involved."""
    ctx, cont = "the cat", " sat on the mat"
    score = score_requests(model, tok, [Request(ctx, cont, 0, 0)], batch_size=1)[0]

    ctx_ids = tok.encode_ordinary(ctx)
    cont_ids = tok.encode_ordinary(cont)
    ids = torch.tensor([ctx_ids + cont_ids])
    with torch.no_grad():
        logprobs = F.log_softmax(model(ids[:, :-1]).logits.float(), dim=-1)
    expected = sum(
        float(logprobs[0, len(ctx_ids) - 1 + i, cont_ids[i]])
        for i in range(len(cont_ids))
    )
    assert abs(score.logprob - expected) < 1e-4, (score.logprob, expected)
    assert score.n_tokens == len(cont_ids)


def test_batching_does_not_change_scores(model, tok):
    """Right padding must not leak: the classic harness bug."""
    reqs = [
        Request("the cat", " sat", 0, 0),
        Request("the dog ran in the park and then", " some much longer continuation here", 1, 0),
        Request("a", " b", 2, 0),
    ]
    one = score_requests(model, tok, reqs, batch_size=1)
    many = score_requests(model, tok, reqs, batch_size=8)
    for a, b in zip(one, many):
        assert abs(a.logprob - b.logprob) < 1e-4, (a.logprob, b.logprob)


def test_reordering_requests_does_not_change_scores(model, tok):
    reqs = [Request("the cat", " sat", 0, 0), Request("the dog", " ran in the park", 1, 0)]
    forward = score_requests(model, tok, reqs, batch_size=8)
    backward = score_requests(model, tok, list(reversed(reqs)), batch_size=8)
    assert abs(forward[0].logprob - backward[1].logprob) < 1e-4


def test_logprobs_are_negative_and_bounded(model, tok):
    s = score_requests(model, tok, [Request("the", " cat sat", 0, 0)])[0]
    assert s.logprob < 0
    assert s.logprob > -s.n_tokens * math.log(tok.vocab_size) * 3


def test_long_input_is_truncated_not_crashed(model, tok):
    long_ctx = "the cat sat on the mat . " * 200
    s = score_requests(model, tok, [Request(long_ctx, " end", 0, 0)])[0]
    assert math.isfinite(s.logprob)


# --- accuracy aggregation ------------------------------------------------------
def test_evaluate_docs_reports_every_metric(model, tok):
    docs = [Doc("the cat", [" sat", " flew"], 0), Doc("the dog", [" ran", " sat"], 0)]
    res = evaluate_docs(model, tok, docs)
    assert set(res) >= {"acc", "acc_norm", "acc_token", "n", "random_baseline"}
    assert res["n"] == 2
    assert res["random_baseline"] == 0.5
    assert all(0.0 <= res[k] <= 1.0 for k in ("acc", "acc_norm", "acc_token"))


def test_a_model_that_memorised_the_corpus_scores_the_right_choice():
    """Train on one sentence, then the harness must prefer the true continuation."""
    text = "the cat sat on the mat . " * 400
    tk = Tokenizer.train([text], vocab_size=300, min_frequency=1)
    ids = torch.tensor(tk.encode_ordinary(text))
    cfg = ModelConfig(vocab_size=tk.vocab_size, n_layer=2, n_head=2, n_kv_head=1,
                      d_model=64, max_seq_len=64, ffn_hidden=128, nope_every=0,
                      doc_masking=False)
    torch.manual_seed(0)
    m = Transformer(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(120):
        i = torch.randint(0, len(ids) - 65, (8,))
        batch = torch.stack([ids[j : j + 65] for j in i])
        opt.zero_grad()
        m(batch[:, :-1], targets=batch[:, 1:]).loss.backward()
        opt.step()
    m.eval()

    docs = [Doc("the cat sat on the", [" mat", " park", " dog"], 0)]
    res = evaluate_docs(m, tk, docs)
    assert res["acc"] == 1.0, "harness failed on a corpus the model memorised"


def test_argmax_ties_do_not_crash(model, tok):
    docs = [Doc("x", [" a", " a"], 0)]
    assert evaluate_docs(model, tok, docs)["n"] == 1


def test_empty_docs_returns_empty(model, tok):
    assert evaluate_docs(model, tok, []) == {}


# --- perplexity ----------------------------------------------------------------
def test_token_perplexity_matches_direct_computation(model, tok, tmp_path):
    from slm.data import ResumableLoader, ShardWriter, TokenDataset, write_index

    d = tmp_path / "toks"
    d.mkdir()
    ids = tok.encode_ordinary(CORPUS)
    with ShardWriter(str(d / "s0.bin"), tok.vocab_size) as w:
        w.write(ids)
    write_index(str(d), [{"name": "s0.bin", "n_tokens": len(ids), "n_docs": 1,
                          "dtype": "uint16"}], tok.vocab_size)
    loader = ResumableLoader(TokenDataset(str(d), 64), 2, num_workers=0)
    res = token_perplexity(model, loader, steps=3, bytes_per_token=3.5)
    assert math.isclose(res["ppl"], math.exp(res["loss"]), rel_tol=1e-6)
    assert math.isclose(res["bpb"], res["loss"] / math.log(2) / 3.5, rel_tol=1e-6)
    assert 0 < res["loss"] < math.log(tok.vocab_size) + 1


# --- task registry -------------------------------------------------------------
def test_sanity_task_is_wellformed():
    docs = load_task("sanity")
    assert docs and all(0 <= d.gold < len(d.choices) for d in docs)
    assert all(len(d.choices) >= 2 for d in docs)


def test_unknown_task_raises():
    with pytest.raises(KeyError, match="unknown task"):
        load_task("not_a_real_task")


def test_registry_covers_the_standard_suite():
    from slm.eval.tasks import DEFAULT_SUITE

    assert set(DEFAULT_SUITE) <= set(REGISTRY)


def test_format_table_renders_markdown():
    out = format_table({"piqa": {"acc_norm": 0.61, "n": 100, "random_baseline": 0.5}})
    assert "| piqa |" in out and "61.00" in out and "**average**" in out
