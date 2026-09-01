"""Sampling and generation tests."""
import pytest
import torch

from slm.config import ModelConfig
from slm.infer.generate import (
    SamplingConfig,
    apply_penalties,
    filter_logits,
    generate,
    generate_stream,
    sample_from,
    speculative_generate,
)
from slm.model import Transformer


def model_for(**kw):
    base = dict(vocab_size=128, n_layer=2, n_head=2, n_kv_head=1, d_model=64,
                max_seq_len=96, ffn_hidden=96, nope_every=0, doc_masking=False)
    base.update(kw)
    return Transformer(ModelConfig(**base)).eval()


# --- logit filtering -----------------------------------------------------------
def test_top_k_keeps_exactly_k():
    logits = torch.randn(2, 100)
    out = filter_logits(logits, SamplingConfig(top_k=7, top_p=1.0, temperature=1.0))
    assert (torch.isfinite(out).sum(-1) == 7).all()


def test_top_p_keeps_the_nucleus():
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.15, 0.06, 0.04]]))
    out = filter_logits(logits, SamplingConfig(top_p=0.9, top_k=0, temperature=1.0))
    kept = torch.isfinite(out)[0]
    assert kept.tolist() == [True, True, True, True, False]


def test_min_p_drops_low_relative_mass():
    logits = torch.log(torch.tensor([[0.8, 0.1, 0.05, 0.05]]))
    out = filter_logits(logits, SamplingConfig(min_p=0.2, top_k=0, top_p=1.0,
                                               temperature=1.0))
    assert torch.isfinite(out)[0].tolist() == [True, False, False, False]


def test_temperature_sharpens_and_flattens():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    cold = filter_logits(logits.clone(), SamplingConfig(temperature=0.1, top_k=0, top_p=1.0))
    hot = filter_logits(logits.clone(), SamplingConfig(temperature=10.0, top_k=0, top_p=1.0))
    assert cold.softmax(-1)[0, 0] > logits.softmax(-1)[0, 0] > hot.softmax(-1)[0, 0]


def test_repetition_penalty_lowers_seen_tokens():
    logits = torch.ones(1, 10)
    seen = torch.tensor([[3, 3, 7]])
    out = apply_penalties(logits.clone(), seen, SamplingConfig(repetition_penalty=2.0))
    assert out[0, 3] < logits[0, 3] and out[0, 7] < logits[0, 7]
    assert out[0, 5] == logits[0, 5]


def test_frequency_penalty_scales_with_count():
    logits = torch.zeros(1, 10)
    seen = torch.tensor([[3, 3, 3, 7]])
    out = apply_penalties(logits.clone(), seen, SamplingConfig(frequency_penalty=1.0))
    assert out[0, 3] < out[0, 7] < out[0, 5]


def test_greedy_sampling_takes_the_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    assert int(sample_from(logits, SamplingConfig(greedy=True))) == 1


# --- generation ----------------------------------------------------------------
def test_generate_respects_the_token_budget():
    m = model_for()
    out = generate(m, torch.randint(0, 128, (2, 5)), SamplingConfig(max_new_tokens=11))
    assert out.shape == (2, 16)


def test_generate_never_exceeds_max_seq_len():
    m = model_for(max_seq_len=20)
    out = generate(m, torch.randint(0, 128, (1, 15)), SamplingConfig(max_new_tokens=100))
    assert out.shape[1] <= 20


def test_generate_is_deterministic_given_a_seed():
    m = model_for()
    p = torch.randint(0, 128, (1, 4))
    a = generate(m, p, SamplingConfig(max_new_tokens=12, seed=7))
    b = generate(m, p, SamplingConfig(max_new_tokens=12, seed=7))
    assert torch.equal(a, b)


def test_different_seeds_diverge():
    m = model_for()
    p = torch.randint(0, 128, (1, 4))
    a = generate(m, p, SamplingConfig(max_new_tokens=20, seed=1, temperature=1.5))
    b = generate(m, p, SamplingConfig(max_new_tokens=20, seed=2, temperature=1.5))
    assert not torch.equal(a, b)


def test_greedy_generation_matches_an_uncached_loop():
    """The cached fast path must agree with recomputing the whole prefix each step."""
    m = model_for().double()
    p = torch.randint(0, 128, (1, 6))
    fast = generate(m, p, SamplingConfig(max_new_tokens=8, greedy=True))

    slow = p.clone()
    for _ in range(8):
        logits = m(slow).logits[:, -1]
        slow = torch.cat([slow, logits.argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(fast, slow)


def test_stream_yields_one_token_at_a_time():
    m = model_for()
    toks = list(generate_stream(m, torch.randint(0, 128, (1, 3)),
                                SamplingConfig(max_new_tokens=9, seed=0)))
    assert len(toks) == 9 and all(t.shape == (1, 1) for t in toks)


def test_stop_token_ends_generation():
    m = model_for(vocab_size=8)
    stop = 3
    out = list(generate_stream(m, torch.randint(0, 8, (1, 4)),
                               SamplingConfig(max_new_tokens=60, seed=0,
                                              stop_tokens=(stop,), temperature=2.0)))
    assert int(out[-1]) == stop
    assert len(out) < 60


def test_batch_generation_rows_are_independent():
    m = model_for()
    p = torch.stack([torch.arange(5), torch.arange(5) + 40])
    out = generate(m, p, SamplingConfig(max_new_tokens=6, greedy=True))
    assert out.shape == (2, 11)
    assert not torch.equal(out[0, 5:], out[1, 5:])


# --- speculative decoding ------------------------------------------------------
def test_speculation_with_an_identical_draft_accepts_everything():
    """draft == target means p_draft == p_target, so the accept rule must never reject."""
    m = model_for(mtp_depth=0)
    seq, stats = speculative_generate(
        m, torch.randint(0, 128, (1, 4)),
        SamplingConfig(max_new_tokens=16, seed=3, temperature=1.0),
        draft=m, gamma=3,
    )
    assert stats["acceptance_rate"] == 1.0, stats
    assert seq.shape[1] > 4


def test_mtp_self_speculation_runs_and_produces_valid_tokens():
    m = model_for(mtp_depth=2)
    seq, stats = speculative_generate(
        m, torch.randint(0, 128, (1, 4)),
        SamplingConfig(max_new_tokens=16, seed=1, temperature=1.0), gamma=3,
    )
    assert seq.shape[1] >= 4
    assert int(seq.max()) < 128 and int(seq.min()) >= 0
    assert 0.0 <= stats["acceptance_rate"] <= 1.0
    assert stats["proposed"] > 0


def test_speculation_requires_a_draft_source():
    m = model_for(mtp_depth=0)
    with pytest.raises(ValueError, match="nothing to speculate"):
        speculative_generate(m, torch.randint(0, 128, (1, 4)), SamplingConfig())
