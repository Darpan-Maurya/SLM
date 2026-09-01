"""Correctness tests for the transformer."""
import math

import pytest
import torch
import torch.nn.functional as F

from slm.config import ModelConfig
from slm.model import Transformer, apply_rope, build_rope_cache
from slm.model.attention import GroupedQueryAttention, repeat_kv
from slm.model.transformer import cross_entropy_loss

torch.manual_seed(0)


def tiny_cfg(**kw) -> ModelConfig:
    """Small config with the exotic features off unless a test asks for them."""
    base = dict(
        vocab_size=97, n_layer=3, n_head=4, n_kv_head=2, d_model=32,
        max_seq_len=16, ffn_hidden=48, nope_every=0, doc_masking=False,
    )
    base.update(kw)
    return ModelConfig(**base)


# --- rope ----------------------------------------------------------------------
def test_rope_is_relative():
    """<RoPE(q, m), RoPE(k, n)> must depend only on (m - n)."""
    hd = 16
    cos, sin = build_rope_cache(64, hd, dtype=torch.float64)
    q = torch.randn(1, 1, 1, hd, dtype=torch.float64)
    k = torch.randn(1, 1, 1, hd, dtype=torch.float64)

    def dot(m, n):
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum().item()

    for delta in (0, 1, 5, 13):
        vals = [dot(m, m - delta) for m in range(delta, delta + 6)]
        assert max(vals) - min(vals) < 1e-9, f"delta={delta} not translation invariant"


def test_rope_norm_preserving():
    cos, sin = build_rope_cache(32, 8, dtype=torch.float64)
    x = torch.randn(2, 3, 32, 8, dtype=torch.float64)
    assert torch.allclose(x.norm(dim=-1), apply_rope(x, cos, sin).norm(dim=-1), atol=1e-10)


# --- attention -----------------------------------------------------------------
def test_repeat_kv_groups_are_contiguous():
    x = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).view(2, 2, 3, 4)
    y = repeat_kv(x, 3)
    assert y.shape == (2, 6, 3, 4)
    for g in range(3):
        assert torch.equal(y[:, g], x[:, 0])
        assert torch.equal(y[:, 3 + g], x[:, 1])


def test_attention_matches_naive_reference():
    """The fused SDPA path must equal a hand-written softmax attention."""
    cfg = tiny_cfg(qk_norm=False)
    attn = GroupedQueryAttention(cfg, 0).double()
    B, T = 2, 9
    x = torch.randn(B, T, cfg.d_model, dtype=torch.float64)
    cos, sin = build_rope_cache(T, cfg.d_head, cfg.rope_theta, dtype=torch.float64)
    fast = attn(x, (cos, sin))

    q, k, v = attn.wqkv(x).split(attn.split, dim=-1)
    q = q.view(B, T, cfg.n_head, cfg.d_head).transpose(1, 2)
    k = k.view(B, T, cfg.n_kv_head, cfg.d_head).transpose(1, 2)
    v = v.view(B, T, cfg.n_kv_head, cfg.d_head).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    k, v = repeat_kv(k, attn.n_rep), repeat_kv(v, attn.n_rep)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(cfg.d_head)
    att = F.softmax(att + torch.full((T, T), float("-inf"), dtype=torch.float64).triu(1), -1)
    ref = attn.wo((att @ v).transpose(1, 2).reshape(B, T, -1))
    assert torch.allclose(fast, ref, atol=1e-10), (fast - ref).abs().max()


# --- causality -----------------------------------------------------------------
def test_model_is_causal():
    """Perturbing token t must not change any logit at position < t."""
    m = Transformer(tiny_cfg()).double().eval()
    idx = torch.randint(0, 97, (1, 12))
    with torch.no_grad():
        base = m(idx).logits
        for t in (3, 7, 11):
            alt = idx.clone()
            alt[0, t] = (alt[0, t] + 5) % 97
            new = m(alt).logits
            assert torch.allclose(base[:, :t], new[:, :t], atol=1e-12), f"leak at t={t}"
            assert not torch.allclose(base[:, t], new[:, t]), f"no effect at t={t}"


def test_gradient_flows_to_every_parameter():
    cfg = tiny_cfg()
    m = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    m(idx, targets=idx, zloss=1e-4).loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


# --- kv cache ------------------------------------------------------------------
def test_kv_cache_matches_full_forward():
    """Token-by-token decoding must reproduce the parallel forward exactly."""
    m = Transformer(tiny_cfg()).double().eval()
    idx = torch.randint(0, 97, (2, 12))
    with torch.no_grad():
        full = m(idx).logits
        cache = m.alloc_cache(batch=2)
        step = torch.cat(
            [m(idx[:, t : t + 1], cache=cache, start_pos=t).logits for t in range(12)],
            dim=1,
        )
    assert torch.allclose(full, step, atol=1e-9), (full - step).abs().max()


def test_kv_cache_prefill_then_decode():
    """Chunked prefill (T>1 into a non-empty cache) must also match."""
    m = Transformer(tiny_cfg()).double().eval()
    idx = torch.randint(0, 97, (1, 12))
    with torch.no_grad():
        full = m(idx).logits
        cache = m.alloc_cache(batch=1)
        parts = [
            m(idx[:, :5], cache=cache, start_pos=0).logits,
            m(idx[:, 5:9], cache=cache, start_pos=5).logits,
            m(idx[:, 9:], cache=cache, start_pos=9).logits,
        ]
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-9)


def test_cache_reorder_and_reset():
    m = Transformer(tiny_cfg()).eval()
    cache = m.alloc_cache(batch=3)
    m(torch.randint(0, 97, (3, 4)), cache=cache, start_pos=0)
    before = cache.slots[0][0].clone()
    cache.reorder(torch.tensor([2, 0, 1]))
    assert torch.equal(cache.slots[0][0][0], before[2])
    assert cache.bytes() > 0
    cache.reset()
    assert cache.slots[0] is None


# --- loss ----------------------------------------------------------------------
def test_loss_mask_excludes_tokens():
    logits = torch.randn(1, 4, 10)
    targets = torch.randint(0, 10, (1, 4))
    masked, mm = cross_entropy_loss(logits, targets, loss_mask=torch.tensor([[1., 1., 0., 0.]]))
    manual, _ = cross_entropy_loss(logits[:, :2], targets[:, :2])
    assert torch.allclose(masked, manual, atol=1e-6)
    assert mm["n_tokens"].item() == 2


def test_ignore_index_matches_mask():
    logits = torch.randn(2, 5, 11)
    targets = torch.randint(0, 11, (2, 5))
    targets[:, 3:] = -100
    loss, m = cross_entropy_loss(logits, targets)
    ref = F.cross_entropy(logits[:, :3].reshape(-1, 11).float(), targets[:, :3].reshape(-1))
    assert torch.allclose(loss, ref, atol=1e-6)
    assert m["n_tokens"].item() == 6


def test_zloss_penalises_large_logits():
    targets = torch.randint(0, 10, (1, 4))
    small = torch.randn(1, 4, 10) * 0.1
    big = small + 12.0
    assert torch.allclose(cross_entropy_loss(small, targets)[0],
                          cross_entropy_loss(big, targets)[0], atol=1e-4)
    z_s = cross_entropy_loss(small, targets, zloss=1.0)[0]
    z_b = cross_entropy_loss(big, targets, zloss=1.0)[0]
    assert z_b > z_s + 10.0, "z-loss must punish an inflated softmax denominator"


def test_untrained_loss_is_near_uniform():
    """A fresh model must start at ln(V); a wrong init shows up here immediately."""
    cfg = tiny_cfg(vocab_size=1024, n_layer=4)
    m = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    targets = torch.randint(0, cfg.vocab_size, (4, 16))
    with torch.no_grad():
        loss = m(idx, targets=targets).loss
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.35, loss.item()


def test_tied_embeddings_produce_a_copy_bias():
    """Documents the flip side of tying: at init, P(x_t | ... x_t) > uniform."""
    cfg = tiny_cfg(vocab_size=1024, n_layer=4)
    m = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    with torch.no_grad():
        assert m(idx, targets=idx).loss.item() < math.log(cfg.vocab_size) - 0.2


# --- structure -----------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    dict(n_layer=12, n_head=12, n_kv_head=3, d_model=768),
    dict(),
    dict(n_layer=26, n_head=20, n_kv_head=4, d_model=1280),
    dict(tie_embeddings=False, n_layer=2, d_model=128, n_head=4, n_kv_head=2),
    dict(qk_norm=False, n_layer=2, d_model=128, n_head=4, n_kv_head=2),
    dict(attn_type="mla", n_layer=2, d_model=128, n_head=4, n_kv_head=4),
    dict(n_layer=4, d_model=128, n_head=4, n_kv_head=2, moe=True, n_experts=8,
         n_experts_active=2, expert_hidden=64),
    dict(n_layer=2, d_model=128, n_head=4, n_kv_head=2, mtp_depth=1),
])
def test_param_count_matches_analytic_formula(kw):
    cfg = ModelConfig(**kw)
    assert cfg.param_count()["total"] == Transformer(cfg).num_params(), kw


def test_tied_embeddings_share_storage():
    m = Transformer(tiny_cfg(tie_embeddings=True))
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    m2 = Transformer(tiny_cfg(tie_embeddings=False))
    assert m2.lm_head.weight.data_ptr() != m2.tok_emb.weight.data_ptr()


def test_param_groups_cover_every_parameter():
    """Every parameter appears exactly once; only matrices are weight-decayed."""
    m = Transformer(tiny_cfg())
    groups = m.param_groups(0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter landed in two groups"
    assert sum(p.numel() for g in groups for p in g["params"]) == m.num_params()
    for g in groups:
        if g["weight_decay"] > 0:
            assert all(p.dim() >= 2 for p in g["params"])
        else:
            assert all(p.dim() < 2 for p in g["params"])


def test_muon_split_excludes_embeddings_and_head():
    m = Transformer(tiny_cfg(tie_embeddings=False, moe=True, n_experts=4,
                             n_experts_active=2, expert_hidden=32, n_layer=2))
    muon, other = m.muon_split()
    assert all(p.dim() == 2 for p in muon)
    ids = {id(p) for p in muon}
    assert id(m.tok_emb.weight) not in ids and id(m.lm_head.weight) not in ids
    for name, p in m.named_parameters():
        if "router" in name:
            assert id(p) not in ids, "router weights must stay on AdamW"
    assert sum(p.numel() for p in muon + other) == m.num_params()


def test_grad_checkpointing_gives_same_gradients():
    torch.manual_seed(7)
    cfg = tiny_cfg()
    a, b = Transformer(cfg).double(), Transformer(cfg).double()
    b.load_state_dict(a.state_dict())
    b.set_grad_checkpoint(True)
    idx = torch.randint(0, cfg.vocab_size, (2, 10))
    for m in (a, b):
        m(idx, targets=idx).loss.backward()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-10), n


def test_seq_len_guard():
    m = Transformer(tiny_cfg(max_seq_len=8))
    with pytest.raises(AssertionError):
        m(torch.zeros(1, 9, dtype=torch.long))
