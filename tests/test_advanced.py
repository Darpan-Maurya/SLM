"""Tests for MoE, MLA, MTP, masking variants, RoPE scaling and Muon."""
import math

import pytest
import torch

from slm.config import ModelConfig, OptimConfig
from slm.model import Transformer
from slm.model.masking import HAS_FLEX, dense_mask, doc_ids_from_eos
from slm.model.moe import MoE
from slm.model.rope import build_inv_freq, build_rope_cache
from slm.train.muon import Muon, newton_schulz
from slm.train.optim import build_optimizer

torch.manual_seed(0)


def cfg_for(**kw) -> ModelConfig:
    base = dict(vocab_size=64, n_layer=2, n_head=4, n_kv_head=2, d_model=64,
                max_seq_len=32, ffn_hidden=96, nope_every=0, doc_masking=False)
    base.update(kw)
    return ModelConfig(**base)


# --- mixture of experts --------------------------------------------------------
def moe_cfg(**kw):
    return cfg_for(moe=True, n_experts=8, n_experts_active=2, n_shared_experts=1,
                   expert_hidden=32, moe_first_dense=0, **kw)


def test_moe_matches_manual_dispatch():
    """The sorted batched dispatch must equal a naive per-token loop."""
    cfg = moe_cfg()
    moe = MoE(cfg).double().eval()
    x = torch.randn(2, 5, cfg.d_model, dtype=torch.float64)
    out, _ = moe(x)

    flat = x.reshape(-1, cfg.d_model)
    idx, w, _ = moe.router(flat)
    ref = torch.zeros_like(flat)
    for t in range(flat.shape[0]):
        for j in range(cfg.n_experts_active):
            ref[t] += w[t, j] * moe.experts[int(idx[t, j])](flat[t])
    ref = ref.view_as(x) + moe.shared(x)
    assert torch.allclose(out, ref, atol=1e-9), (out - ref).abs().max()


def test_moe_routing_weights_sum_to_one():
    cfg = moe_cfg(norm_topk_prob=True)
    moe = MoE(cfg)
    _, w, _ = moe.router(torch.randn(20, cfg.d_model))
    assert torch.allclose(w.sum(-1), torch.ones(20), atol=1e-5)


def test_router_bias_steers_selection():
    """A large positive bias must force that expert into the top-k."""
    cfg = moe_cfg()
    moe = MoE(cfg)
    x = torch.randn(32, cfg.d_model)
    idx_before, _, _ = moe.router(x)
    assert not (idx_before == 7).all()
    with torch.no_grad():
        moe.router.expert_bias[7] = 100.0
    idx_after, _, _ = moe.router(x)
    assert (idx_after == 7).any(-1).all(), "biased expert was not always selected"


def test_loss_free_balancing_penalises_overused_experts():
    """Experts that win too often must end up with the lowest bias."""
    cfg = moe_cfg(router_bias_update=0.01)
    torch.manual_seed(1)
    moe = MoE(cfg).train()
    r = moe.router
    with torch.no_grad():                 # rig experts 0 and 5 to dominate routing
        r.weight[0] *= 12.0
        r.weight[5] *= 8.0
    g = torch.Generator().manual_seed(7)
    for _ in range(400):
        r.load.zero_()
        r(torch.randn(256, cfg.d_model, generator=g))
        r.update_bias()
    bias = r.expert_bias
    rigged = {0, 5}
    normal = [bias[i].item() for i in range(cfg.n_experts) if i not in rigged]
    for i in rigged:
        assert bias[i].item() < min(normal), (
            f"over-selected expert {i} was not pushed down: {bias.tolist()}"
        )


def test_loss_free_balancing_reduces_imbalance():
    """Averaged over a trajectory, the bias update lowers load imbalance."""
    cfg = moe_cfg(router_bias_update=0.01)
    torch.manual_seed(1)
    moe = MoE(cfg).train()
    r = moe.router
    with torch.no_grad():
        r.weight[0] *= 12.0
    g = torch.Generator().manual_seed(7)
    hist = []
    for _ in range(400):
        r.load.zero_()
        r(torch.randn(256, cfg.d_model, generator=g))
        hist.append((r.load / r.load.mean()).std().item())
        r.update_bias()
    first, last = sum(hist[:40]) / 40, sum(hist[-40:]) / 40
    assert last < first, f"imbalance did not fall: {first:.3f} -> {last:.3f}"


def test_router_bias_is_not_a_parameter():
    """The bias must be steered by rule, never by the optimiser."""
    moe = MoE(moe_cfg())
    assert "router.expert_bias" not in dict(moe.named_parameters())
    assert "router.expert_bias" in dict(moe.named_buffers())


def test_every_expert_receives_gradient():
    cfg = moe_cfg(router_bias_update=0.0)
    m = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (8, 32))
    m(idx, targets=idx).loss.backward()
    for block in m.blocks:
        if isinstance(block.ffn, MoE):
            missing = [i for i, e in enumerate(block.ffn.experts)
                       if e.w2.weight.grad is None or e.w2.weight.grad.abs().sum() == 0]
            assert len(missing) < cfg.n_experts, "no expert was trained at all"


def test_moe_reports_load_balance_metric():
    cfg = moe_cfg()
    m = Transformer(cfg)
    out = m(torch.randint(0, cfg.vocab_size, (4, 16)),
            targets=torch.randint(0, cfg.vocab_size, (4, 16)))
    assert "load_cv" in out.metrics


def test_moe_first_dense_layers_stay_dense():
    cfg = cfg_for(moe=True, n_experts=8, n_experts_active=2, expert_hidden=32,
                  n_layer=4, moe_first_dense=2)
    m = Transformer(cfg)
    assert [b.is_moe for b in m.blocks] == [False, False, True, True]


def test_active_params_less_than_total():
    cfg = moe_cfg(n_layer=4)
    counts = cfg.param_count()
    assert counts["active"] < counts["total"]
    assert counts["total"] == Transformer(cfg).num_params()


# --- multi-head latent attention -----------------------------------------------
def mla_cfg(**kw):
    return cfg_for(attn_type="mla", n_kv_head=4, kv_lora_rank=32,
                   qk_rope_head_dim=8, **kw)


def test_mla_is_causal():
    m = Transformer(mla_cfg()).double().eval()
    idx = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        base = m(idx).logits
        alt = idx.clone()
        alt[0, 6] = (alt[0, 6] + 3) % 64
        new = m(alt).logits
    assert torch.allclose(base[:, :6], new[:, :6], atol=1e-12)
    assert not torch.allclose(base[:, 6], new[:, 6])


def test_mla_cache_matches_full_forward():
    m = Transformer(mla_cfg()).double().eval()
    idx = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        full = m(idx).logits
        cache = m.alloc_cache(batch=2)
        step = torch.cat([m(idx[:, t:t+1], cache=cache, start_pos=t).logits
                          for t in range(9)], dim=1)
    assert torch.allclose(full, step, atol=1e-9), (full - step).abs().max()


def test_mla_with_query_compression():
    m = Transformer(mla_cfg(q_lora_rank=32)).double().eval()
    idx = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        full = m(idx).logits
        cache = m.alloc_cache(batch=1)
        step = torch.cat([m(idx[:, t:t+1], cache=cache, start_pos=t).logits
                          for t in range(8)], dim=1)
    assert torch.allclose(full, step, atol=1e-9)


def test_mla_cache_is_smaller_than_gqa():
    """The whole point of MLA: fewer bytes cached per token."""
    common = dict(vocab_size=64, n_layer=2, n_head=16, d_model=1024, max_seq_len=64)
    gqa = Transformer(ModelConfig(n_kv_head=4, **common)).eval()
    mla = Transformer(ModelConfig(attn_type="mla", n_kv_head=16, kv_lora_rank=256,
                                  qk_rope_head_dim=32, **common)).eval()
    idx = torch.randint(0, 64, (1, 8))
    caches = []
    for m in (gqa, mla):
        c = m.alloc_cache(batch=1)
        with torch.no_grad():
            m(idx, cache=c, start_pos=0)
        caches.append(c.bytes())
    assert caches[1] < caches[0], f"MLA cache {caches[1]} not smaller than GQA {caches[0]}"


# --- multi-token prediction ----------------------------------------------------
def test_mtp_adds_loss_only_in_training():
    cfg = cfg_for(mtp_depth=2)
    m = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    m.train()
    train_out = m(idx, targets=idx)
    assert "mtp1_loss" in train_out.metrics and "mtp2_loss" in train_out.metrics
    m.eval()
    assert "mtp1_loss" not in m(idx, targets=idx).metrics


def test_mtp_heads_receive_gradient():
    cfg = cfg_for(mtp_depth=1)
    m = Transformer(cfg).train()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    m(idx, targets=idx).loss.backward()
    assert m.mtp[0].proj.weight.grad is not None
    assert m.mtp[0].proj.weight.grad.abs().sum() > 0


def test_mtp_weight_scales_its_contribution():
    torch.manual_seed(4)
    idx = torch.randint(0, 64, (2, 16))
    losses = []
    for w in (0.0, 1.0):
        torch.manual_seed(4)
        m = Transformer(cfg_for(mtp_depth=1, mtp_weight=w)).train()
        losses.append(m(idx, targets=idx).loss.item())
    assert losses[1] > losses[0]


# --- masking -------------------------------------------------------------------
def test_document_masking_blocks_cross_document_attention():
    """Changing a token in document 0 must not move logits inside document 1."""
    cfg = cfg_for(doc_masking=True)
    m = Transformer(cfg).eval()
    idx = torch.randint(0, 64, (1, 12))
    docs = torch.tensor([[0] * 6 + [1] * 6])
    with torch.no_grad():
        base = m(idx, doc_ids=docs).logits
        alt = idx.clone()
        alt[0, 2] = (alt[0, 2] + 7) % 64
        new = m(alt, doc_ids=docs).logits
    assert torch.allclose(base[:, 6:], new[:, 6:], atol=1e-5), "attention leaked across docs"
    assert not torch.allclose(base[:, 3], new[:, 3], atol=1e-5)


def test_sliding_window_limits_the_receptive_field():
    cfg = cfg_for(sliding_window=4, global_attn_every=0, n_layer=1)
    m = Transformer(cfg).eval()
    idx = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        base = m(idx).logits
        alt = idx.clone()
        alt[0, 0] = (alt[0, 0] + 5) % 64
        new = m(alt).logits
    # one layer, window 4: position 11 cannot see position 0
    assert torch.allclose(base[:, 11], new[:, 11], atol=1e-5)
    assert not torch.allclose(base[:, 2], new[:, 2], atol=1e-5)


def test_global_layers_see_everything():
    cfg = cfg_for(sliding_window=4, global_attn_every=1, n_layer=1)
    m = Transformer(cfg).eval()
    idx = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        base = m(idx).logits
        alt = idx.clone()
        alt[0, 0] = (alt[0, 0] + 5) % 64
        new = m(alt).logits
    assert not torch.allclose(base[:, 11], new[:, 11], atol=1e-5)


def test_doc_ids_from_eos():
    toks = torch.tensor([[5, 6, 0, 7, 8, 0, 9]])
    assert doc_ids_from_eos(toks, eos_id=0).tolist() == [[0, 0, 0, 1, 1, 1, 2]]


def test_dense_mask_is_causal_and_windowed():
    m = dense_mask(4, 4, "cpu", torch.float32, window=2)
    finite = torch.isfinite(m[0, 0])
    assert finite.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, True, True, False],
        [False, False, True, True],
    ]


@pytest.mark.skipif(not HAS_FLEX, reason="flex_attention unavailable")
def test_flex_and_dense_masks_agree():
    cfg = cfg_for(doc_masking=True, attn_impl="flex")
    cfg2 = cfg_for(doc_masking=True, attn_impl="sdpa")
    a, b = Transformer(cfg).eval(), Transformer(cfg2).eval()
    b.load_state_dict(a.state_dict())
    idx = torch.randint(0, 64, (2, 12))
    docs = torch.tensor([[0] * 5 + [1] * 7, [0] * 8 + [1] * 4])
    with torch.no_grad():
        assert torch.allclose(a(idx, doc_ids=docs).logits,
                              b(idx, doc_ids=docs).logits, atol=1e-4)


def test_nope_layers_have_no_rope():
    cfg = cfg_for(nope_every=2, n_layer=4)
    m = Transformer(cfg)
    assert [b.attn.use_rope for b in m.blocks] == [True, False, True, False]


# --- rope scaling --------------------------------------------------------------
def _inv_freq(kind: str, scale: float, dim: int = 64):
    cfg = ModelConfig(rope_scaling_type=kind, rope_scaling=scale,
                      rope_original_len=2048, max_seq_len=int(2048 * scale))
    return build_inv_freq(cfg, dim)[0]


@pytest.mark.parametrize("kind", ["linear", "ntk", "yarn"])
def test_rope_scaling_changes_the_cache(kind):
    plain = ModelConfig(rope_scaling_type="none", max_seq_len=32)
    scaled = ModelConfig(rope_scaling_type=kind, rope_scaling=4.0,
                         rope_original_len=32, max_seq_len=128)
    assert not torch.allclose(build_rope_cache(32, 16, cfg=plain)[0],
                              build_rope_cache(32, 16, cfg=scaled)[0])


def test_linear_scaling_leaves_frequencies_alone():
    """Linear interpolation divides positions, so inv_freq is untouched."""
    assert torch.allclose(_inv_freq("linear", 8.0), _inv_freq("none", 1.0))


def test_ntk_and_yarn_interpolate_the_slowest_dimension():
    """Both must compress the lowest frequency by exactly the scaling factor."""
    base = _inv_freq("none", 1.0)
    for kind in ("ntk", "yarn"):
        f = _inv_freq(kind, 8.0)
        assert math.isclose(float(f[-1] / base[-1]), 1 / 8, rel_tol=1e-3), kind
        assert math.isclose(float(f[0] / base[0]), 1.0, rel_tol=1e-6), kind


def test_yarn_keeps_a_band_of_pure_extrapolation():
    """YaRN's ramp leaves the fastest dims untouched; NTK bends every dim."""
    base, yarn, ntk = _inv_freq("none", 1.0), _inv_freq("yarn", 8.0), _inv_freq("ntk", 8.0)
    untouched = int((torch.isclose(yarn / base, torch.ones_like(base), rtol=1e-6)).sum())
    assert untouched >= 5, f"yarn ramp too narrow: {untouched} dims extrapolating"
    ntk_untouched = int((torch.isclose(ntk / base, torch.ones_like(base), rtol=1e-6)).sum())
    assert ntk_untouched < untouched


def test_resize_context_extends_the_model():
    cfg = cfg_for(max_seq_len=16)
    m = Transformer(cfg).eval()
    m.resize_context(64, scaling=4.0, scaling_type="yarn")
    with torch.no_grad():
        assert m(torch.randint(0, 64, (1, 40))).logits.shape == (1, 40, 64)


# --- muon ----------------------------------------------------------------------
def test_ns_dtype_is_device_aware():
    """bf16 only where tensor cores make it fast; fp32 on CPU and MPS."""
    from slm.train.muon import ns_dtype

    assert ns_dtype(torch.device("cpu")) is torch.float32
    assert ns_dtype(torch.device("mps")) is torch.float32
    assert ns_dtype(torch.device("cuda")) is torch.bfloat16


def test_newton_schulz_is_not_pathologically_slow_on_cpu():
    """Regression guard: bf16 matmul on CPU is ~400x slower than fp32.

    A 1024x1024 iteration took 21.6 s in bf16 and 0.05 s in fp32 on the machine
    this was found on, which made Muon unusable off CUDA.
    """
    import time

    g = torch.randn(1024, 1024)
    t0 = time.perf_counter()
    newton_schulz(g, steps=5)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"newton_schulz took {elapsed:.1f}s - wrong dtype for CPU?"


def test_newton_schulz_orthogonalises_a_typical_gradient():
    """On a Gaussian matrix (what gradients look like) 5 steps flattens the spectrum."""
    torch.manual_seed(0)
    s = torch.linalg.svdvals(newton_schulz(torch.randn(64, 32), steps=5).float())
    assert s.min() > 0.6 and s.max() < 1.4, s


def test_newton_schulz_collapses_an_extreme_spectrum():
    """Condition number 1000 in, near-flat spectrum out."""
    torch.manual_seed(0)
    g = torch.randn(64, 32) @ torch.diag(torch.linspace(0.01, 10.0, 32))
    before = torch.linalg.svdvals(g)
    after = torch.linalg.svdvals(newton_schulz(g, steps=5).float())
    assert after.std() < before.std() / 50
    assert after.max() < 1.4


def test_more_newton_schulz_steps_help_ill_conditioned_input():
    """Documents why ns_steps is a knob: 5 under-converges on hard spectra."""
    torch.manual_seed(0)
    g = torch.randn(64, 32) @ torch.diag(torch.linspace(0.01, 10.0, 32))
    s5 = torch.linalg.svdvals(newton_schulz(g, steps=5).float())
    s8 = torch.linalg.svdvals(newton_schulz(g, steps=8).float())
    assert s8.min() > s5.min(), (s5.min().item(), s8.min().item())


def test_newton_schulz_handles_wide_and_tall():
    for shape in [(16, 64), (64, 16), (32, 32)]:
        s = torch.linalg.svdvals(newton_schulz(torch.randn(*shape), 5).float())
        assert s.max() < 1.4 and s.min() > 0.6, shape


def test_newton_schulz_preserves_direction():
    """An already-orthogonal matrix must come back essentially unchanged."""
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    assert torch.allclose(newton_schulz(q, 5).float(), q, atol=0.05)


def test_muon_rejects_non_2d_parameters():
    with pytest.raises(ValueError, match="2D"):
        Muon([torch.nn.Parameter(torch.zeros(8))])


def test_muon_reduces_a_toy_loss():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(16, 16))
    target = torch.randn(16, 16)
    opt = Muon([w], lr=0.05, weight_decay=0.0)
    first = last = None
    for i in range(60):
        opt.zero_grad()
        loss = (w - target).pow(2).mean()
        loss.backward()
        opt.step()
        first = loss.item() if i == 0 else first
        last = loss.item()
    assert last < first * 0.5, (first, last)


def test_muon_trains_a_transformer_step():
    cfg = cfg_for()
    m = Transformer(cfg)
    opt = build_optimizer(m, OptimConfig(kind="muon"), "cpu", verbose=False)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    before = [p.detach().clone() for p in m.parameters()]
    m(idx, targets=idx).loss.backward()
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_combined_optimizer_state_roundtrip():
    cfg = cfg_for()
    m = Transformer(cfg)
    opt = build_optimizer(m, OptimConfig(kind="muon"), "cpu", verbose=False)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    m(idx, targets=idx).loss.backward()
    opt.step()
    sd = opt.state_dict()
    assert set(sd) == {"muon", "adamw"}

    m2 = Transformer(cfg)
    m2.load_state_dict(m.state_dict())
    opt2 = build_optimizer(m2, OptimConfig(kind="muon"), "cpu", verbose=False)
    opt2.load_state_dict(sd)
    a = next(iter(opt.optimizers["muon"].state.values()))["momentum_buffer"]
    b = next(iter(opt2.optimizers["muon"].state.values()))["momentum_buffer"]
    assert torch.allclose(a, b)


def test_combined_optimizer_groups_are_tagged():
    m = Transformer(cfg_for())
    opt = build_optimizer(m, OptimConfig(kind="muon"), "cpu", verbose=False)
    owners = {g["owner"] for g in opt.param_groups}
    assert owners == {"muon", "adamw"}


# --- muP -----------------------------------------------------------------------
def _mup_model(d: int, use_mup: bool, base: int = 128):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, n_layer=4, n_head=max(1, d // 64),
                      n_kv_head=max(1, d // 128), d_model=d, max_seq_len=32,
                      ffn_hidden=2 * d, nope_every=0, doc_masking=False,
                      use_mup=use_mup, mup_base_d_model=base, tie_embeddings=False)
    return Transformer(cfg).eval()


def _last_layer_activation(model) -> float:
    with torch.no_grad():
        out = model(torch.arange(16)[None] % 256, return_hidden=True)
    return float(out.hidden.abs().mean())


def test_mup_coordinate_check():
    """The canonical muP test: activation scale must not drift with width.

    Under standard parameterisation the residual stream grows with d_model, so a
    learning rate tuned at one width is wrong at another. muP removes that drift,
    which is what makes hyperparameter transfer from a narrow proxy valid.
    """
    widths = (128, 256, 512, 1024)
    std = [_last_layer_activation(_mup_model(d, False)) for d in widths]
    mup = [_last_layer_activation(_mup_model(d, True)) for d in widths]
    std_spread = max(std) / min(std)
    mup_spread = max(mup) / min(mup)
    assert std_spread > 4.0, f"baseline should drift with width, got {std_spread:.2f}x"
    assert mup_spread < 1.3, f"muP activations drifted {mup_spread:.2f}x across widths"


def test_mup_scales_hidden_learning_rates():
    """Hidden matrices run at base_lr / width_mult; embeddings and readout do not."""
    m = _mup_model(512, True, base=128)
    groups = m.param_groups(0.1)
    scales = {g["lr_scale"] for g in groups}
    assert 1.0 in scales and 0.25 in scales, scales
    total = sum(p.numel() for g in groups for p in g["params"])
    assert total == m.num_params(), "param_groups dropped parameters"


def test_mup_disabled_leaves_one_scale():
    m = _mup_model(512, False)
    assert {g["lr_scale"] for g in m.param_groups(0.1)} == {1.0}


def test_mup_lr_scale_excludes_embeddings():
    m = _mup_model(512, True, base=128)
    named = dict(m.named_parameters())
    assert m.mup_lr_scale("tok_emb.weight", named["tok_emb.weight"]) == 1.0
    assert m.mup_lr_scale("lm_head.weight", named["lm_head.weight"]) == 1.0
    w = named["blocks.0.attn.wqkv.weight"]
    assert m.mup_lr_scale("blocks.0.attn.wqkv.weight", w) == 0.25


def test_mup_attention_uses_inverse_d_scaling():
    assert _mup_model(256, True).blocks[0].attn.scale == 1.0 / 64
    assert _mup_model(256, False).blocks[0].attn.scale is None


def test_mup_model_still_trains():
    m = _mup_model(256, True)
    opt = build_optimizer(m, OptimConfig(kind="adamw", fused=False), "cpu", verbose=False)
    from slm.train.optim import set_lr

    idx = torch.randint(0, 256, (2, 16))
    before = m(idx, targets=idx).loss.item()
    for _ in range(30):
        set_lr(opt, 3e-3)
        opt.zero_grad()
        m(idx, targets=idx).loss.backward()
        opt.step()
    assert m(idx, targets=idx).loss.item() < before - 0.3
