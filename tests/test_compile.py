"""torch.compile is the path every GPU run takes; eager-only tests never exercise it."""
import pytest
import torch
from torch._dynamo.utils import counters

from slm.config import ModelConfig
from slm.model import Transformer


def cfg_for(**kw) -> ModelConfig:
    base = dict(vocab_size=256, n_layer=2, n_head=4, n_kv_head=2, d_model=64, max_seq_len=64,
                ffn_hidden=128, nope_every=0, doc_masking=True, attn_impl="sdpa")
    base.update(kw)
    return ModelConfig(**base)


def batch(seed: int, eos: int = 0, high: int = 256):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, high, (2, 64), generator=g)
    x[:, torch.randint(0, 64, (3,), generator=g)] = eos    # documents change every batch
    return x, torch.roll(x, -1, 1)


@pytest.mark.parametrize("kw", [
    dict(),                                               # zloss off: the case inductor broke
    dict(doc_masking=False),
    dict(moe=True, n_experts=4, n_experts_active=2, expert_hidden=32, moe_first_dense=0),
])
def test_compiled_loss_keeps_its_gradient(kw):
    torch._dynamo.reset()
    m = Transformer(cfg_for(**kw))
    x, y = batch(0)
    out = torch.compile(m)(x, targets=y, doc_ids=m.doc_ids(x, 0))
    assert out.loss.requires_grad and out.loss.grad_fn is not None


def test_compiled_gradients_match_eager():
    torch._dynamo.reset()
    torch.manual_seed(0)
    a, b = Transformer(cfg_for()), Transformer(cfg_for())
    b.load_state_dict(a.state_dict())
    x, y = batch(1)
    for model, run in ((a, torch.compile(a)), (b, b)):
        run(x, targets=y, doc_ids=model.doc_ids(x, 0)).loss.backward()
    worst = max((pa.grad - pb.grad).abs().max().item()
                for pa, pb in zip(a.parameters(), b.parameters()))
    assert worst < 1e-4, worst


def test_evaluation_reuses_the_training_graph():
    """A no_grad/eval() call compiles a second graph, which crashed on an A100; the trainer's doesn't."""
    torch._dynamo.reset()
    counters.clear()
    m = Transformer(cfg_for())
    cm = torch.compile(m)
    x, y = batch(0)
    cm(x, targets=y, doc_ids=m.doc_ids(x, 0), zloss=1e-4).loss.backward()
    x, y = batch(1)
    cm(x, targets=y, doc_ids=m.doc_ids(x, 0), zloss=1e-4)
    assert counters["stats"]["unique_graphs"] == 1
    m.eval()
    with torch.no_grad():
        cm(x, targets=y, doc_ids=m.doc_ids(x, 0))
    assert counters["stats"]["unique_graphs"] == 2


def test_compiled_training_does_not_recompile_as_documents_change():
    torch._dynamo.reset()
    counters.clear()
    m = Transformer(cfg_for())
    cm = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for step in range(6):
        x, y = batch(step, high=9)          # a learnable distribution, so the loss must fall
        opt.zero_grad()
        out = cm(x, targets=y, doc_ids=m.doc_ids(x, 0))
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    assert sum(counters["recompiles"].values()) == 0, dict(counters["recompiles"])
    assert losses[-1] < losses[0]
