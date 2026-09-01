"""Checkpoint integrity, rotation, portability, and mirroring."""
import json
import os
import shutil

import pytest
import torch

from slm.config import CheckpointConfig, ModelConfig
from slm.model import Transformer
from slm.train.checkpoint import (
    CheckpointManager,
    LocalStore,
    clean_state_dict,
    load_into,
    make_store,
    rng_state,
    set_rng_state,
    sha256_file,
    unwrap_model,
)


def tiny_model():
    return Transformer(ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=1,
                                   d_model=32, max_seq_len=16, ffn_hidden=32))


@pytest.fixture
def mgr(tmp_path):
    return CheckpointManager(str(tmp_path / "run"), CheckpointConfig(async_save=False))


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #
def test_save_and_restore_exactly(mgr):
    m, m2 = tiny_model(), tiny_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # take a real step so the optimiser has non-trivial moment state
    idx = torch.randint(0, 64, (2, 8))
    m(idx, targets=idx).loss.backward()
    opt.step()

    path = mgr.save(step=42, model=m, optimizer=opt,
                    loader_state={"epoch": 1, "cursor": 128, "tokens_seen": 999,
                                  "seed": 1337},
                    config={"hello": "world"}, metrics={"val_loss": 3.21})
    assert os.path.isdir(path)

    ck = CheckpointManager.load(path)
    assert ck["trainer"]["step"] == 42
    assert ck["trainer"]["loader"]["cursor"] == 128
    assert ck["config"]["hello"] == "world"

    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    load_into({"model": m2, "optim": opt2}, ck)
    for (n, a), (_, b) in zip(m.named_parameters(), m2.named_parameters()):
        assert torch.equal(a, b), n
    # optimiser moments must come back too, or the first post-resume step lurches
    s1 = opt.state_dict()["state"]
    s2 = opt2.state_dict()["state"]
    assert set(s1) == set(s2)
    for k in s1:
        assert torch.allclose(s1[k]["exp_avg"], s2[k]["exp_avg"])
        assert torch.allclose(s1[k]["exp_avg_sq"], s2[k]["exp_avg_sq"])


def test_manifest_hashes_every_file(mgr):
    path = mgr.save(step=1, model=tiny_model())
    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)
    assert set(man["files"]) == {"model.pt", "trainer.pt", "config.json"}
    for fn, meta in man["files"].items():
        assert meta["sha256"] == sha256_file(os.path.join(path, fn))
        assert meta["size"] == os.path.getsize(os.path.join(path, fn))


# --------------------------------------------------------------------------- #
# integrity
# --------------------------------------------------------------------------- #
def test_corruption_is_detected(mgr):
    path = mgr.save(step=10, model=tiny_model())
    assert mgr.verify(path)
    with open(os.path.join(path, "model.pt"), "r+b") as f:
        f.seek(600)
        f.write(b"\xde\xad\xbe\xef")
    assert not mgr.verify(path), "a flipped byte slipped past verification"


def test_truncation_is_detected(mgr):
    path = mgr.save(step=10, model=tiny_model())
    with open(os.path.join(path, "model.pt"), "r+b") as f:
        f.truncate(64)
    assert not mgr.verify(path)


def test_latest_falls_back_past_a_corrupt_checkpoint(mgr):
    good = mgr.save(step=1, model=tiny_model())
    bad = mgr.save(step=2, model=tiny_model())
    with open(os.path.join(bad, "trainer.pt"), "r+b") as f:
        f.truncate(10)
    assert mgr.latest() == good, "a corrupt newest checkpoint must not be selected"


def test_partial_directory_is_invisible(mgr):
    """A .tmp left by a killed process must never be picked up."""
    tmpdir = mgr.path_for(77) + ".tmp"
    os.makedirs(tmpdir, exist_ok=True)
    with open(os.path.join(tmpdir, "model.pt"), "wb") as f:
        f.write(b"garbage")
    good = mgr.save(step=5, model=tiny_model())
    assert mgr.latest() == good
    assert [i.step for i in mgr.list_checkpoints()] == [5]


def test_latest_pointer_is_a_plain_file(mgr):
    mgr.save(step=3, model=tiny_model())
    p = os.path.join(mgr.ckpt_dir, "LATEST")
    assert os.path.isfile(p) and not os.path.islink(p)
    with open(p) as f:
        assert f.read().strip() == "step_00000003"


# --------------------------------------------------------------------------- #
# rotation
# --------------------------------------------------------------------------- #
def test_rotation_keeps_last_n_and_all_milestones(tmp_path):
    cfg = CheckpointConfig(keep_last=2, permanent_every=5, async_save=False)
    mgr = CheckpointManager(str(tmp_path / "r"), cfg)
    for step in range(1, 13):
        mgr.save(step=step, model=tiny_model())
    kept = sorted(i.step for i in mgr.list_checkpoints())
    assert 5 in kept and 10 in kept, "milestone checkpoints were rotated away"
    assert 11 in kept and 12 in kept, "the two newest must survive"
    assert 1 not in kept and 3 not in kept


def test_explicit_permanent_survives_rotation(tmp_path):
    cfg = CheckpointConfig(keep_last=1, permanent_every=0, async_save=False)
    mgr = CheckpointManager(str(tmp_path / "r"), cfg)
    mgr.save(step=1, model=tiny_model(), permanent=True)
    for s in range(2, 8):
        mgr.save(step=s, model=tiny_model())
    kept = sorted(i.step for i in mgr.list_checkpoints())
    assert kept == [1, 7]


# --------------------------------------------------------------------------- #
# portability
# --------------------------------------------------------------------------- #
def test_wrapper_prefixes_are_stripped():
    sd = {"_orig_mod.module.blocks.0.attn.wo.weight": torch.zeros(2),
          "module._orig_mod.norm.weight": torch.zeros(2),
          "tok_emb.weight": torch.zeros(2)}
    assert set(clean_state_dict(sd)) == {
        "blocks.0.attn.wo.weight", "norm.weight", "tok_emb.weight"
    }


def test_unwrap_model_finds_the_real_module():
    m = tiny_model()

    class Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    assert unwrap_model(Wrap(Wrap(m))) is m


def test_saved_weights_are_device_agnostic(mgr):
    """Anything CUDA-specific in the payload would break a cross-provider move."""
    path = mgr.save(step=1, model=tiny_model())
    sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu",
                    weights_only=False)
    assert all(v.device.type == "cpu" for v in sd.values())


def test_inference_export_drops_the_optimiser(mgr, tmp_path):
    m = tiny_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    path = mgr.save(step=9, model=m, optimizer=opt, config={"model": {"n_layer": 2}})
    dest = str(tmp_path / "model_fp16.pt")
    CheckpointManager.export_inference(path, dest, dtype=torch.float16)
    blob = torch.load(dest, map_location="cpu", weights_only=False)
    assert blob["format"] == "slm-inference-v1"
    assert all(v.dtype == torch.float16 for v in blob["model"].values()
               if v.is_floating_point())
    assert set(blob) == {"model", "config", "format"}
    assert "optim" not in blob


# --------------------------------------------------------------------------- #
# RNG
# --------------------------------------------------------------------------- #
def test_rng_state_roundtrip():
    torch.manual_seed(0)
    state = rng_state()
    a = torch.randn(5)
    set_rng_state(state)
    assert torch.equal(a, torch.randn(5))


def test_rng_state_survives_serialisation(tmp_path):
    torch.manual_seed(3)
    p = str(tmp_path / "rng.pt")
    torch.save(rng_state(), p)
    a = torch.randn(4)
    set_rng_state(torch.load(p, map_location="cpu", weights_only=False))
    assert torch.equal(a, torch.randn(4))


# --------------------------------------------------------------------------- #
# async
# --------------------------------------------------------------------------- #
def test_async_save_lands(tmp_path):
    mgr = CheckpointManager(str(tmp_path / "r"), CheckpointConfig(async_save=True))
    m = tiny_model()
    path = mgr.save(step=7, model=m)
    mgr.wait()
    assert mgr.verify(path)
    sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu",
                    weights_only=False)
    assert torch.equal(sd["tok_emb.weight"], m.tok_emb.weight)


def test_async_snapshot_is_isolated_from_later_updates(tmp_path):
    """Weights mutated right after save() must not leak into the saved file."""
    mgr = CheckpointManager(str(tmp_path / "r"), CheckpointConfig(async_save=True))
    m = tiny_model()
    before = m.tok_emb.weight.detach().clone()
    mgr.save(step=1, model=m)
    with torch.no_grad():
        m.tok_emb.weight.add_(100.0)
    mgr.wait()
    sd = torch.load(os.path.join(mgr.path_for(1), "model.pt"), map_location="cpu",
                    weights_only=False)
    assert torch.equal(sd["tok_emb.weight"], before)


# --------------------------------------------------------------------------- #
# remote mirror
# --------------------------------------------------------------------------- #
def test_local_store_push_pull(tmp_path):
    store = LocalStore(str(tmp_path / "mirror"))
    mgr = CheckpointManager(str(tmp_path / "run"),
                            CheckpointConfig(async_save=False), store=store)
    m = tiny_model()
    mgr.save(step=4, model=m)
    assert store.list() == ["step_00000004"]

    # simulate the machine dying: wipe local state entirely
    shutil.rmtree(mgr.ckpt_dir)
    revived = CheckpointManager(str(tmp_path / "run2"),
                                CheckpointConfig(async_save=False), store=store)
    path = revived.fetch_remote()
    assert path is not None and revived.verify(path)
    m2 = tiny_model()
    load_into({"model": m2}, CheckpointManager.load(path))
    assert torch.equal(m.tok_emb.weight, m2.tok_emb.weight)


def test_fetch_remote_rejects_a_corrupt_mirror(tmp_path):
    store = LocalStore(str(tmp_path / "mirror"))
    mgr = CheckpointManager(str(tmp_path / "run"),
                            CheckpointConfig(async_save=False), store=store)
    mgr.save(step=2, model=tiny_model())
    with open(os.path.join(store.root, "step_00000002", "model.pt"), "r+b") as f:
        f.truncate(20)
    shutil.rmtree(mgr.ckpt_dir)
    assert CheckpointManager(str(tmp_path / "run3"),
                             CheckpointConfig(async_save=False),
                             store=store).fetch_remote() is None


def test_store_uri_dispatch(tmp_path):
    assert make_store("") is None
    assert isinstance(make_store(str(tmp_path / "x")), LocalStore)
    assert isinstance(make_store("file://" + str(tmp_path / "y")), LocalStore)


def test_step_name_roundtrip():
    assert CheckpointManager.step_of(CheckpointManager.name_for(12345)) == 12345
    assert CheckpointManager.name_for(7) == "step_00000007"
