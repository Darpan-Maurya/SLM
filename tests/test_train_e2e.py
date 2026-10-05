"""End-to-end training tests, including the interrupt/resume guarantee."""
import json
import os

import numpy as np
import pytest
import torch

from slm.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from slm.data import ShardWriter, write_index
from slm.train.distributed import DistInfo
from slm.train.trainer import FINISH_FILE, Trainer


def make_corpus(root, name, n_tokens=40_000, vocab=256):
    d = root / name
    if (d / "index.json").exists():          # already built (e.g. by a parent process)
        return str(d)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # a learnable pattern, not noise: loss must actually fall for these tests to mean anything
    base = rng.integers(0, vocab // 4, size=97)
    toks = np.tile(base, n_tokens // len(base) + 1)[:n_tokens]
    with ShardWriter(str(d / "shard_00000.bin"), vocab) as w:
        w.write(toks)
    write_index(str(d), [{"name": "shard_00000.bin", "n_tokens": n_tokens,
                          "n_docs": 1, "dtype": "uint16"}], vocab,
                {"eos_id": 0, "bytes_per_token": 3.0})
    return str(d)


def make_cfg(tmp_path, steps, seed=1234, **train_kw):
    train_dir = make_corpus(tmp_path, "train")
    val_dir = make_corpus(tmp_path, "val", n_tokens=8_000)
    return Config(
        model=ModelConfig(vocab_size=256, n_layer=2, n_head=2, n_kv_head=1,
                          d_model=64, max_seq_len=64, ffn_hidden=96,
                          nope_every=0, doc_masking=False, mtp_depth=0),
        data=DataConfig(train_dir=train_dir, val_dir=val_dir, seed=seed,
                        num_workers=0, eos_id=0),
        optim=OptimConfig(kind="adamw", lr=3e-3, warmup_steps=3, schedule="wsd",
                          decay_steps=5, fused=False, zloss=0.0),
        train=TrainConfig(run_name="t", out_dir=str(tmp_path / "runs"), seed=seed,
                          max_steps=steps, global_batch_tokens=512,
                          micro_batch_size=4, dtype="float32", compile=False,
                          log_every=1, eval_every=0, **train_kw),
        ckpt=CheckpointConfig(interval=0, async_save=False, keep_last=5,
                              permanent_every=0),
    )


def run(cfg) -> Trainer:
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    t.train()
    return t


class InterruptAfter(Trainer):
    """Trainer that fires the real preemption path after N steps."""

    stop_at = 0

    def train_step(self, index):
        metrics = super().train_step(index)
        if index + 1 >= self.stop_at:
            self.guard.trigger("test-interrupt")
        return metrics


def run_interrupted(cfg, stop_at: int) -> Trainer:
    t = InterruptAfter(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    t.stop_at = stop_at
    t.train()
    return t


def losses_from(run_dir) -> dict[int, float]:
    path = os.path.join(run_dir, "logs", "metrics.jsonl")
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "train/loss" in rec:
                out[rec["step"]] = rec["train/loss"]
    return out


# --- the core guarantee --------------------------------------------------------
def test_training_resumes_with_an_identical_loss_trajectory(tmp_path):
    """Kill at step 12, resume, and steps 12-23 must reproduce the unbroken run."""
    ref_cfg = make_cfg(tmp_path / "ref", 24)
    ref = run(ref_cfg)
    reference = losses_from(ref.run_dir)
    assert len(reference) == 24
    assert reference[23] < reference[0] - 0.3, "reference run did not learn"

    # the interrupted run must target the SAME 24 steps: the WSD schedule is a
    # function of max_steps, so a 12-step run is a genuinely different experiment
    root = tmp_path / "split"
    first = make_cfg(root, 24)
    run_interrupted(first, stop_at=12)

    second = make_cfg(root, 24)
    second.train.resume = "auto"
    resumed = run(second)
    after = losses_from(resumed.run_dir)

    for step in range(12, 24):
        assert step in after, f"resumed run never reached step {step}"
        assert abs(after[step] - reference[step]) < 1e-4, (
            f"step {step}: resumed {after[step]:.6f} vs reference {reference[step]:.6f}"
        )


def test_resume_restores_the_token_counter(tmp_path):
    cfg = make_cfg(tmp_path, 16)
    t = run_interrupted(cfg, stop_at=8)
    tokens = t.state.tokens
    cfg2 = make_cfg(tmp_path, 16)
    cfg2.train.resume = "auto"
    t2 = run(cfg2)
    assert t2.state.tokens == tokens * 2


def test_resume_warns_when_the_schedule_changes(tmp_path, capsys):
    """Changing max_steps silently rewrites the LR curve; the run must say so."""
    run_interrupted(make_cfg(tmp_path, 20), stop_at=6)
    cfg2 = make_cfg(tmp_path, 40)
    cfg2.train.resume = "auto"
    t = Trainer(cfg2, dist_info=DistInfo(device=torch.device("cpu")))
    t.resume()
    t.close()
    assert "max_steps changed 20 -> 40" in capsys.readouterr().out


def test_resume_picks_up_optimizer_moments(tmp_path):
    """Without optimiser state the first post-resume step would jump."""
    cfg = make_cfg(tmp_path, 10)
    run(cfg)
    cfg2 = make_cfg(tmp_path, 11)
    cfg2.train.resume = "auto"
    t2 = Trainer(cfg2, dist_info=DistInfo(device=torch.device("cpu")))
    t2.resume()
    state = t2.optimizer.state_dict()["state"]
    assert state, "optimiser came back empty"
    assert any(v["exp_avg"].abs().sum() > 0 for v in state.values())
    t2.close()


# --- graceful finish -----------------------------------------------------------
def test_finish_file_triggers_the_decay_and_stops(tmp_path):
    """Dropping FINISH must re-point max_steps so the run ends properly annealed."""
    cfg = make_cfg(tmp_path, 500)
    cfg.optim.decay_steps = 4
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    os.makedirs(t.run_dir, exist_ok=True)
    with open(os.path.join(t.run_dir, FINISH_FILE), "w") as f:
        f.write("4")
    t.train()
    assert t.max_steps == 4, t.max_steps
    seen = losses_from(t.run_dir)
    assert max(seen) < 10, "run did not stop early"


def test_stop_file_preempts_cleanly(tmp_path):
    cfg = make_cfg(tmp_path, 200)
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    os.makedirs(t.run_dir, exist_ok=True)
    t.guard.trigger("test")
    code = t.train()
    assert code == cfg.train.preempt_exit_code
    assert t.ckpt.latest() is not None, "a preempted run must leave a checkpoint"


# --- learning ------------------------------------------------------------------
def test_model_overfits_a_repeating_pattern(tmp_path):
    """A working trainer must drive loss well below ln(V) on a memorisable corpus."""
    cfg = make_cfg(tmp_path, 60)
    cfg.optim.lr = 6e-3
    t = run(cfg)
    seen = losses_from(t.run_dir)
    assert seen[59] < 1.5, f"final loss {seen[59]:.3f} - trainer is not learning"


def test_muon_also_trains(tmp_path):
    cfg = make_cfg(tmp_path, 40)
    cfg.optim.kind = "muon"
    cfg.optim.muon_lr = 0.02
    t = run(cfg)
    seen = losses_from(t.run_dir)
    assert seen[39] < seen[0] - 1.0, f"{seen[0]:.3f} -> {seen[39]:.3f}"


def test_moe_trains_end_to_end(tmp_path):
    cfg = make_cfg(tmp_path, 30)
    cfg.model = ModelConfig(vocab_size=256, n_layer=2, n_head=2, n_kv_head=1,
                            d_model=64, max_seq_len=64, ffn_hidden=96,
                            nope_every=0, doc_masking=False, moe=True, n_experts=4,
                            n_experts_active=2, expert_hidden=32, moe_first_dense=0)
    t = run(cfg)
    seen = losses_from(t.run_dir)
    assert seen[29] < seen[0] - 0.5


@pytest.mark.parametrize("moe", [False, True])
def test_evaluation_matches_plain_inference_and_leaves_training_untouched(tmp_path, moe):
    cfg = make_cfg(tmp_path, 4)
    if moe:
        cfg.model = ModelConfig(vocab_size=256, n_layer=2, n_head=2, n_kv_head=1,
                                d_model=64, max_seq_len=64, ffn_hidden=96,
                                nope_every=0, doc_masking=False, moe=True, n_experts=4,
                                n_experts_active=2, expert_hidden=32, moe_first_dense=0)
    t = run(cfg)
    params = [p.detach().clone() for p in t.raw_model.parameters()]
    grads = [None if p.grad is None else p.grad.clone() for p in t.raw_model.parameters()]
    loads = [b.clone() for name, b in t.raw_model.named_buffers() if name.endswith("load")]
    assert len(loads) == (2 if moe else 0)
    start = t.val_loader.state_dict()
    res = t.evaluate(steps=3)
    assert t.raw_model.training
    assert all(torch.equal(a, b) for a, b in zip(params, t.raw_model.parameters()))
    assert all((a is None and b.grad is None) or torch.equal(a, b.grad)
               for a, b in zip(grads, t.raw_model.parameters()))
    after = [b for name, b in t.raw_model.named_buffers() if name.endswith("load")]
    assert all(torch.equal(a, b) for a, b in zip(loads, after))   # MoE balancing ignores eval

    t.val_loader.load_state_dict(start)
    t.raw_model.eval()
    with torch.no_grad():
        ref = [float(t.raw_model(*t.val_loader.next_batch()).metrics["ce"]) for _ in range(3)]
    assert res["loss"] == pytest.approx(sum(ref) / 3, rel=1e-5)
    t.close()


def test_grad_accumulation_matches_the_token_budget(tmp_path):
    cfg = make_cfg(tmp_path, 2)
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    assert t.accum * cfg.train.micro_batch_size * cfg.data.seq_len == t.tokens_per_step
    t.close()


def test_checkpoint_carries_the_config(tmp_path):
    cfg = make_cfg(tmp_path, 4)
    t = run(cfg)
    path = t.ckpt.latest()
    with open(os.path.join(path, "config.json")) as f:
        saved = json.load(f)
    assert saved["model"]["d_model"] == cfg.model.d_model
    assert saved["optim"]["kind"] == cfg.optim.kind


# --- data mixture and curriculum ------------------------------------------------
def make_cfg_mixture(tmp_path, steps, mixture=None, curriculum=None):
    """Corpus laid out as train/<domain>/ subdirectories."""
    root = tmp_path / "mix"
    for name in ("web", "code"):
        make_corpus(root / "train", name, n_tokens=20_000)
    make_corpus(root, "val", n_tokens=8_000)
    cfg = make_cfg(tmp_path / "base", steps)
    cfg.data.train_dir = str(root / "train")
    cfg.data.val_dir = str(root / "val")
    cfg.data.mixture = mixture or {"web": 0.75, "code": 0.25}
    cfg.data.curriculum = curriculum or []
    cfg.train.out_dir = str(tmp_path / "runs-mix")
    return cfg


def test_mixture_loader_is_used_and_hits_its_ratios(tmp_path):
    cfg = make_cfg_mixture(tmp_path, 6)
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    from slm.data import MixtureLoader

    assert isinstance(t.train_loader, MixtureLoader)
    got = t.train_loader.realised_weights(20_000)
    assert abs(got["web"] - 0.75) < 0.01 and abs(got["code"] - 0.25) < 0.01
    t.close()


def test_mixture_run_trains(tmp_path):
    cfg = make_cfg_mixture(tmp_path, 20)
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    assert t.train() == 0
    seen = losses_from(t.run_dir)
    assert seen[19] < seen[0] - 0.3


def test_curriculum_switches_the_mixture_mid_run(tmp_path, capsys):
    """Stage boundaries must actually change the sampling weights."""
    curriculum = [
        {"until": 0.5, "mixture": {"web": 0.9, "code": 0.1}},
        {"until": 1.0, "mixture": {"web": 0.3, "code": 0.7}},
    ]
    cfg = make_cfg_mixture(tmp_path, 10, curriculum=curriculum)
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    assert t._active_mixture == {"web": 0.9, "code": 0.1}
    t.train()
    assert t._active_mixture == {"web": 0.3, "code": 0.7}
    assert "curriculum stage change" in capsys.readouterr().out


def test_curriculum_stage_lookup_is_monotone(tmp_path):
    curriculum = [
        {"until": 0.7, "mixture": {"web": 1.0, "code": 0.0}},
        {"until": 0.9, "mixture": {"web": 0.5, "code": 0.5}},
        {"until": 1.0, "mixture": {"web": 0.2, "code": 0.8}},
    ]
    cfg = make_cfg_mixture(tmp_path, 100, curriculum=curriculum)
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    assert t._stage_mixture(0)["web"] == 1.0
    assert t._stage_mixture(69)["web"] == 1.0
    assert t._stage_mixture(70)["web"] == 0.5
    assert t._stage_mixture(95)["web"] == 0.2
    assert t._stage_mixture(200)["web"] == 0.2      # past the end clamps
    t.close()


# --- fp16 loss scaling ----------------------------------------------------------
@pytest.mark.parametrize("kind", ["adamw", "muon"])
def test_fp16_loss_scaling_trains_and_checkpoints_the_scale(tmp_path, kind):
    cfg = make_cfg(tmp_path, 30)
    cfg.train.dtype = "float16"
    cfg.optim.kind = kind
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    assert t.scaler.is_enabled()
    t.train()
    seen = losses_from(t.run_dir)
    assert seen[29] < seen[0] - 0.5
    from slm.train.checkpoint import CheckpointManager

    extra = CheckpointManager.load(t.ckpt.latest())["trainer"]["extra"]
    assert extra["scaler"]["scale"] > 0


def test_fp16_overflow_skips_the_step_and_halves_the_scale(tmp_path):
    cfg = make_cfg(tmp_path, 10)
    cfg.train.dtype = "float16"
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    before = [p.detach().clone() for p in t.raw_model.parameters()]
    scale = t.scaler.get_scale()
    hook = next(t.raw_model.parameters()).register_hook(lambda g: g * float("inf"))
    metrics = t.train_step(0)
    hook.remove()
    assert metrics["skipped"] == 1.0
    assert t.scaler.get_scale() == scale / 2
    assert all(torch.equal(a, b) for a, b in zip(before, t.raw_model.parameters()))
    assert t.train_step(1)["skipped"] == 0.0
    t.close()


def test_bf16_and_fp32_runs_have_no_scaler(tmp_path):
    t = Trainer(make_cfg(tmp_path, 2), dist_info=DistInfo(device=torch.device("cpu")))
    assert not t.scaler.is_enabled()
    t.close()


# --- time budget ----------------------------------------------------------------
def test_plan_max_steps_fits_the_remaining_time():
    from slm.train.trainer import plan_max_steps

    decay = lambda m: max(1, int(0.2 * m))
    # 100 s left at 1 s/step from step 10 -> 110
    assert plan_max_steps(10, elapsed=0, budget=100, margin=0, sec_per_step=1.0,
                          decay_len=decay) == 110
    assert plan_max_steps(10, 50, 100, 10, 2.0, decay) == 30


def test_plan_max_steps_never_starts_the_decay_in_the_past():
    from slm.train.trainer import plan_max_steps

    decay = lambda m: max(1, int(0.2 * m))
    new_max = plan_max_steps(1000, elapsed=500, budget=100, margin=0, sec_per_step=1.0,
                             decay_len=decay)
    assert new_max - decay(new_max) >= 1000


def test_time_budget_fits_the_run_to_the_clock(tmp_path):
    """Without the budget this run would take 100k steps; it must end annealed near 3 s."""
    import time as _time

    from slm.train.optim import lr_at

    cfg = make_cfg(tmp_path, 100_000)
    cfg.optim.warmup_steps = 3
    cfg.optim.decay_steps = 0
    cfg.train.time_budget_sec = 3
    cfg.train.budget_margin_sec = 0
    cfg.train.budget_calibrate_steps = 5
    cfg.train.budget_recalibrate_every = 20
    cfg.train.max_runtime_sec = 60             # safety net so a bug cannot hang the suite
    cfg.train.log_every = 1000
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    t0 = _time.monotonic()
    t.train()
    elapsed = _time.monotonic() - t0
    assert os.path.exists(os.path.join(t.run_dir, "DONE")), "run was cut off, not finished"
    assert 10 < t.max_steps < 100_000
    assert elapsed < 12, f"budget of 3 s overran to {elapsed:.1f} s"
    final_lr = lr_at(t.max_steps - 1, cfg.optim, t.max_steps)
    assert final_lr < 0.3 * cfg.optim.lr, "the run did not end in its decay phase"


def test_time_budget_never_exceeds_the_token_budget(tmp_path):
    cfg = make_cfg(tmp_path, 40)
    cfg.train.time_budget_sec = 3600
    cfg.train.budget_calibrate_steps = 5
    t = run(cfg)
    assert t.max_steps == 40


def test_time_budget_requires_wsd(tmp_path):
    cfg = make_cfg(tmp_path, 10)
    cfg.train.time_budget_sec = 60
    cfg.optim.schedule = "cosine"
    with pytest.raises(ValueError, match="wsd"):
        Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))


def test_resume_keeps_a_budget_fitted_schedule(tmp_path):
    cfg = make_cfg(tmp_path, 50)
    cfg.train.time_budget_sec = 3600
    t = run_interrupted(cfg, stop_at=10)
    t.max_steps = 37                           # pretend the budget had refitted it
    t.save(blocking=True)
    cfg2 = make_cfg(tmp_path, 50)
    cfg2.train.time_budget_sec = 3600
    cfg2.train.resume = "auto"
    t2 = Trainer(cfg2, dist_info=DistInfo(device=torch.device("cpu")))
    t2.resume()
    assert t2.max_steps == 37 and t2._budget_locked
    t2.close()
