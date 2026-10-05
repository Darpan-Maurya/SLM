"""Two real processes over gloo on CPU: the control flow a 2x T4 run depends on."""
import json
import os
import socket
import time

import pytest
import torch
import torch.multiprocessing as mp
from test_train_e2e import make_cfg


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, root, mode, out_dir):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    from pathlib import Path

    from slm.train.distributed import setup_distributed
    from slm.train.trainer import Trainer

    torch.set_num_threads(1)
    cfg = make_cfg(Path(root), 24)
    if mode == "plain":
        cfg.train.eval_every, cfg.train.eval_steps = 5, 2     # eval must not upset DDP's sync
    if mode == "budget":
        cfg.train.max_steps = 100_000
        cfg.optim.decay_steps = 0
        cfg.train.time_budget_sec = 4
        cfg.train.budget_margin_sec = 0
        cfg.train.budget_calibrate_steps = 4
        cfg.train.budget_recalibrate_every = 10
        cfg.train.max_runtime_sec = 60

    class OneRankStops(Trainer):
        """Only rank 1 is told to stop, like a signal delivered to one process."""

        def train_step(self, index):
            metrics = super().train_step(index)
            if mode == "stop" and rank == 1 and index + 1 >= 6:
                self.guard.trigger("test: rank 1 only")
            return metrics

    t = OneRankStops(cfg, dist_info=setup_distributed("cpu"))
    code = t.train()
    checksum = float(sum(p.detach().double().sum() for p in t.raw_model.parameters()))
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump({"step": t.state.step, "max_steps": t.max_steps, "code": code,
                   "checksum": checksum}, f)


def _launch(tmp_path, mode, timeout=120.0):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    make_cfg(tmp_path, 1)                    # builds the corpus once, before spawning
    ctx = mp.start_processes(_worker, args=(2, _free_port(), str(tmp_path), mode, str(out_dir)),
                             nprocs=2, join=False, start_method="spawn")
    deadline = time.monotonic() + timeout
    while not ctx.join(timeout=1.0):
        if time.monotonic() > deadline:
            for p in ctx.processes:
                p.kill()
            pytest.fail(f"distributed run ({mode}) hung past {timeout:.0f} s")
    return [json.loads((out_dir / f"rank{r}.json").read_text()) for r in range(2)]


def test_ddp_ranks_train_in_lockstep(tmp_path):
    r0, r1 = _launch(tmp_path, "plain")
    assert r0["step"] == r1["step"] == 24
    assert r0["checksum"] == pytest.approx(r1["checksum"], rel=1e-9), "replicas diverged"


def test_a_stop_on_one_rank_stops_every_rank(tmp_path):
    """Without the synced decision, rank 0 would block forever in the next all-reduce."""
    r0, r1 = _launch(tmp_path, "stop")
    assert r0["step"] == r1["step"] == 6


def test_budget_refit_is_identical_on_every_rank(tmp_path):
    r0, r1 = _launch(tmp_path, "budget")
    assert r0["max_steps"] == r1["max_steps"] < 100_000
    assert r0["step"] == r1["step"] == r0["max_steps"]
