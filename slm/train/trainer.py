"""The training loop."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import torch

from slm.config import Config
from slm.data import MixtureLoader, ResumableLoader, TokenDataset, read_index
from slm.model import Transformer
from slm.train.checkpoint import CheckpointManager, load_into, unwrap_model
from slm.train.distributed import (
    DistInfo,
    all_reduce_mean,
    barrier,
    cleanup_distributed,
    setup_distributed,
    wrap_model,
)
from slm.train.optim import build_optimizer, clip_grad_norm, lr_at, set_lr
from slm.train.preempt import PreemptionGuard
from slm.train.telemetry import Telemetry, format_params

FINISH_FILE = "FINISH"


@dataclass
class TrainState:
    """Mutable run position; `step` counts completed optimiser steps."""

    step: int = 0
    tokens: int = 0
    best_val: float = float("inf")


class Trainer:
    """Owns the model, data, optimiser, checkpoints and the step loop."""

    def __init__(self, cfg: Config, dist_info: DistInfo | None = None):
        self.cfg = cfg
        self.dist = dist_info or setup_distributed()
        self.device = self.dist.device
        self.state = TrainState()
        self.run_dir = os.path.join(cfg.train.out_dir, cfg.train.run_name)
        self._seed_everything(cfg.train.seed + self.dist.rank)

        self.accum = self._grad_accum()
        self.model = self._build_model()
        self.raw_model = unwrap_model(self.model)
        self.optimizer = build_optimizer(
            self.raw_model, cfg.optim, self.device.type, verbose=self.dist.is_main
        )
        self.max_steps = cfg.train.max_steps
        self._finished_early = False
        self._active_mixture = self._stage_mixture(0)
        self.teacher = self._load_teacher()
        self.train_loader, self.val_loader = self._build_loaders()
        self.ckpt = CheckpointManager(self.run_dir, cfg.ckpt, rank=self.dist.rank)
        self.guard = PreemptionGuard(
            max_runtime_sec=cfg.train.max_runtime_sec,
            poll_cloud=cfg.train.poll_cloud_preemption,
            stop_file=os.path.join(self.run_dir, "STOP"),
        )
        self.telemetry = Telemetry(
            self.run_dir, cfg.model.flops_per_token(), self.device,
            is_main=self.dist.is_main, wandb_project=cfg.train.wandb_project,
            wandb_entity=cfg.train.wandb_entity, config=cfg.to_dict(),
            run_name=cfg.train.run_name,
        )
        if self.dist.is_main:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                f.write(cfg.to_json())

    # --- setup ---------------------------------------------------------------
    def _seed_everything(self, seed: int) -> None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision(self.cfg.train.matmul_precision)

    def _grad_accum(self) -> int:
        """Micro-steps per optimiser step so every job hits the same global batch."""
        cfg = self.cfg
        per_micro = cfg.train.micro_batch_size * cfg.data.seq_len * self.dist.world_size
        accum = max(1, round(cfg.train.global_batch_tokens / per_micro))
        actual = accum * per_micro
        if actual != cfg.train.global_batch_tokens and self.dist.is_main:
            print(
                f"[train] global batch rounded {cfg.train.global_batch_tokens:,} -> "
                f"{actual:,} tokens ({accum} x {per_micro:,})"
            )
        self.tokens_per_step = actual
        return accum

    def _build_model(self) -> torch.nn.Module:
        cfg = self.cfg
        model = Transformer(cfg.model)
        model.set_grad_checkpoint(cfg.train.grad_checkpoint)
        model.to(self.device)
        if self.dist.is_main:
            counts = cfg.model.param_count()
            print(
                f"[model] {format_params(counts['total'])} params "
                f"({format_params(counts['active'])} active, "
                f"{format_params(counts['non_embedding'])} non-embedding) | "
                f"{cfg.model.flops_per_token()/1e9:.2f} GFLOP/token"
            )
        if cfg.train.compile and self.device.type == "cuda":
            model = torch.compile(model)
        return wrap_model(model, cfg.train.strategy, self.dist, cfg)

    def _build_loaders(self):
        if self.cfg.finetune.mode == "sft":
            return self._build_sft_loaders()
        cfg = self.cfg
        if cfg.data.mixture or cfg.data.curriculum:
            return self._build_mixture_loaders()
        train_ds = TokenDataset(cfg.data.train_dir, cfg.data.seq_len)
        if cfg.data.eos_id < 0:
            try:
                cfg.data.eos_id = read_index(cfg.data.train_dir).get("eos_id", -1)
            except FileNotFoundError:
                pass
        common = dict(
            micro_batch_size=cfg.train.micro_batch_size, rank=self.dist.rank,
            world_size=self.dist.world_size, seed=cfg.data.seed,
            num_workers=cfg.data.num_workers, prefetch=cfg.data.prefetch,
            device=self.device,
        )
        train = ResumableLoader(train_ds, **common)
        val = None
        if os.path.isdir(cfg.data.val_dir):
            val_ds = TokenDataset(cfg.data.val_dir, cfg.data.seq_len)
            val = ResumableLoader(val_ds, **{**common, "num_workers": 0})
        if self.dist.is_main:
            print(f"[data] train {train_ds} | accum {self.accum} | "
                  f"{self.tokens_per_step:,} tokens/step")
            if not train.state_dict()["exact"]:
                print("[data] WARNING num_workers>1: resume may replay a few batches")
        return train, val

    def _stage_mixture(self, step: int) -> dict[str, float]:
        """Mixture weights for this step, following the curriculum if one is set."""
        cfg = self.cfg.data
        if not cfg.curriculum:
            return cfg.mixture
        progress = step / max(self.max_steps, 1)
        for stage in cfg.curriculum:
            if progress < float(stage.get("until", 1.0)):
                return stage["mixture"]
        return cfg.curriculum[-1]["mixture"]

    def _build_mixture_loaders(self):
        """One sub-loader per corpus directory, interleaved at fixed token ratios."""
        cfg = self.cfg
        weights = self._stage_mixture(0)
        names = sorted(weights)
        common = dict(
            micro_batch_size=cfg.train.micro_batch_size, rank=self.dist.rank,
            world_size=self.dist.world_size, seed=cfg.data.seed,
            num_workers=cfg.data.num_workers, prefetch=cfg.data.prefetch,
            device=self.device,
        )
        loaders = {}
        for name in names:
            path = os.path.join(cfg.data.train_dir, name)
            loaders[name] = ResumableLoader(
                TokenDataset(path, cfg.data.seq_len), **common
            )
        if cfg.data.eos_id < 0:
            try:
                cfg.data.eos_id = read_index(
                    os.path.join(cfg.data.train_dir, names[0])
                ).get("eos_id", -1)
            except FileNotFoundError:
                pass
        train = MixtureLoader(loaders, weights)

        val = None
        if os.path.isdir(cfg.data.val_dir):
            val = ResumableLoader(
                TokenDataset(cfg.data.val_dir, cfg.data.seq_len),
                **{**common, "num_workers": 0},
            )
        if self.dist.is_main:
            got = train.realised_weights(20_000)
            print("[data] mixture " + ", ".join(
                f"{n}={100*got[n]:.1f}%" for n in names) + f" | accum {self.accum}")
            if cfg.data.curriculum:
                print(f"[data] curriculum with {len(cfg.data.curriculum)} stages")
        return train, val

    def _apply_curriculum(self, step: int) -> None:
        """Switch the mixture when the run crosses a curriculum boundary."""
        if not self.cfg.data.curriculum or not isinstance(self.train_loader, MixtureLoader):
            return
        want = self._stage_mixture(step)
        if want != self._active_mixture:
            self.train_loader.set_weights(want)
            self._active_mixture = want
            if self.dist.is_main:
                pretty = ", ".join(f"{k}={v}" for k, v in sorted(want.items()))
                print(f"\n[data] curriculum stage change at step {step}: {pretty}\n",
                      flush=True)

    def _build_sft_loaders(self):
        """Instruction data with prompt tokens masked out of the loss."""
        from slm.sft import SFTDataset, SFTLoader
        from slm.tokenizer import Tokenizer

        cfg = self.cfg
        tok = Tokenizer.load(cfg.finetune.tokenizer)
        cfg.data.eos_id = tok.eos_id
        common = dict(micro_batch_size=cfg.train.micro_batch_size, rank=self.dist.rank,
                      world_size=self.dist.world_size, seed=cfg.data.seed,
                      device=self.device)
        train_ds = SFTDataset(cfg.finetune.data, tok, cfg.data.seq_len,
                              cfg.finetune.drop_long)
        train = SFTLoader(train_ds, **common)
        val = None
        if cfg.finetune.val_data and os.path.exists(cfg.finetune.val_data):
            val = SFTLoader(SFTDataset(cfg.finetune.val_data, tok, cfg.data.seq_len,
                                       cfg.finetune.drop_long), **common)
        if self.dist.is_main:
            print(f"[sft] {train_ds} | accum {self.accum}")
        return train, val

    def _load_teacher(self):
        if not self.cfg.distill.teacher:
            return None
        from slm.train.distill import load_teacher

        teacher = load_teacher(self.cfg.distill.teacher, self.device)
        if self.dist.is_main:
            print(f"[distill] teacher {teacher.num_params()/1e6:.1f}M params, "
                  f"alpha={self.cfg.distill.alpha} T={self.cfg.distill.temperature}")
        return teacher

    def _init_from_checkpoint(self) -> None:
        """Start a fine-tune from pretrained weights without inheriting run state."""
        path = self.cfg.finetune.init_from
        if not path:
            return
        ck = CheckpointManager.load(path, map_location="cpu")
        load_into({"model": self.model}, ck, strict=True, verbose=self.dist.is_main)
        self.raw_model.to(self.device)
        if self.dist.is_main:
            print(f"[init] weights loaded from {path} (optimiser and data reset)")

    # --- resume --------------------------------------------------------------
    def resume(self) -> int:
        """Restore from the newest valid checkpoint, pulling from the mirror if needed."""
        mode = self.cfg.train.resume
        if mode == "none":
            return 0
        path = self.ckpt.latest() if mode == "auto" else mode
        if path is None:
            path = self.ckpt.fetch_remote()
        if path is None:
            if self.dist.is_main:
                print("[resume] no checkpoint found; starting from scratch")
            return 0

        ck = CheckpointManager.load(path, map_location="cpu")
        load_into(
            {"model": self.model, "optim": self.optimizer, "loader": self.train_loader},
            ck, strict=True, verbose=self.dist.is_main,
        )
        self.raw_model.to(self.device)
        trainer_sd = ck["trainer"]
        # the stored step is a count of completed steps, so it is also the next index
        self.state.step = trainer_sd["step"]
        self.state.tokens = trainer_sd.get("extra", {}).get("tokens", 0)
        self.state.best_val = trainer_sd.get("extra", {}).get("best_val", float("inf"))
        saved_max = trainer_sd.get("extra", {}).get("max_steps")
        if saved_max and saved_max != self.max_steps and self.dist.is_main:
            print(
                f"[resume] WARNING max_steps changed {saved_max} -> {self.max_steps}. "
                "The LR schedule is a function of max_steps, so this run will not "
                "reproduce the original trajectory. Set train.max_steps="
                f"{saved_max} to continue the same schedule."
            )
        if self.dist.is_main:
            saved_world = trainer_sd.get("world_size", 1)
            note = "" if saved_world == self.dist.world_size else (
                f" (was {saved_world} ranks, now {self.dist.world_size})"
            )
            print(
                f"[resume] continuing at step {self.state.step} from "
                f"{os.path.basename(path)}{note} | {self.state.tokens:,} tokens seen"
            )
        return self.state.step

    # --- graceful finish -----------------------------------------------------
    def _check_finish_file(self) -> None:
        """A FINISH file re-points max_steps so the LR decay starts now."""
        if not self.cfg.train.watch_finish_file or self._finished_early:
            return
        path = os.path.join(self.run_dir, FINISH_FILE)
        if not os.path.exists(path):
            return
        decay = self.cfg.optim.decay_steps or max(1, int(0.2 * self.max_steps))
        try:
            with open(path) as f:
                override = int(f.read().strip() or 0)
        except (ValueError, OSError):
            override = 0
        decay = override or decay
        self.max_steps = self.state.step + decay
        self.cfg.optim.decay_steps = decay
        self._finished_early = True
        if self.dist.is_main:
            print(
                f"\n[finish] FINISH requested: decaying over {decay} steps and "
                f"stopping at step {self.max_steps}\n", flush=True
            )

    # --- steps ---------------------------------------------------------------
    def _autocast(self):
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[self.cfg.train.dtype]
        if self.device.type != "cuda" or dtype == torch.float32:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _doc_ids(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.cfg.model.doc_masking or self.cfg.data.eos_id < 0:
            return None
        return self.raw_model.doc_ids(x, self.cfg.data.eos_id)

    def train_step(self, index: int) -> dict:
        """One optimiser step: accum micro-batches, clip, step, rebalance routers."""
        cfg = self.cfg
        lr = lr_at(index, cfg.optim, self.max_steps)
        muon_lr = None
        if cfg.optim.kind == "muon":
            muon_lr = lr * (cfg.optim.muon_lr / max(cfg.optim.lr, 1e-12))
        set_lr(self.optimizer, lr, muon_lr)
        self.optimizer.zero_grad(set_to_none=True)

        totals = {"loss": 0.0, "ce": 0.0}
        for micro in range(self.accum):
            x, y, mask = _unpack(self.train_loader.next_batch())
            is_last = micro == self.accum - 1
            ctx = (
                self.model.no_sync()
                if (not is_last and hasattr(self.model, "no_sync"))
                else _null_context()
            )
            with ctx, self._autocast():
                out = self.model(x, targets=y, doc_ids=self._doc_ids(x),
                                 loss_mask=mask, zloss=cfg.optim.zloss,
                                 return_logits=self.teacher is not None)
                loss = out.loss
                if self.teacher is not None:
                    loss, kd = self._blend_distillation(loss, out.logits, x, mask)
                    totals["kd"] = kd
                loss = loss / self.accum
            loss.backward()
            totals["loss"] += float(out.loss.detach())
            totals["ce"] += float(out.metrics.get("ce", out.loss).detach())
            if "load_cv" in out.metrics:
                totals["load_cv"] = float(out.metrics["load_cv"])

        grad_norm = clip_grad_norm(self.model, cfg.optim.grad_clip)
        self.optimizer.step()
        self.raw_model.update_router_biases()

        self.state.tokens += self.tokens_per_step
        metrics = {k: v / self.accum if k in ("loss", "ce") else v
                   for k, v in totals.items()}
        metrics.update(lr=lr, grad_norm=float(grad_norm))
        return metrics

    def _blend_distillation(self, ce: torch.Tensor, student_logits, x, mask):
        """loss = (1-alpha)*CE + alpha*KL(teacher || student)."""
        from slm.train.distill import distillation_loss

        d = self.cfg.distill
        with torch.no_grad():
            t_logits = self.teacher(x, return_logits=True).logits
        kd = distillation_loss(student_logits, t_logits, d.temperature, mask, d.top_k)
        return (1 - d.alpha) * ce + d.alpha * kd, float(kd.detach())

    @torch.no_grad()
    def evaluate(self, steps: int | None = None) -> dict:
        """Mean validation loss over a fixed number of batches."""
        if self.val_loader is None:
            return {}
        steps = steps or self.cfg.train.eval_steps
        self.model.eval()
        total, n = 0.0, 0
        for _ in range(steps):
            x, y, mask = _unpack(self.val_loader.next_batch())
            with self._autocast():
                out = self.model(x, targets=y, doc_ids=self._doc_ids(x),
                                 loss_mask=mask, return_logits=False)
            total += float(out.metrics["ce"])
            n += 1
        self.model.train()
        loss = all_reduce_mean(total / max(n, 1), self.device)
        return {"loss": loss, "ppl": math.exp(min(loss, 20)),
                "bpb": loss / math.log(2) / self._bytes_per_token()}

    def _bytes_per_token(self) -> float:
        """Compression of the tokenizer, so bits-per-byte is comparable across vocabs."""
        try:
            return read_index(self.cfg.data.train_dir).get("bytes_per_token", 1.0) or 1.0
        except (FileNotFoundError, KeyError):
            return 1.0

    # --- checkpointing -------------------------------------------------------
    def save(self, permanent: bool = False, blocking: bool | None = None) -> str | None:
        return self.ckpt.save(
            step=self.state.step,
            model=self.model,
            optimizer=self.optimizer,
            loader_state=self.train_loader.state_dict(),
            extra={"tokens": self.state.tokens, "best_val": self.state.best_val,
                   "max_steps": self.max_steps},
            config=self.cfg.to_dict(),
            metrics={"val_loss": self.state.best_val},
            permanent=permanent,
            blocking=blocking,
        )

    # --- main loop -----------------------------------------------------------
    def train(self) -> int:
        """Run to max_steps or until preempted. Returns the process exit code."""
        cfg = self.cfg
        start = self.resume()
        if start == 0:
            self._init_from_checkpoint()
        self.guard.install()
        self.model.train()
        if self.dist.is_main:
            budget = self.max_steps * self.tokens_per_step
            unit = f"{budget/1e9:.1f}B" if budget >= 1e9 else f"{budget/1e6:.1f}M"
            print(f"[train] steps {start} -> {self.max_steps} | target {unit} tokens")

        t_last = time.perf_counter()
        stopped = False
        try:
            while self.state.step < self.max_steps:
                index = self.state.step
                self._check_finish_file()
                self._apply_curriculum(index)
                if index >= self.max_steps:
                    break

                metrics = self.train_step(index)
                self.state.step = index + 1
                dt = self.timer_tick(t_last)
                t_last = time.perf_counter()

                done = self.state.step
                if index % cfg.train.log_every == 0:
                    self._log_train(index, metrics, dt)
                if cfg.train.eval_every and done % cfg.train.eval_every == 0:
                    self._run_eval(index)
                if cfg.ckpt.interval and done % cfg.ckpt.interval == 0:
                    milestone = bool(cfg.ckpt.permanent_every) and (
                        done % cfg.ckpt.permanent_every == 0
                    )
                    self.save(permanent=milestone)

                if self.guard.should_stop():
                    stopped = True
                    break

            if cfg.ckpt.save_on_exit:
                if self.dist.is_main:
                    print(f"[ckpt] final save after {self.state.step} steps", flush=True)
                self.save(permanent=True, blocking=True)
                self.ckpt.wait()
            if not stopped and self.state.step >= self.max_steps and self.dist.is_main:
                # the marker a supervisor loop watches to know not to relaunch
                with open(os.path.join(self.run_dir, "DONE"), "w") as f:
                    f.write(f"{self.state.step}\n{self.state.tokens}\n")
        finally:
            self.close()

        if stopped and self.dist.is_main:
            print(f"[train] stopped early: {self.guard.reason}")
        return cfg.train.preempt_exit_code if stopped else 0

    def timer_tick(self, t_last: float) -> float:
        dt = time.perf_counter() - t_last
        self.telemetry.timer.times.append(dt)
        return dt

    def _log_train(self, step: int, metrics: dict, dt: float) -> None:
        mean_dt = self.telemetry.timer.mean or dt
        tps = self.tokens_per_step / mean_dt if mean_dt else 0.0
        remaining = (self.max_steps - step) * mean_dt
        full = {
            **metrics,
            "tokens": self.state.tokens,
            "tokens_per_sec": tps,
            "step_time": mean_dt,
            "mfu": self.telemetry.mfu(self.tokens_per_step, mean_dt),
            "eta_hours": remaining / 3600,
            "epoch": self.train_loader.state.epoch,
        }
        self.telemetry.log(step, full)
        self.telemetry.console(step, self.max_steps, full)

    def _run_eval(self, step: int) -> None:
        res = self.evaluate()
        if not res:
            return
        self.telemetry.log(step, res, prefix="val")
        if self.dist.is_main:
            print(f"           val loss {res['loss']:.4f} | ppl {res['ppl']:.2f} "
                  f"| bpb {res['bpb']:.4f}", flush=True)
        if res["loss"] < self.state.best_val:
            self.state.best_val = res["loss"]

    def close(self) -> None:
        """Release every background thread this trainer started."""
        self.guard.uninstall()
        self.ckpt.close()
        self.train_loader.stop()
        if self.val_loader is not None:
            self.val_loader.stop()
        self.telemetry.finish()
        barrier()
        cleanup_distributed()


def _unpack(batch):
    """Loaders yield (x, y) for pretraining and (x, y, loss_mask) for SFT."""
    if batch is None:
        raise RuntimeError("data loader was exhausted mid-run")
    return batch if len(batch) == 3 else (batch[0], batch[1], None)


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
