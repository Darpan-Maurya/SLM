"""Checkpointing built for machines that disappear without warning."""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import random
import shutil
import subprocess
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from slm.config import CheckpointConfig

MANIFEST = "manifest.json"
LATEST = "LATEST"
CKPT_PREFIX = "step_"


# --- helpers -------------------------------------------------------------------
def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Peel off DDP / torch.compile / FSDP wrappers to reach the real module."""
    seen = 0
    while seen < 8:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod           # torch.compile
        elif hasattr(model, "module"):
            model = model.module              # DDP / FSDP / DataParallel
        else:
            break
        seen += 1
    return model


def clean_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip wrapper prefixes so a checkpoint never encodes how it was trained."""
    out = {}
    for k, v in sd.items():
        changed = True
        while changed:
            changed = False
            for prefix in ("_orig_mod.", "module."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    changed = True
        out[k] = v
    return out


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fsync_dir(path: str) -> None:
    # A rename is only durable once the *directory* entry is flushed. Without this a power loss...
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _torch_save(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(_as_tuple(state["python"]))
    if "numpy" in state:
        np.random.set_state(_as_tuple(state["numpy"]))
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda"]])
        except (RuntimeError, ValueError):
            # different GPU count on the new host: CUDA RNG is not portable, and nothing that affects the...
            pass


def _as_tuple(x):
    """JSON round-trips turn RNG state tuples into lists; put them back."""
    if isinstance(x, list):
        return tuple(_as_tuple(i) for i in x)
    return x


def snapshot_to_cpu(sd: dict) -> dict:
    """Detached CPU copy, so training can proceed while the writer thread works."""
    out = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().to("cpu", copy=True)
        elif isinstance(v, dict):
            out[k] = snapshot_to_cpu(v)
        elif isinstance(v, (list, tuple)):
            out[k] = type(v)(
                snapshot_to_cpu(i) if isinstance(i, dict)
                else i.detach().to("cpu", copy=True) if isinstance(i, torch.Tensor)
                else i
                for i in v
            )
        else:
            out[k] = v
    return out


# --- remote --------------------------------------------------------------------
class RemoteStore(Protocol):
    def push(self, local_dir: str, name: str) -> None: ...
    def pull(self, name: str, local_dir: str) -> bool: ...
    def list(self) -> list[str]: ...


class LocalStore:
    """A second filesystem: a network mount, an attached volume, anywhere else."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def push(self, local_dir: str, name: str) -> None:
        dest = os.path.join(self.root, name)
        tmp = dest + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        shutil.copytree(local_dir, tmp)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.replace(tmp, dest)
        _fsync_dir(self.root)

    def pull(self, name: str, local_dir: str) -> bool:
        src = os.path.join(self.root, name)
        if not os.path.isdir(src):
            return False
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        shutil.copytree(src, local_dir)
        return True

    def list(self) -> list[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(d for d in os.listdir(self.root) if d.startswith(CKPT_PREFIX))


class RsyncStore:
    """``rsync://user@host:/path`` - works with any box you can ssh into."""

    def __init__(self, target: str):
        self.target = target.removeprefix("rsync://")

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, timeout=3600)

    def push(self, local_dir: str, name: str) -> None:
        dest = f"{self.target.rstrip('/')}/{name}/"
        r = self._run(["rsync", "-az", "--partial", "--mkpath",
                       local_dir.rstrip("/") + "/", dest])
        if r.returncode != 0:
            raise RuntimeError(f"rsync push failed: {r.stderr[:400]}")

    def pull(self, name: str, local_dir: str) -> bool:
        os.makedirs(local_dir, exist_ok=True)
        r = self._run(["rsync", "-az", f"{self.target.rstrip('/')}/{name}/", local_dir])
        return r.returncode == 0

    def list(self) -> list[str]:
        host, _, path = self.target.partition(":")
        r = self._run(["ssh", host, f"ls -1 {path}"])
        if r.returncode != 0:
            return []
        return sorted(x for x in r.stdout.split() if x.startswith(CKPT_PREFIX))


class S3Store:
    def __init__(self, uri: str):
        import boto3

        rest = uri.removeprefix("s3://")
        self.bucket, _, self.prefix = rest.partition("/")
        self.prefix = self.prefix.strip("/")
        self.s3 = boto3.client("s3")

    def _key(self, *parts: str) -> str:
        return "/".join(p for p in (self.prefix, *parts) if p)

    def push(self, local_dir: str, name: str) -> None:
        for fn in sorted(os.listdir(local_dir)):
            self.s3.upload_file(os.path.join(local_dir, fn), self.bucket,
                                self._key(name, fn))

    def pull(self, name: str, local_dir: str) -> bool:
        os.makedirs(local_dir, exist_ok=True)
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=self._key(name) + "/")
        contents = resp.get("Contents", [])
        if not contents:
            return False
        for obj in contents:
            fn = obj["Key"].rsplit("/", 1)[-1]
            self.s3.download_file(self.bucket, obj["Key"], os.path.join(local_dir, fn))
        return True

    def list(self) -> list[str]:
        paginator = self.s3.get_paginator("list_objects_v2")
        names = set()
        for page in paginator.paginate(
            Bucket=self.bucket, Prefix=self._key() + "/", Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                leaf = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                if leaf.startswith(CKPT_PREFIX):
                    names.add(leaf)
        return sorted(names)


class HFStore:
    """A Hugging Face repo as the mirror - free, versioned, and resumable."""

    def __init__(self, uri: str, repo_type: str = "model"):
        from huggingface_hub import HfApi

        self.repo_id = uri.removeprefix("hf://")
        self.repo_type = repo_type
        self.api = HfApi()
        self.api.create_repo(self.repo_id, repo_type=repo_type, exist_ok=True,
                             private=True)

    def push(self, local_dir: str, name: str) -> None:
        self.api.upload_folder(
            folder_path=local_dir, path_in_repo=name,
            repo_id=self.repo_id, repo_type=self.repo_type,
            commit_message=f"checkpoint {name}",
        )

    def pull(self, name: str, local_dir: str) -> bool:
        from huggingface_hub import snapshot_download

        try:
            path = snapshot_download(
                self.repo_id, repo_type=self.repo_type, allow_patterns=f"{name}/*",
            )
        except Exception:
            return False
        src = os.path.join(path, name)
        if not os.path.isdir(src):
            return False
        os.makedirs(local_dir, exist_ok=True)
        for fn in os.listdir(src):
            shutil.copy2(os.path.join(src, fn), os.path.join(local_dir, fn))
        return True

    def list(self) -> list[str]:
        try:
            files = self.api.list_repo_files(self.repo_id, repo_type=self.repo_type)
        except Exception:
            return []
        return sorted({f.split("/")[0] for f in files if f.startswith(CKPT_PREFIX)})


def make_store(uri: str) -> RemoteStore | None:
    """``""`` -> None, otherwise dispatch on scheme."""
    if not uri:
        return None
    if uri.startswith("s3://"):
        return S3Store(uri)
    if uri.startswith("hf://"):
        return HFStore(uri)
    if uri.startswith("rsync://"):
        return RsyncStore(uri)
    return LocalStore(uri.removeprefix("file://"))


# --- manager -------------------------------------------------------------------
# a writer thread blocked in queue.get() stalls interpreter shutdown, so every
# manager that starts one is tracked and retired at exit
_LIVE_MANAGERS: weakref.WeakSet[CheckpointManager] = weakref.WeakSet()


def _shutdown_managers() -> None:
    for mgr in list(_LIVE_MANAGERS):
        try:
            mgr.close(timeout=30.0)
        except Exception:
            pass


atexit.register(_shutdown_managers)


@dataclass
class CheckpointInfo:
    path: str
    step: int
    tokens: int
    val_loss: float | None = None


class CheckpointManager:
    """Writes, verifies, rotates, and mirrors checkpoints for one run."""

    def __init__(
        self,
        run_dir: str,
        cfg: CheckpointConfig | None = None,
        rank: int = 0,
        store: RemoteStore | None = None,
    ):
        self.cfg = cfg or CheckpointConfig()
        self.run_dir = run_dir
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.rank = rank
        self.is_main = rank == 0
        if self.is_main:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.store = store if store is not None else make_store(self.cfg.remote)
        self._n_saved = 0
        self._best: float | None = None
        self._q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._errors: list[BaseException] = []

    # -- naming -------------------------------------------------------------- #
    @staticmethod
    def name_for(step: int) -> str:
        return f"{CKPT_PREFIX}{step:08d}"

    @staticmethod
    def step_of(name: str) -> int:
        return int(os.path.basename(name.rstrip("/")).removeprefix(CKPT_PREFIX))

    def path_for(self, step: int) -> str:
        return os.path.join(self.ckpt_dir, self.name_for(step))

    # -- save ---------------------------------------------------------------- #
    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        loader_state: dict | None = None,
        extra: dict | None = None,
        config: dict | None = None,
        metrics: dict | None = None,
        permanent: bool = False,
        blocking: bool | None = None,
    ) -> str | None:
        """Persist everything needed to continue this run. Rank 0 only."""
        if not self.is_main:
            return None

        model_sd = clean_state_dict(unwrap_model(model).state_dict())
        optim_sd = (
            optimizer.state_dict() if (optimizer and self.cfg.save_optimizer) else None
        )
        trainer_sd = {
            "step": step,
            "loader": loader_state or {},
            "rng": rng_state(),
            "metrics": metrics or {},
            "extra": extra or {},
            "wall_time": time.time(),
            "git_commit": git_commit(),
            "torch_version": torch.__version__,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        }

        blocking = (not self.cfg.async_save) if blocking is None else blocking
        if not blocking:
            model_sd = snapshot_to_cpu(model_sd)
            optim_sd = snapshot_to_cpu(optim_sd) if optim_sd else None
            trainer_sd = snapshot_to_cpu(trainer_sd)

        payload = dict(
            step=step, model=model_sd, optim=optim_sd, trainer=trainer_sd,
            config=config or {}, permanent=permanent,
            val_loss=(metrics or {}).get("val_loss"),
        )
        if blocking:
            return self._write(payload)
        self._enqueue(payload)
        return self.path_for(step)

    def _enqueue(self, payload: dict) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._drain, daemon=True)
            self._worker.start()
            _LIVE_MANAGERS.add(self)
        self._q.put(payload)

    def _drain(self) -> None:
        while True:
            payload = self._q.get()
            if payload is None:
                self._q.task_done()
                return
            try:
                self._write(payload)
            except BaseException as exc:                      # never kill training
                self._errors.append(exc)
                print(f"[ckpt] WARNING background save failed: {exc!r}", flush=True)
            finally:
                self._q.task_done()

    def _write(self, payload: dict) -> str:
        step = payload["step"]
        final = self.path_for(step)
        tmp = final + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)

        files: list[str] = []
        _torch_save(payload["model"], os.path.join(tmp, "model.pt"))
        files.append("model.pt")
        if payload["optim"] is not None:
            _torch_save(payload["optim"], os.path.join(tmp, "optim.pt"))
            files.append("optim.pt")
        _torch_save(payload["trainer"], os.path.join(tmp, "trainer.pt"))
        files.append("trainer.pt")
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(payload["config"], f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        files.append("config.json")

        manifest = {
            "step": step,
            "created": time.time(),
            "permanent": bool(payload["permanent"]),
            "val_loss": payload["val_loss"],
            "files": {
                fn: {
                    "size": os.path.getsize(os.path.join(tmp, fn)),
                    "sha256": sha256_file(os.path.join(tmp, fn))
                    if self.cfg.verify_hash else None,
                }
                for fn in files
            },
        }
        with open(os.path.join(tmp, MANIFEST), "w") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(tmp)

        if os.path.exists(final):
            shutil.rmtree(final)
        os.replace(tmp, final)              # <- the checkpoint becomes visible here
        _fsync_dir(self.ckpt_dir)
        self._write_latest(step)

        self._n_saved += 1
        if self.store is not None and self._n_saved % max(1, self.cfg.remote_every) == 0:
            try:
                self.store.push(final, self.name_for(step))
            except Exception as exc:
                print(f"[ckpt] WARNING remote push failed: {exc!r}", flush=True)

        self._rotate()
        return final

    def _write_latest(self, step: int) -> None:
        # A plain text file, not a symlink: symlinks do not survive S3, some container volumes, or...
        path = os.path.join(self.ckpt_dir, LATEST)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(self.name_for(step) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(self.ckpt_dir)

    def close(self, timeout: float = 300.0) -> None:
        """Flush pending saves and retire the writer thread."""
        worker, self._worker = self._worker, None
        if worker is not None and worker.is_alive():
            self._q.put(None)
            worker.join(timeout=timeout)
        _LIVE_MANAGERS.discard(self)

    def __del__(self) -> None:
        try:
            self.close(timeout=5.0)
        except Exception:
            pass

    def wait(self, timeout: float = 900.0) -> None:
        """Block until every queued save has been written to disk."""
        deadline = time.time() + timeout
        # unfinished_tasks, not empty(): a dequeued item is still being written
        while self._q.unfinished_tasks and time.time() < deadline:
            time.sleep(0.02)
        if self._errors:
            raise self._errors[-1]

    # -- rotation ------------------------------------------------------------ #
    def _rotate(self) -> None:
        keep = max(1, self.cfg.keep_last)
        entries = self.list_checkpoints()
        rolling = []
        for info in entries:
            m = self._manifest(info.path)
            if m.get("permanent"):
                continue
            if self.cfg.permanent_every and info.step % self.cfg.permanent_every == 0:
                continue
            rolling.append(info)
        for info in rolling[:-keep] if len(rolling) > keep else []:
            shutil.rmtree(info.path, ignore_errors=True)

    def _manifest(self, path: str) -> dict:
        try:
            with open(os.path.join(path, MANIFEST)) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    # -- discovery ----------------------------------------------------------- #
    def list_checkpoints(self, verify: bool = False) -> list[CheckpointInfo]:
        if not os.path.isdir(self.ckpt_dir):
            return []
        out = []
        for name in sorted(os.listdir(self.ckpt_dir)):
            path = os.path.join(self.ckpt_dir, name)
            if not name.startswith(CKPT_PREFIX) or not os.path.isdir(path):
                continue
            if name.endswith(".tmp"):
                continue
            m = self._manifest(path)
            if not m:
                continue
            if verify and not self.verify(path):
                continue
            out.append(CheckpointInfo(path, m["step"], m.get("tokens", 0),
                                      m.get("val_loss")))
        return sorted(out, key=lambda i: i.step)

    def latest(self, verify: bool = True) -> str | None:
        """Newest complete checkpoint, falling back through older ones."""
        pointer = os.path.join(self.ckpt_dir, LATEST)
        candidates: list[str] = []
        if os.path.exists(pointer):
            with open(pointer) as f:
                name = f.read().strip()
            p = os.path.join(self.ckpt_dir, name)
            if os.path.isdir(p):
                candidates.append(p)
        candidates += [i.path for i in reversed(self.list_checkpoints())]

        for path in candidates:
            if not verify or self.verify(path):
                return path
            print(f"[ckpt] rejecting corrupt checkpoint {path}", flush=True)
        return None

    def verify(self, path: str) -> bool:
        m = self._manifest(path)
        if not m or "files" not in m:
            return False
        for fn, meta in m["files"].items():
            fp = os.path.join(path, fn)
            if not os.path.exists(fp):
                return False
            if os.path.getsize(fp) != meta["size"]:
                return False
            if meta.get("sha256") and sha256_file(fp) != meta["sha256"]:
                return False
        return True

    def fetch_remote(self, name: str | None = None) -> str | None:
        """Pull a checkpoint down from the mirror - the cross-provider path."""
        if self.store is None:
            return None
        names = self.store.list()
        if not names:
            return None
        name = name or names[-1]
        local = os.path.join(self.ckpt_dir, name)
        if os.path.isdir(local) and self.verify(local):
            return local
        print(f"[ckpt] pulling {name} from remote mirror", flush=True)
        if not self.store.pull(name, local):
            return None
        if not self.verify(local):
            print(f"[ckpt] remote copy of {name} failed verification", flush=True)
            return None
        self._write_latest(self.step_of(name))
        return local

    # -- load ---------------------------------------------------------------- #
    @staticmethod
    def load(path: str, map_location: str | torch.device = "cpu") -> dict:
        def _load(fn, required=True):
            fp = os.path.join(path, fn)
            if not os.path.exists(fp):
                if required:
                    raise FileNotFoundError(fp)
                return None
            return torch.load(fp, map_location=map_location, weights_only=False)

        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        return {
            "model": _load("model.pt"),
            "optim": _load("optim.pt", required=False),
            "trainer": _load("trainer.pt"),
            "config": config,
            "path": path,
        }

    @staticmethod
    def export_inference(src: str, dest: str, dtype: torch.dtype | None = None) -> str:
        """Weights + config only: what you actually ship."""
        os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
        sd = torch.load(os.path.join(src, "model.pt"), map_location="cpu",
                        weights_only=False)
        if dtype is not None:
            sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}
        with open(os.path.join(src, "config.json")) as f:
            config = json.load(f)
        _torch_save({"model": sd, "config": config, "format": "slm-inference-v1"}, dest)
        return dest


def load_into(
    objects: dict[str, Any], ckpt: dict, strict: bool = True, verbose: bool = True
) -> None:
    """Restore ``{"model": m, "optim": o, ...}`` from a loaded checkpoint dict."""
    if "model" in objects and ckpt.get("model"):
        target = unwrap_model(objects["model"])
        missing, unexpected = target.load_state_dict(
            clean_state_dict(ckpt["model"]), strict=strict
        )
        if verbose and (missing or unexpected):
            print(f"[ckpt] missing={list(missing)} unexpected={list(unexpected)}")
    if "optim" in objects and ckpt.get("optim") and objects["optim"] is not None:
        objects["optim"].load_state_dict(ckpt["optim"])
    trainer = ckpt.get("trainer") or {}
    if "loader" in objects and trainer.get("loader") and objects["loader"] is not None:
        objects["loader"].load_state_dict(trainer["loader"])
    set_rng_state(trainer.get("rng", {}))
