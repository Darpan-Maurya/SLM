# Operations

Running this on rented, unreliable GPUs.

## The model of failure

Assume the machine dies without warning, at any moment, repeatedly. Everything
below follows from that.

| Failure | What happens | Recovery |
|---|---|---|
| SIGTERM from the container runtime | guard flags it, current step finishes, checkpoint is written, exit 0 | supervisor relaunches, resumes at the exact step |
| Instance vanishes with no signal | last rolling checkpoint is on the remote mirror | new machine pulls it and continues |
| Checkpoint corrupted in transit | sha256 mismatch on load | previous checkpoint is used; one interval lost |
| Wall clock / session limit | `train.max_runtime_sec` stops it voluntarily first | relaunch |
| Credits run out mid-run | `echo N > runs/<name>/FINISH` | LR decays over N steps, run ends properly annealed |
| Different GPU count on the new host | data cursor is global, not per-rank | resumes; a warning prints if the schedule changed |

## Launching

Single GPU:

```bash
python scripts/train.py configs/train/pretrain-300m.yaml
```

Multi-GPU:

```bash
torchrun --nproc_per_node=8 scripts/train.py configs/train/pretrain-300m.yaml
```

Spot / preemptible, with auto-restart:

```bash
MAX_RESTARTS=1000 bash scripts/supervise.sh configs/train/pretrain-300m.yaml \
    ckpt.remote=s3://my-bucket/slm-300m \
    train.poll_cloud_preemption=true \
    ckpt.interval=500
```

`supervise.sh` relaunches the identical command until `runs/<name>/DONE` appears.
Every launch resumes; none restarts.

Docker (the entrypoint *is* the supervisor):

```bash
docker build -t slm .
docker run --gpus all -v $PWD/data:/workspace/data -v $PWD/runs:/workspace/runs \
    -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
    slm configs/train/pretrain-300m.yaml ckpt.remote=s3://my-bucket/slm-300m
```

## Choosing a checkpoint interval

The tradeoff is checkpoint write time against work lost on preemption. Expected
loss per preemption is half the interval.

| Interval | Work at risk (0.5M tok/step) | Write cost (300M, bf16) |
|---|---|---|
| 100 steps | ~25M tokens | ~1.2 GB every ~4 min |
| 500 steps | ~130M tokens | ~1.2 GB every ~20 min |
| 2000 steps | ~500M tokens | ~1.2 GB every ~80 min |

500 is a reasonable default on spot instances; 2000 on reserved capacity. Saves
are asynchronous (`ckpt.async_save=true`) — weights are copied to CPU and written
from a worker thread, so the step loop does not stall.

`ckpt.keep_last` rolling checkpoints are retained plus every
`ckpt.permanent_every` as a never-deleted milestone, so mid-training branches
stay available for the WSD-decay trick.

## Remote mirrors

```
ckpt.remote=s3://bucket/prefix        # needs boto3 and AWS credentials
ckpt.remote=hf://user/repo            # needs huggingface_hub; private by default
ckpt.remote=rsync://user@host:/path   # needs ssh access
ckpt.remote=/mnt/network-volume       # any second filesystem
```

Uploads happen after the local rename, from the same background thread, and a
failure prints a warning rather than killing the run. `ckpt.remote_every=N` mirrors
only every Nth checkpoint if bandwidth is the constraint.

## Monitoring

- `runs/<name>/logs/metrics.jsonl` — one JSON object per logged step
- console line: loss, ppl, lr, grad norm, tokens/sec, **MFU**, step time, ETA
- `train.wandb_project=...` to mirror to Weights & Biases

MFU is computed against a table of per-device peak bf16 FLOPs. Below ~30% on an
H100 at this model size, look at: `micro_batch_size` too small, `compile=false`,
gradient checkpointing on when it does not need to be, or a slow data path
(`num_workers`).

## Health signals

The metrics that tell you a run is going wrong before the loss does:

| Signal | Healthy | Trouble |
|---|---|---|
| `grad_norm` | steady, near `grad_clip` | spikes >10× baseline, or collapse to ~0 |
| `zloss` | small and flat | rising — the softmax denominator is drifting |
| `load_cv` (MoE) | falling then flat | rising — experts are collapsing onto a few |
| `mfu` | flat | sawtooth — data starvation |
| `epoch` | 0 for a single-epoch budget | >0 means you are repeating data |

## Stopping

```bash
touch runs/<name>/STOP          # checkpoint and exit; supervisor will not relaunch
echo 2000 > runs/<name>/FINISH  # decay over 2000 steps, then finish properly
```

`FINISH` is the one to use when the budget is ending. `STOP` is for aborting.

## Cost

Estimated at 40% MFU, dense bf16 peak.

| Run | Tokens | GPU-hours (H100) | At $2.50/hr |
|---|---|---|---|
| Ablation grid (12 × 100M × 2B) | 24B | ~18 | ~$45 |
| `slm-300m` Chinchilla (6B) | 6B | ~9 | ~$23 |
| `slm-300m` target (60B) | 60B | ~93 | ~$233 |
| `slm-moe-500m` (60B) | 60B | ~46 | ~$115 |

The MoE is cheaper per token than the dense model at the same parameter count —
that is the entire point of it.
