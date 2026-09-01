# SLM — a small language model, trained from scratch, built to survive preemption

A complete pretraining stack for a 300M-parameter language model: tokenizer, data
pipeline, model, trainer, evaluation harness and inference — no `transformers`, no
`datasets` required, ~6,000 lines of Python.

The design constraint that shaped everything: **the machine can disappear at any
moment, and the run has to continue somewhere else.** Rented A100s get reclaimed,
Colab sessions end, credits run out. So the interesting engineering here is not
the model — it is that a run interrupted at step N and restarted on a different
provider with a different number of GPUs sees *the same tokens in the same order*
and continues the same loss curve.

That is a testable claim, so it is a test:

```
tests/test_train_e2e.py::test_training_resumes_with_an_identical_loss_trajectory
```

It trains 24 steps, kills the run at step 12 through the real preemption path,
restarts it, and asserts every loss from step 12 to 23 matches the uninterrupted
run to within 1e-4.

---

## Quickstart

```bash
make install          # venv + dependencies
make test             # 241 tests, ~30 seconds
make smoke            # full pipeline on TinyShakespeare: ~1 minute, CPU only
```

`make smoke` trains a tokenizer, shards a corpus, starts training, gets preempted
mid-run, resumes from the checkpoint, finishes, and samples from the result.

Real training:

```bash
# 1. tokenizer (32,768 byte-level BPE, trained on your corpus)
python scripts/train_tokenizer.py hf://HuggingFaceFW/fineweb-edu:train -o data/tokenizer.json

# 2. shards (uint16, memory-mapped, resumable)
python scripts/prepare_data.py hf://HuggingFaceFW/fineweb-edu:train -o data/tokens/train

# 3. train (single GPU)
python scripts/train.py configs/train/pretrain-300m.yaml

# 3b. train (8 GPUs, auto-restarting on preemption, mirrored to S3)
torchrun --nproc_per_node=8 scripts/train.py configs/train/pretrain-300m.yaml \
    ckpt.remote=s3://my-bucket/slm-300m train.poll_cloud_preemption=true
```

---

## The models

| Config | Params | Active | Layers | d | Heads (q/kv) | FFN | GFLOP/token |
|---|---|---|---|---|---|---|---|
| `slm-100m` | 100M | 100M | 12 | 768 | 12 / 3 | 2048 dense | 0.67 |
| `slm-300m` | 318M | 304M | 24 | 1024 | 16 / 4 | 2816 dense | 2.23 |
| `slm-moe-500m` | 521M | 133M | 24 | 768 | 12 / 3 | 32 experts × 256, top-4 + 1 shared | 1.10 |

Two flagships at the same parameter budget answering different questions: the
dense one is what you deploy when memory is scarce, the sparse one when FLOPs
are. `slm-100m` is the proxy every ablation runs on.

### Architecture

Every component is from a paper, and every choice is defended in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) against the alternative it beat:

- **RMSNorm** pre-norm, fp32 statistics
- **RoPE** with a **NoPE hybrid** — no rotary on every 4th layer (SmolLM3 ablation)
- **Grouped-query attention**, 4 query heads per KV head — 4× smaller cache than MHA
  at indistinguishable quality
- **Multi-head latent attention (MLA)** as a switchable alternative — DeepSeek-V2
  style low-rank KV with decoupled RoPE, 0.14× the cache of MHA
- **QK-norm** — the single highest-leverage stability fix for bf16 runs
- **SwiGLU**, no biases, tied embeddings
- **Mixture of experts** with fine-grained experts, a shared expert, and
  **auxiliary-loss-free load balancing** (DeepSeek-V3's per-expert bias, steered
  by a rule rather than by a gradient that fights the loss)
- **Multi-token prediction** — denser training signal, and self-speculative
  decoding for free at inference
- **Intra-document masking** so packed documents never attend across each other
- **YaRN / NTK / linear** RoPE scaling for post-hoc context extension
- **muP** so hyperparameters tuned on a narrow proxy transfer to full width -
  verified by a coordinate check: activations drift 8.4x across d=128..1024 under
  standard parameterisation and 1.03x under muP (`test_mup_coordinate_check`)

### Training

- **Muon + AdamW hybrid.** Muon orthogonalises the momentum matrix by
  Newton–Schulz and applies it to the 2D hidden weights; embeddings, norms, the
  LM head and routers stay on AdamW. Reported at ~2× the compute efficiency of
  AdamW at scale — but the literature disagrees under well-tuned baselines, so
  `A1` measures it here rather than assuming it. `optim.kind=adamw` is one flag.
- **WSD schedule**, not cosine. Cosine needs the total step count up front; with
  an unknown budget that is the wrong shape. WSD stays flat through the middle
  and decays over the last ~20%, so a run can be ended *at any step* with a
  proper anneal — see "Finishing early" below.
- bf16 autocast, gradient accumulation to a fixed token budget, DDP/FSDP,
  `torch.compile`, activation checkpointing, MFU and tokens/sec telemetry.
- Optional **logit distillation** from a larger same-tokenizer teacher.
- **Staged data curriculum**: the mixture shifts toward code, maths and
  instruction-like text as the run progresses, with the highest-quality stage
  deliberately coinciding with the WSD decay.

---

## How the resumability actually works

Three properties, each enforced by a test.

**1. The data order is a pure function, not stored state.**
Sequence *k* of epoch *e* is `permute(k, n_sequences, epoch_seed(seed, e))`, where
`permute` is a keyed [Feistel network](slm/data/permute.py) over the index space
with cycle walking. It is a bijection by construction, costs O(1) memory at any
dataset size, and is identical on every machine. So the resume state is **one
integer**: how many sequences the job has consumed.

That integer is *global*, not per-rank, which is why the world size can change
across a restart — the ranks re-slice the same stream.

**2. No partial checkpoint is ever loadable.**
Checkpoints are written to `step_XXXXXXXX.tmp/`, fsynced, then renamed. A
directory named `step_00012000` therefore always means complete. Every file
carries a sha256 in `manifest.json`; a checkpoint that failed a flaky S3 download
is rejected on load, and the loader falls back to the previous one — costing one
interval instead of the run.

**3. Preemption is a normal exit path.**
SIGTERM, SIGINT, a cloud spot-reclaim notice, a wall-clock deadline or a `STOP`
file all set one flag. The loop finishes its step, checkpoints, writes `DONE` if
the budget was reached, and exits. `scripts/supervise.sh` relaunches until `DONE`
appears — so "my pod was reclaimed" becomes "start a pod somewhere else and run
the same command".

The checkpoint holds model, optimiser moments, the data cursor, all four RNG
streams, the config, and the git SHA. Weights are saved unwrapped and on CPU, so
a checkpoint written by 8×H100 FSDP loads on one 4090, on Apple MPS, or on CPU.

### Moving providers mid-run

```bash
# on the machine that is about to die - nothing to do, it already mirrors:
ckpt.remote=s3://bucket/run-1     # or hf://user/repo, rsync://host:/path, or a path

# on the new machine, anywhere:
python scripts/train.py configs/train/pretrain-300m.yaml ckpt.remote=s3://bucket/run-1
```

The second command finds no local checkpoint, pulls the newest verified one from
the mirror, and continues at the exact step.

### Finishing early

Compute running out with 40% of the budget unspent? Do not just stop — a model
abandoned at a high learning rate is measurably worse than one trained for the
shorter budget properly.

```bash
echo 2000 > runs/slm-300m/FINISH
```

The trainer notices, re-points `max_steps` so the WSD decay starts immediately,
anneals over 2000 steps, and exits with a finished model. This is the schedule
choice paying for itself.

---

## Evaluation

The eval harness is built and **validated before** any real training run, because
a harness that silently mis-scores makes every later comparison meaningless:

- scores match a hand-computed `log_softmax` to 1e-4 (`test_score_matches_a_hand_computed_logprob`)
- batching and reordering do not change a single score — the classic harness bug
  (`test_batching_does_not_change_scores`)
- a model trained to memorise a corpus is scored correctly by the harness
  (`test_a_model_that_memorised_the_corpus_scores_the_right_choice`)

Tasks: HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande, OpenBookQA, LAMBADA, in
cloze/continuation form (small models cannot follow a multiple-choice format
until late in training). Reported as `acc`, `acc_norm` and per-token accuracy,
always against the random baseline, plus **bits-per-byte** — the only quality
number comparable across tokenizers.

```bash
python scripts/evaluate.py runs/slm-300m/checkpoints/step_00114000 -t data/tokenizer.json
python scripts/benchmark.py            # tokens/sec and MFU for each config
```

---

## Repo layout

```
slm/
  config.py            typed config: dataclasses < YAML < CLI overrides
  tokenizer/bpe.py     byte-level BPE, trained from scratch (12 MB/s encode)
  data/
    shards.py          uint16 token shards behind a page-aligned header
    permute.py         Feistel bijection - O(1)-memory deterministic shuffling
    loader.py          resumable, world-size-agnostic, mixture-aware loader
    prepare.py         corpus -> shards, multiprocess
  model/
    transformer.py     blocks, MTP heads, the model
    attention.py       grouped-query and multi-head latent attention
    moe.py             experts + loss-free load balancing
    masking.py         causal / sliding-window / intra-document, flex or SDPA
    rope.py            RoPE with linear, NTK and YaRN scaling
  train/
    trainer.py         the loop
    checkpoint.py      atomic, verified, mirrored checkpoints
    muon.py            Muon optimiser (Newton-Schulz orthogonalisation)
    preempt.py         signals, spot notices, deadlines -> one flag
    distill.py         logit distillation from a teacher
  eval/                loglikelihood harness and task loaders
  cli.py               console entry points (slm-train / slm-eval / slm-generate)
  infer/generate.py    KV-cache sampling, speculative decoding
  sft/                 chat templating with prompt-masked loss
docs/                  RESEARCH.md, ARCHITECTURE.md, ABLATIONS.md, OPERATIONS.md
```

2,700 lines of tests across 241 cases, including a differential test of the BPE
trainer against a deliberately naive reference implementation, KV-cache/full-forward
equivalence, and Newton–Schulz convergence.

---

## Ablations

Every architectural claim above is a hypothesis with a matching experiment on the
100M proxy at 2B tokens, one config key changed at a time:

```bash
python scripts/run_ablation.py --list      # A1 through A10
python scripts/run_ablation.py --all
python scripts/run_ablation.py --report    # loss deltas vs baseline
```

See [docs/ABLATIONS.md](docs/ABLATIONS.md) for what each one is asking and what
result would change the design.

---

## Status

**Complete and tested:** tokenizer, data pipeline, model (dense + MoE + MLA +
MTP), trainer (Muon/AdamW, DDP/FSDP, distillation), checkpointing and
cross-provider resume, evaluation harness, inference with speculative decoding,
SFT. 241 tests green, lint clean; the full pipeline verified end to end on CPU.

**Not yet done:** the flagship runs. The infrastructure is finished and the
budget is costed (~93 H100-hours for 60B tokens at 300M — see
[docs/ARCHITECTURE.md §5](docs/ARCHITECTURE.md)), but no large-scale numbers are
claimed here yet, and none will be until the runs and their ablations exist.
Benchmark tables get filled in from `scripts/evaluate.py` output, not from hope.

---

## Reading order

1. [docs/RESEARCH.md](docs/RESEARCH.md) — the evidence, with citations, gathered
   before any design was fixed
2. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the decisions, and what each one
   beat
3. [docs/ABLATIONS.md](docs/ABLATIONS.md) — the experiments that test them
4. [docs/OPERATIONS.md](docs/OPERATIONS.md) — running this on rented, unreliable GPUs
5. [docs/RESULTS.md](docs/RESULTS.md) — the numbers, filled in as runs finish
