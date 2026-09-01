# Architecture

Design decisions and the evidence behind them. `R#` references are findings in
[RESEARCH.md](RESEARCH.md); `A#` are ablations in [ABLATIONS.md](ABLATIONS.md)
that test a decision rather than assuming it.

The governing constraint: **200-500M parameters, unpredictable compute budget,
must survive losing the machine at any moment.** Every choice below is downstream
of one of those three.

---

## 1. Model family

Three configs, one codebase. Sizes are exact, verified against the built module
by `test_param_count_matches_analytic_formula`.

| Config | Params | Active | d | L | heads (q/kv) | FFN | Purpose |
|---|---|---|---|---|---|---|---|
| `slm-100m` | 100M | 100M | 768 | 12 | 12 / 3 | 2048 | ablation proxy (A1-A10) |
| `slm-300m` | 304M | 304M | 1024 | 24 | 16 / 4 | 2816 | **dense flagship** |
| `slm-moe-500m` | 528M | 131M | 768 | 24 | 12 / 3 | 32 experts x 256, top-4 + 1 shared | **sparse flagship** |

The two flagships sit at the same total-parameter budget and answer different
questions: the dense one is what you deploy when memory is scarce, the sparse one
is what you deploy when FLOPs are scarce. Publishing both from one codebase is
the point (R5).

### Why these shapes

- **Aspect ratio.** `d/L` = 42 for the 300M. Kaplan et al. found loss is
  remarkably flat in aspect ratio between ~30 and ~100; staying near the low end
  buys depth, which helps reasoning-shaped evals at fixed parameters.
- **`d_head = 64` everywhere.** Tensor cores want multiples of 64, and the
  flash-attention kernels have their fastest paths at 64 and 128.
- **`n_kv_head = n_head / 4`.** SmolLM3 measured GQA-2/4/8 as indistinguishable
  from MHA, and MQA as clearly worse (R3). 4 is the middle of the safe range.
- **FFN width `8d/3` rounded to 128.** Keeps SwiGLU's three matrices at the same
  parameter cost as a 4d two-matrix GELU block (Shazeer 2020).

## 2. Component decisions

| Component | Choice | Why | Rejected |
|---|---|---|---|
| Norm | RMSNorm, pre-norm, fp32 statistic | cheaper than LayerNorm, no measured loss | LayerNorm; post-norm |
| Position | RoPE, **NoPE every 4th layer** | SmolLM3 ablation: equal short-context, better long (R9) | learned; ALiBi; pure RoPE |
| Attention | **GQA-4**, MLA switchable | GQA-4 ≈ MHA at 4x smaller cache; MLA claim gets tested not assumed (R3) | MHA (cache); MQA (quality) |
| Attn stability | **QK-norm** | bounds `q·k` so bf16 runs cannot drift into divergence | logit softcap (needs eager attention) |
| FFN | SwiGLU | standard, parameter-matched | GELU MLP; ReLU² |
| Doc handling | **intra-document masking** | free now, required for long context later (R9) | plain causal packing |
| Embeddings | **tied** | SmolLM3: 1.2B tied ≈ 1.46B untied, 18% fewer params | untied |
| Vocab | **32,768**, trained here | 128k would be 33% of a 300M model (R2) | 128k (Llama-3); 50k (GPT-2) |
| Objective | CE + **MTP depth 1** + z-loss | denser signal, and free speculative decoding (R6) | CE alone |
| Bias terms | none | no measured benefit, fewer things to blow up | biases everywhere |
| Parameterisation | **muP**, optional | HP tuned on a 100M proxy transfers to 300M; verified by a coordinate check (8.4x activation drift across widths without it, 1.03x with) | standard parameterisation and re-tuning per width |

## 3. Optimiser

**Hybrid Muon + AdamW** (R4).

- **Muon** on every 2D hidden weight: attention and FFN matrices. Momentum is
  orthogonalised with a 5-step Newton-Schulz iteration in bf16 before the update,
  which equalises the singular values of the step and stops a few directions from
  dominating.
- **AdamW** on everything Muon cannot handle: embeddings, the LM head, all norm
  gains, and router weights. These are not "hidden matrices" and Muon degrades on
  them.
- Reported gains are 1.3-2x token efficiency, but under well-tuned AdamW the
  margin narrows (R4). **A1 decides it on our own data**, and `--optim.kind=adamw`
  is a single flag away.

Schedule: **WSD (warmup-stable-decay)**, not cosine. Cosine has to know the total
step count in advance; when the budget is unknown, that is the wrong shape. WSD
is flat through the middle and decays over the final ~20%, so:

- the run can be ended at *any* step with a short proper anneal,
- the stable-phase weights are a legitimate branch point for several finals,
- extending a run is just "keep going", with no LR discontinuity.

This is the schedule-level expression of the same constraint that drives the
checkpointing design.

## 4. Data

Ablation mix, fixed across all of A1-A10 so architecture results are not
confounded (R8):

```
FineWeb-Edu        70%   general web, educational-quality filtered
Stack-Edu-Python   20%   code
FineMath-3+        10%   mathematics
```

Main run, three stages (R8, SmolLM3-style):

| Stage | Share of budget | Mix | Intent |
|---|---|---|---|
| 1 — breadth | 0-70% | 75 web / 15 code / 10 math | general competence |
| 2 — density | 70-90% | 55 web / 25 code / 20 math | shift toward reasoning-dense text |
| 3 — anneal | 90-100% | 40 web / 25 code / 20 math / 15 instruction-like | coincides with the WSD decay |

Stage 3 deliberately coincides with the LR decay: the highest-quality data is
seen when the learning rate is smallest and the model is least likely to forget.

This is implemented, not aspirational — `data.curriculum` in the flagship configs,
switched by `Trainer._apply_curriculum`, tested by
`test_curriculum_switches_the_mixture_mid_run`. `train_dir` holds one
subdirectory per domain and `MixtureLoader` interleaves them with a golden-ratio
low-discrepancy sequence, so the realised ratios are exact and the whole thing
stays resumable from a single counter.

## 5. Token budget

Chinchilla-optimal for 300M is 6B tokens. That is the floor, not the target (R1).

| Milestone | Tokens | Tokens/param | H100-hours (est.) |
|---|---|---|---|
| Smoke | 0.1B | 0.3 | 0.2 |
| Ablation run (100M proxy) | 2B | 20 | 1.5 |
| Chinchilla 300M | 6B | 20 | 9 |
| **Target 300M** | **60B** | **200** | **~93** |
| Stretch | 180B | 600 | ~280 |

Estimate: 2.2 GFLOP/token forward+backward at 300M/2k context, 40% MFU on an
H100 at 989 TFLOP/s bf16 peak. 60B tokens ≈ 93 GPU-hours ≈ 12 hours on 8xH100.
The whole flagship run is a day of rented compute — which is exactly why the
interesting engineering is in surviving interruption, not in raw scale.

## 6. Infrastructure invariants

These are properties the test suite enforces, not aspirations.

1. **Exact resume.** A run killed at step N and restarted on different hardware
   with a different GPU count sees the same tokens in the same order.
   → `test_resume_is_bit_exact`, `test_resume_at_a_different_world_size`
2. **No partial checkpoint is ever loadable.** Atomic rename, sha256 manifest,
   fall back to the previous checkpoint on corruption.
   → `test_corruption_is_detected`, `test_latest_falls_back_past_a_corrupt_checkpoint`
3. **The data order is a pure function**, not stored state — a Feistel bijection
   over the index space, so resume state is one integer at any dataset size.
   → `test_is_a_bijection`, `test_slices_are_consistent`
4. **Preemption is a normal exit path.** SIGTERM/SIGINT/cloud-notice → finish the
   step, checkpoint, exit 0. A supervisor relaunches; the run continues.
5. **The mirror is the source of truth.** Every checkpoint is pushed to a remote
   store, so "the pod is gone" means "start a pod elsewhere, same command".
6. **No background thread outlives its owner.** Loader prefetch, the checkpoint
   writer and the preemption watcher are all joined on shutdown - a training
   process that will not exit is an outage on a metered instance.

## 7. Deliverables

1. `slm-300m` and `slm-moe-500m` weights + full training curves.
2. Ablation table A1-A10 with loss deltas and eval deltas, on the 100M proxy.
3. Eval harness validated against a published model's reported numbers *before*
   the main run (R10.1).
4. A tokenizer trained from scratch, with its compression measured against
   GPT-2/Llama-3 tokenizers on our corpus.
5. Inference: KV-cache generation, MTP self-speculative decoding, throughput
   numbers.
6. An SFT stage producing a chat model, and a written account of what failed.

## 8. Explicit non-goals

- **Not competing on benchmarks with 7B models.** A 300M model that claims MMLU
  parity with Llama-3-8B has a broken eval harness, and that is the first thing a
  reviewer will check.
- **No novel architecture claims.** Every component here is from a paper. The
  contribution is the *system*: reproducible, resumable, ablated, measured.
- **No RLHF.** Out of scope at this budget; SFT is the end of the chain.
