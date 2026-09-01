# Results

Filled from `scripts/evaluate.py` and `scripts/benchmark.py` output. Nothing is
written here until the run that produced it has finished.

## Status

| Run | Tokens | Status |
|---|---|---|
| Ablation grid (A1–A10, 100M proxy) | 2B each | not started |
| `slm-300m` | 60B | not started |
| `slm-moe-500m` | 60B | not started |
| `slm-300m-sft` | — | not started |

The infrastructure is complete and verified end to end; the runs are waiting on
compute. See [ARCHITECTURE.md §5](ARCHITECTURE.md) for the costing.

## Throughput

`python scripts/benchmark.py` — measured, not estimated.

| Config | Device | Params (active) | Batch × seq | ms/step | ktok/s | MFU |
|---|---|---|---|---|---|---|
| debug | Apple M4 (MPS) | 1.3M | 8 × 256 | 17.9 | 114.4 | n/a |
| slm-100m | Apple M4 (MPS) | 99.5M | 1 × 256 | 911.1 | 0.3 | n/a |

MPS numbers are a smoke check that the config path works, not a performance
claim — MFU is only computed where the device's peak bf16 throughput is known.

Fill in `slm-300m` and `slm-moe-500m` on the training hardware before the real
run — a run started at 15% MFU wastes more money than every other decision here
combined.

## Measured engineering findings

Things learned by running the code, not by reading about it.

### bf16 matmul is 434× slower than fp32 on CPU

Muon's Newton–Schulz iteration is five matmuls per parameter per step. The
reference implementations run it in bf16, which is correct on CUDA and
catastrophic everywhere else:

| Device | dtype | 1024×1024, 5 iterations |
|---|---|---|
| CPU | bfloat16 | 21,597 ms |
| CPU | float32 | 50 ms |
| MPS | bfloat16 | 341 ms |
| MPS | float32 | 16 ms |

`ns_dtype()` now picks bf16 only on CUDA. Guarded by
`test_newton_schulz_is_not_pathologically_slow_on_cpu`, because the unit tests
use small matrices where the difference is invisible — it only showed up when
benchmarking a real config.

### muP coordinate check

Mean |activation| in the residual stream at the last block, fixed input, 4 layers:

| d_model | 128 | 256 | 512 | 1024 | spread |
|---|---|---|---|---|---|
| standard | 0.0248 | 0.0472 | 0.0916 | 0.2077 | **8.37×** |
| muP | 0.0229 | 0.0234 | 0.0229 | 0.0235 | **1.03×** |

This is what makes hyperparameter transfer valid: tune the LR on the 100M proxy,
use it at 300M without re-tuning.

### Newton–Schulz convergence depends on the input spectrum

5 iterations is enough for a Gaussian matrix (singular values land in
[0.67, 1.06]) but not for a condition-number-1000 matrix, where the smallest
singular value only reaches 0.12 at 5 steps and 0.68 at 7. Real gradients look
Gaussian, so the default of 5 stands — but `ns_steps` is a knob for a reason.

### Tokenizer

| Corpus | Vocab | Train time | Encode | Compression |
|---|---|---|---|---|
| TinyShakespeare (1.1 MB) | 4,096 | 0.3 s | 12.4 MB/s | 3.59 bytes/token |

Pure Python, made viable by the per-word encoding cache and the word-frequency
trainer. The merges are byte-identical to a naive reference implementation
(`test_fast_trainer_matches_naive_trainer`).

## Pretraining

| Model | Tokens | Val loss | Bits/byte | HellaSwag | ARC-e | ARC-c | PIQA | WinoGrande | OBQA | LAMBADA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `slm-300m` | — | — | — | — | — | — | — | — | — | — | — |
| `slm-moe-500m` | — | — | — | — | — | — | — | — | — | — | — |
| random baseline | — | — | — | 25.0 | 25.0 | 25.0 | 50.0 | 50.0 | 25.0 | 0.0 | — |

Reported as `acc_norm` (length-normalised) at temperature 0, cloze formulation.
Every number comes from `scripts/evaluate.py`, whose scorer is verified against a
hand-computed `log_softmax` — see the README's Evaluation section.

### Reference points

Included so the numbers can be read honestly, not to claim parity. These are
published figures for models trained on 10–100x more tokens:

| Model | Params | Tokens | HellaSwag | ARC-e | PIQA |
|---|---|---|---|---|---|
| GPT-2 (124M) | 124M | ~10B | 31.1 | 43.8 | 62.9 |
| Pythia-410M | 410M | 300B | 40.9 | 52.1 | 67.1 |
| SmolLM2-360M | 360M | 4T | 54.5 | 70.3 | 71.6 |

A 300M model trained on 60B tokens should land between GPT-2 and Pythia-410M. If
it lands above SmolLM2-360M, the eval harness is wrong — check that first.

## Ablations

See [ABLATIONS.md](ABLATIONS.md) for the protocol and the empty results table.

## Inference

| Model | Device | Batch | Tokens/s | KV cache/token | Speculative speedup |
|---|---|---|---|---|---|
| `slm-300m` | — | — | — | — | — |

Speculative decoding uses the model's own MTP heads; the acceptance rate is
printed by `scripts/generate.py --speculative`.
