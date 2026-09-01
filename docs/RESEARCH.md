# Research notes

Evidence gathered before the architecture was fixed. Every design decision in
[ARCHITECTURE.md](ARCHITECTURE.md) points back to a numbered finding here.

---

## R1 — Token budget: ignore Chinchilla, train past it

Chinchilla (Hoffmann et al. 2022) says ~20 tokens/parameter is *compute*-optimal.
Nobody ships that any more, because it optimises the wrong thing: it minimises
training cost while ignoring that the model is then served millions of times.

| Model | Params | Tokens | Tokens/param |
|---|---|---|---|
| Chinchilla-optimal 300M | 300M | 6B | 20 |
| Llama-2-7B | 7B | 2T | 290 |
| Gemma-7B | 7B | 6T | 857 |
| Llama-3-8B | 8B | 15T | 1875 |
| SmolLM3-3B | 3B | 11T | 3667 |

Sardana et al. 2024 formalises it: once inference demand is counted, the optimum
moves toward smaller models trained far longer. A 300M model trained on 60B
tokens (200 tok/param) beats a 600M model trained on 12B at equal training cost
*and* costs half as much to serve.

**Taken:** target 200 tok/param (60B), treat 20 tok/param (6B) as the floor a
run must clear to be worth reporting.

## R2 — Vocabulary size is a parameter-budget decision, not a free choice

At `d_model = 1024`, the embedding matrix costs `vocab x 1024` parameters:

| Vocab | Embedding params | Share of a ~300M model |
|---|---|---|
| 32,768 | 33.6M | 11% |
| 49,152 | 50.3M | 16% |
| 128,256 (Llama-3 / SmolLM3) | 131.3M | **33%** |

SmolLM3 can afford a 128k vocab because it is 3B parameters and multilingual.
Copying it at 300M would spend a third of the model on an embedding table that
does almost no reasoning. The counter-pressure is compression: a bigger vocab
means fewer tokens per document, so a fixed token budget covers more text.

**Taken:** 32,768, trained on our own corpus. Treated as an ablation, not an
assumption — see A2.

## R3 — Attention: GQA is enough at this scale; MLA is the interesting bet

- SmolLM3 ablations: GQA with 2/4/8 groups "roughly match MHA"; MQA (1 KV head)
  underperforms clearly.
- DeepSeek-V2 ablations: GQA looked *worse* than MHA; MLA matched or beat MHA
  while compressing the KV cache much further.
- TransMLA (2025): any GQA model can be rewritten as an MLA model, and the
  expressive power of MLA strictly contains GQA's.
- "Latent Multi-Head Attention for Small Language Models" (2025) finds MLA+RoPE
  the best quality/cache tradeoff at small scale.

KV-cache arithmetic per token per layer at `d=1024, n_head=16, d_head=64`:

| Scheme | Cached floats | Relative |
|---|---|---|
| MHA | 2048 | 1.00x |
| GQA-4 | 512 | 0.25x |
| MLA (`kv_lora=256, rope=32`) | 288 | 0.14x |

**Taken:** GQA-4 as the default (proven, fast, simple); MLA implemented as a
first-class switchable option so the claim can be *tested* rather than repeated.

## R4 — Optimiser: Muon is the largest single efficiency win available

- Moonshot / Kimi K2 (arXiv 2502.16982): Muon reaches ~2x the computational
  efficiency of AdamW at compute-optimal training, and holds data efficiency far
  beyond the critical batch size.
- Practical Efficiency of Muon (arXiv 2505.02222): confirms the Pareto
  improvement; the win grows with batch size.
- Independent reports: 30-40% fewer tokens to a target loss on small models.
- Caveat — "Fantastic Pretraining Optimizers and Where to Find Them"
  (2509.02046): gains shrink under properly tuned baselines, and AdEMaMix/Mars
  are competitive. So this must be measured here, not assumed.

Muon orthogonalises the momentum matrix via Newton-Schulz and applies it only to
2D hidden weights; embeddings, norms and the LM head stay on AdamW.

**Taken:** hybrid Muon+AdamW as the default, with a one-flag fallback to pure
AdamW, and A1 as the ablation that decides it.

## R5 — MoE gets dense quality for ~1/3 the FLOPs

OLMoE (Ai2, 2024): 7B total / 1B active, trained on 5T tokens, beats all models
of comparable *active* size and matches Llama2-13B-Chat. Their headline: the MoE
matched the dense model with ~3x fewer FLOPs, and trained 2x faster in wall
clock.

Modern practice (DeepSeek-V3, Qwen3) converges on: fine-grained experts, 1+
always-on shared expert, top-k routing, and **auxiliary-loss-free load
balancing** — a per-expert bias nudged toward balance outside the gradient, which
avoids the aux-loss-vs-quality tug of war.

**Taken:** a second flagship, `slm-500m-a130m`, at the same total parameter
budget as the dense 500M. Same infrastructure, two Pareto points.

## R6 — Multi-token prediction: densifies the signal and pays twice

DeepSeek-V3 introduced MTP: lightweight extra heads predict tokens t+2, t+3.
Ablations since (Nemotron 3, MiniMax-M2) report consistent gains on validation
loss and downstream benchmarks, largest on reasoning-heavy tasks. At inference
the same heads give self-speculative decoding: 80-90% acceptance on the second
token, ~1.8x throughput.

**Taken:** MTP depth 1 by default. One extra block, gains on both axes.

## R7 — Distillation is why the best small models are good

Gemma-2 2B and 9B were trained by distillation from a larger teacher rather than
plain next-token prediction, and Google attributes their "competitive with models
2-3x bigger" result to it. The mechanism is a richer gradient: a full soft
distribution per position instead of a one-hot target.

**Taken:** optional logit-distillation mode in the trainer. Honest framing — a
distilled model is not "trained from scratch" in the same sense, so it is
reported as a separate track.

## R8 — Data quality beats architecture at this scale

- FineWeb-Edu: 1.3T tokens kept by an educational-quality classifier.
- DCLM: 3.8T tokens via a fastText classifier.
- Nemotron-CC (2024): combines classifiers to raise *recall*, because
  FineWeb-Edu/DCLM discard ~90% of data and run out on long horizons. 6.3T
  tokens; +5.6 MMLU over DCLM at 8B/1T.
- Ultra-FineWeb (2025): ~1T tokens, beats FineWeb-Edu for small models trained
  from scratch.

SmolLM3's ablation mix — **70% FineWeb-Edu, 20% Stack-Edu-Python, 10%
FineMath-3+** — is a well-tested starting point that gives early signal across
knowledge, code and math.

**Taken:** that mix for ablations; a three-stage curriculum for the main run.

## R9 — Positional encoding and document masking

SmolLM3 ablations: removing RoPE from every 4th layer (NoPE hybrid) performed on
par with full RoPE at short context and improved long-context behaviour.
Intra-document masking gave identical short-context curves but became "crucial
when scaling to long sequences".

**Taken:** NoPE every 4th layer, and document masking on from the start — it
costs nothing measurable now and is required later.

## R10 — Process findings that matter more than any of the above

From the Smol Training Playbook, stated bluntly:

1. **Validate the eval harness before training anything.** Reproduce published
   numbers for a released model first, or every later comparison is noise.
2. **Ablations and debugging cost more than the main run.** SmolLM3: ~438k GPU
   hours total, of which the main run was 276k. Budget accordingly.
3. **Change one thing at a time.** Their worst debugging episode traced to an
   unrelated dependency update days earlier.
4. **Never switch frameworks between ablation and final run.**
5. Use **cloze/likelihood** scoring for early evals; small models cannot follow
   a multiple-choice format until late in training.

**Taken:** the eval harness is built and validated *before* the main run;
ablations run on a fixed 100M proxy; one config knob per ablation; everything
version-pinned and checkpoint-recorded with the git SHA.

---

## Sources

- [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Beyond Chinchilla-Optimal: Accounting for Inference](https://arxiv.org/pdf/2401.00448)
- [The Smol Training Playbook](https://huggingfacetb-smol-training-playbook.hf.space/)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [Practical Efficiency of Muon for Pretraining](https://arxiv.org/html/2505.02222v1)
- [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/html/2509.02046v2)
- [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
- [TransMLA: Multi-Head Latent Attention Is All You Need](https://arxiv.org/pdf/2502.07864)
- [Latent Multi-Head Attention for Small Language Models](https://arxiv.org/html/2506.09342v1)
- [Nemotron-CC](https://arxiv.org/abs/2412.02595)
- [Gemma 2 Technical Report](https://arxiv.org/html/2408.00118v1)
- [Olmo 3](https://allenai.org/blog/olmo3)
