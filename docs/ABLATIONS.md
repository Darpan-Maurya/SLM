# Ablations

Every architectural claim in [ARCHITECTURE.md](ARCHITECTURE.md) is a hypothesis.
This is the experiment that tests it.

## Protocol

Fixed across all runs, so only one thing varies:

| | |
|---|---|
| Model | `slm-100m` (12L, d=768, GQA-4) |
| Tokens | 2B (20 tok/param — enough for stable signal) |
| Data | 70% FineWeb-Edu / 20% Stack-Edu-Python / 10% FineMath-3+ |
| Batch | 0.5M tokens/step, seq 2048 |
| Optimiser | Muon+AdamW, WSD, warmup 400, lr 1e-3 |
| Seed | 1337, and the data order is a pure function of it |

Rules, taken from the Smol Training Playbook's list of things that went wrong:

1. **One key changes per run.** `run_ablation.py` enforces this — each entry is a
   name and a list of overrides applied to one shared base config.
2. **Never switch frameworks between ablation and final run.** The ablation and
   the flagship use the same `scripts/train.py`.
3. **The eval harness is validated first.** Done — see the README's Evaluation
   section for the three tests that pin it down.
4. **Report the loss delta and the eval delta.** A change that improves loss but
   not downstream evals at 2B tokens is not yet evidence of anything.
5. **Budget for it.** SmolLM3 spent more compute on ablations and debugging than
   on the main run. Twelve runs at 2B tokens on the 100M proxy is ~18 H100-hours
   in total — cheap insurance against a wrong flagship.

```bash
python scripts/run_ablation.py --list
python scripts/run_ablation.py A1_adamw            # one experiment
python scripts/run_ablation.py --all               # the grid
python scripts/run_ablation.py --report            # deltas vs baseline
```

## The experiments

| ID | Question | Change | What would change the design |
|---|---|---|---|
| **A1** | Is Muon actually better than AdamW here? | `optim.kind=adamw` | If AdamW matches within noise, drop Muon — it adds a Newton–Schulz iteration per step and a second optimiser to checkpoint. The literature is split (R4), so this one is genuinely open. |
| **A2** | Is a 32k vocab the right budget split? | `model.vocab_size=49152` | 49k costs 12.6M more parameters at d=768 and buys better compression. If bits-per-byte improves more than the parameters cost in loss, move up. |
| **A3a** | Does GQA-4 lose anything vs MHA? | `model.n_kv_head=12` | SmolLM3 says no (R3). If MHA wins clearly here, the KV-cache saving is not free and GQA-8 is the compromise. |
| **A3b** | Is MLA worth its complexity at 300M? | `attn_type=mla` | DeepSeek-V2 says MLA ≥ MHA with a much smaller cache (R3). If it matches GQA on loss, MLA becomes the default on cache grounds alone. If it trains slower for equal loss, it stays optional. |
| **A4** | Does multi-token prediction help at this scale? | `model.mtp_depth=1` | Reported gains are from much larger models (R6). If it does not help loss at 100M, keep it anyway *only* if self-speculative decoding measures a real throughput win. |
| **A5** | Is the NoPE hybrid safe at short context? | `model.nope_every=0` | SmolLM3 found parity at short context and gains at long (R9). If NoPE hurts here, revert to full RoPE — the long-context benefit is not yet being measured at 2k. |
| **A6** | Does QK-norm cost anything? | `model.qk_norm=false` | Expected: no quality cost, large stability benefit. If QK-norm *hurts* loss measurably, it becomes a bf16-only safety switch rather than a default. |
| **A7** | Does sparsity beat density at matched active FLOPs? | 16 experts, top-2 | OLMoE reports ~3× FLOP efficiency (R5). This is the headline test for `slm-moe-500m`. Matched active FLOPs, not matched parameters — that is the comparison that means something. |
| **A8** | Does intra-document masking matter at 2k? | `model.doc_masking=false` | Expected: no measurable difference at short context (R9). Kept regardless because it is required for long context, but if it costs throughput with zero gain, gate it behind sequence length. |
| **A9** | Is WSD as good as cosine at a *known* budget? | `optim.schedule=cosine` | WSD is chosen for budget flexibility, not for loss. If cosine is meaningfully better at a fixed budget, that is the price of flexibility and it should be stated, not hidden. |
| **A10** | Does the logit z-loss cost quality? | `optim.zloss=0.0` | Expected: negligible loss cost, real stability benefit at bf16. If it costs nothing and prevents nothing at this scale, it is cargo cult and should go. |

## Reporting

Each run writes `runs/ablation-<name>/logs/metrics.jsonl`. `--report` reads the
final train and val loss and prints the delta against the baseline. The full
table also needs the downstream evals:

```bash
for r in runs/ablation-*/checkpoints/step_*; do
  python scripts/evaluate.py "$r" -t data/tokenizer.json --limit 500 \
      -o "${r%/*}/../eval.json"
done
```

Results go here as a table with, for each ablation: final val loss, bits-per-byte,
the six-task average, and wall-clock tokens/sec. A result is only reported once
the run finished — no partial curves, no "trending toward".

## Results

*Empty until the runs exist. Filled from `scripts/evaluate.py` output.*

| ID | val loss | Δ | bpb | 6-task avg | tok/s | verdict |
|---|---|---|---|---|---|---|
| baseline | — | — | — | — | — | — |
