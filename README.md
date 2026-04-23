# mathnet-eval-harness

> _TODO elevator pitch: one-sentence framing of what this repo shows and why it matters._

## Headline results

_Smoke-test row in; full 500-problem frontier comparison + QLoRA row land in Week 4._

### Frontier model comparison (cumulative)

| Model | Params | MathNet accuracy | Cost / 1K problems | Run |
|---|---|---|---|---|
| Claude Sonnet 4.6 *(smoke, n=20)* | — | 70.0% (14/20) | $0.50 total ($25/1K extrapolated) | [results/smoke/sonnet-4-6/](results/smoke/sonnet-4-6/) |
| Claude Sonnet 4.6 *(n=500)* | — | _TBD_ | _TBD_ | _Day 2_ |
| Claude Opus 4.7 *(n=500)* | — | _TBD_ | _TBD_ | _Day 2_ |
| GPT-5 *(n=500)* | — | _TBD_ | _TBD_ | _Day 2_ |
| Gemini 3 Pro *(n=500)* | — | _TBD_ | _TBD_ | _Day 2_ |
| Qwen-2.5-1.5B-Instruct *(base)* | 1.5B | _TBD_ | — | _Day 3_ |
| Qwen-2.5-1.5B-Instruct + QLoRA *(ours)* | 1.5B | _TBD_ | — | _Week 4_ |

### Grader path breakdown (Sonnet smoke, n=20)

| Path | Count | Share |
|---|---|---|
| `exact` | 4 | 20% |
| `normalized` | 2 | 10% |
| `symbolic` (sympy) | 1 | 5% |
| `judge` (LLM) | 7 | 35% |
| `miss` | 6 | 30% |

7/14 = 50% of correct grades resolve on the objective layers (exact / normalized / sympy); the judge handles the rest. 1/6 misses was a `max_tokens` truncation (now fixed by bumping the default to 8192).

## Key findings

_TODO: 3–5 bullets of the most interesting things we learned — e.g. where the fine-tuned small model wins, where it fails, what categories of problems are hardest, what the cost/accuracy Pareto looks like._

## Architecture

_TODO diagram or ASCII sketch:_ data loading → stratified split → (a) frontier-model eval via API → raw responses on disk → grading; (b) QLoRA training on Hyak → adapter checkpoint → local/Hyak inference → grading. Grading and analysis are shared.

## Reproducing

```bash
# 1. Clone and install
git clone https://github.com/sanmarcog/mathnet-eval-harness.git
cd mathnet-eval-harness
pip install -e .

# 2. Configure secrets
cp .env.example .env
# edit .env with your API keys

# 3. Build the eval / train splits
python scripts/build_splits.py --out data/splits

# 4. Run frontier-model eval (example: Claude Sonnet 4.6 on 20 problems)
python scripts/run_eval.py --model sonnet-4-6 --split eval --n 20

# 5. Train QLoRA adapter on Hyak
sbatch slurm/train_qlora.sbatch
```

_TODO: fill in once scripts stabilize._

## Structure

```
src/mathnet_eval/     # core library (importable)
  data.py             # MathNet loading, stratified splits, prompt formatting
  inference.py        # unified client for Claude / OpenAI / Gemini / local HF
  grading.py          # answer extraction + correctness
  training.py         # QLoRA training loop (TRL SFTTrainer)
  analysis.py         # aggregate results, plots, cost/accuracy tables
scripts/              # thin CLI entrypoints (argparse → library calls)
slurm/                # sbatch scripts for Hyak (ckpt-all partition)
notebooks/            # exploration, figure drafts
results/              # committed JSON outputs and figures
tests/                # pytest unit tests
docs/                 # notes, design docs
```

## Tech stack

- **Base model**: Qwen-2.5-1.5B-Instruct (HuggingFace)
- **Fine-tuning**: QLoRA via `peft` + `trl` + `bitsandbytes` 4-bit NF4
- **Dataset**: [MathNet](https://huggingface.co/datasets) — 30K+ Olympiad problems, multilingual, multimodal
- **Frontier baselines**: Claude Opus 4.7, Claude Sonnet 4.6, GPT-5, Gemini 3 Pro (via official SDKs)
- **Compute**: UW Hyak Klone cluster, `ckpt-all` partition, 2080Ti GPUs

## Blog

_TODO: link to the write-up once published._

---

_Portfolio project. Week 1 of 4 — 2026-04-23._
