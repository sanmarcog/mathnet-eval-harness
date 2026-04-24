# mathnet-eval-harness

> _TODO elevator pitch: one-sentence framing of what this repo shows and why it matters._

## Headline results

_Frontier comparison from Day-2 full run (2026-04-23). QLoRA row lands in Week 4._

### Frontier model comparison

| Model | N scored | **MathNet accuracy** | Eval cost | Notes |
|---|---|---|---|---|
| **Claude Opus 4.7** | 100 | **84.0%** | $6.14 | Spot-check sample size |
| **Gemini 3 Pro** | 240 / 300 | **73.3%** | $13.55 | `thinking_budget=4096`, 60 problems deferred (daily quota cap) |
| **Claude Sonnet 4.6** | 500 | **65.0%** | $10.35 | |
| **GPT-5.4** | 495 / 500 | **57.8%** | $9.52 | 5 OpenAI safety-filter rejections |
| **GPT-5.4 Mini** | 498 / 500 | **36.7%** | $1.51 | 2 OpenAI safety-filter rejections |
| Qwen-2.5-1.5B-Instruct *(base)* | — | _TBD_ | — | Week 3 |
| Qwen-2.5-1.5B-Instruct + QLoRA *(ours)* | — | _TBD_ | — | Week 4 |

Full methodology + caveats: [results/full/NOTES.md](results/full/NOTES.md). Day-2 report: [docs/day2_report.md](docs/day2_report.md).

### Grader path breakdown (Sonnet 4.6 full run, n=500)

| Path | Count | Share |
|---|---|---|
| `exact` | 87 | 17.4% |
| `normalized` | 30 | 6.0% |
| `symbolic` (sympy) | 16 | 3.2% |
| `judge` (LLM) | 192 | 38.4% |
| `miss` | 175 | 35.0% |

133 / 325 = 40.9% of correct grades resolve on the objective (non-judge) layers. The LLM judge catches the other 59% — set-valued answers, notation synonyms, prose-wrapped solutions. Day-1 judge calibration on 9 accepted answers found 0 false positives.

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
