# mathnet-eval-harness

Five frontier LLMs and an open-weights 1.7B base — with a QLoRA fine-tune on top — measured against 500 Olympiad-level problems from the [MathNet](https://huggingface.co/datasets) benchmark. Where does fine-tuning still add value when the open base already matches the cheap commercial tier?

## Scoreboard

| Model | N scored | **MathNet accuracy** | Eval cost |
|---|---|---|---|
| Claude Opus 4.7 | 100 | **84.0%** | $6.14 |
| Gemini 3 Pro | 240 / 300 | **73.3%** | $13.55 |
| Claude Sonnet 4.6 | 500 | **65.0%** | $10.35 |
| GPT-5.4 | 495 / 500 | **57.8%** | $9.52 |
| **Qwen3-1.7B base**  *(open, thinking-on, vLLM, 16K)* | 500 | **36.8%** | — |
| GPT-5.4 Mini | 498 / 500 | **36.7%** | $1.51 |
| Qwen3-1.7B + Run 4 self-distill *(ours)* | — | *(pending Run 4 eval)* | — |

![Scoreboard](results/figures/scoreboard.png)

Full methodology, caveats, and secondary findings: [docs/findings.md](docs/findings.md).

## Key findings

- **A current-gen 1.7B open-weights base already matches GPT-5.4 Mini.** Qwen3-1.7B with thinking-on, served via vLLM at 16K-token budget, scores **36.8%** vs Mini's **36.7%** on the same 500-problem split. No fine-tuning. This re-anchors the project: we initially targeted Mini at 36.7% as the bar a 1.5B QLoRA needed to clear, but the open base already clears it. The new question is **where fine-tuning still adds value on top of an already-competitive open base.**
- **The 36.8% / 36.7% parity is not apples-to-apples.** Both models run "in their preferred inference mode" (Qwen3 thinking-on at 16K via vLLM; Mini with OpenAI's default reasoning settings). Identical-constraint comparisons would land somewhere different.
- **Sonnet 4.6 beats GPT-5.4 by 7pp** at comparable cost (65.0% vs 57.8%, $10.35 vs $9.52). The Anthropic lineage outperforms the OpenAI lineage on MathNet-style olympiad problems in our setup.
- **The 4-layer grading pipeline (exact → normalized → sympy → LLM judge) reduces LLM-judge spend by ~40%** vs judge-everything. 41% of correct grades resolve on the objective non-judge layers.
- **GPT-5 family has elevated `miss` rates even after the judge runs.** Investigated on a pre-registered 40-sample manual audit: 85% are genuine model errors, only 10% are grader artifacts — below the 15% pre-registered fix threshold. Numbers stand. See [docs/gpt-missrate-analysis.md](docs/gpt-missrate-analysis.md).
- **Same 63% miss rate on Qwen3 base and GPT-5.4 Mini, different causes.** Mini misses are mostly genuine wrong answers (Day-3 categorization). Qwen3 misses are dominated by **convergence failure** — 35% of Qwen3 outputs hit the 16K token ceiling without emitting a final answer. Fine-tuning on solution+answer training data should target this specifically.

![Grader paths](results/figures/grader_paths.png)

## Architecture

![Pipeline](docs/architecture.png)

MathNet (27,817 problems) → English/text/has-answer filters → stratified splits (500 eval / 3,596 train / 14,585 multilingual train) → unified inference harness (5 API backends + local HF / vLLM with disk cache) → 4-layer grading pipeline → committed JSON results per problem.

## Reproducing

```bash
# 1. Clone + install
git clone https://github.com/sanmarcog/mathnet-eval-harness.git
cd mathnet-eval-harness
pip install -e .

# 2. Configure API keys (Anthropic / OpenAI / Google) for frontier eval
cp .env.example .env
# edit .env

# 3. Build the eval / train splits from MathNet
python scripts/build_splits.py --out data/splits

# 4. Frontier eval (example: Claude Sonnet 4.6 on 20 problems)
python scripts/run_eval.py --model sonnet-4-6 --split eval --n 20

# 5. Open-base eval (Qwen3-1.7B via vLLM, thinking-on, 16K)
sbatch slurm/eval_qwen3_base.sbatch

# 6. QLoRA training on the UW Hyak cluster (A40 GPU)
sbatch slurm/train_qlora_run2.sbatch
```

Full 500-problem frontier eval costs **~$41 end-to-end**. Local training requires a GPU with ≥24 GB VRAM; an interactive slot on Hyak is:

```bash
salloc --account=demo --partition=ckpt-all --gpus-per-node=a40:1 \
       --mem=32G --cpus-per-task=4 --time=4:00:00
```

## Repo structure

```
src/mathnet_eval/     # core library (importable)
  data.py             # MathNet loading, stratified splits, prompt formatting
  inference.py        # unified client for Claude / OpenAI / Gemini / local HF + vLLM
  grading.py          # 4-layer grader: exact → normalized → sympy → judge
  training.py         # QLoRA training loop (TRL SFTTrainer)
scripts/              # CLI entrypoints (argparse → library calls)
  merge_adapter.py    # post-training PEFT merge into bf16 weights for vLLM serving
  make_figures.py     # headline figure generation
slurm/                # sbatch scripts for Hyak (ckpt-all partition)
results/              # committed JSON outputs and figures
  full/               # 500-problem runs, per-model subdirs
  figures/            # headline plots
docs/                 # findings report, methodology notes, investigations
tests/                # pytest unit tests
```

## Tech stack

Python 3.11 · HuggingFace transformers / peft / trl / bitsandbytes · vLLM · Anthropic + OpenAI + Google GenAI SDKs · PyTorch · UW Hyak Klone (Slurm, A40 GPU).

## Blog

Write-up (pending Run 2 training completion): [docs/blog_post.md](docs/blog_post.md)

---

*Portfolio project. Week 1 of 4 — first commit 2026-04-22.*
