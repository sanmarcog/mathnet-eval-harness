# mathnet-eval-harness

How well do five frontier LLMs — and a QLoRA-fine-tuned 1.5B open model — solve 500 Olympiad-level math problems from the [MathNet](https://huggingface.co/datasets) benchmark?

## Scoreboard

| Model | N scored | **MathNet accuracy** | Eval cost |
|---|---|---|---|
| Claude Opus 4.7 | 100 | **84.0%** | $6.14 |
| Gemini 3 Pro | 240 / 300 | **73.3%** | $13.55 |
| Claude Sonnet 4.6 | 500 | **65.0%** | $10.35 |
| GPT-5.4 | 495 / 500 | **57.8%** | $9.52 |
| GPT-5.4 Mini | 498 / 500 | **36.7%** | $1.51 |
| Qwen-2.5-1.5B-Instruct + QLoRA *(ours)* | — | *(in progress)* | — |

![Scoreboard](results/figures/scoreboard.png)

Full methodology, caveats, and secondary findings: [docs/findings.md](docs/findings.md).

## Key findings

- **Sonnet 4.6 beats GPT-5.4 by 7pp** at comparable cost (65.0% vs 57.8%, $10.35 vs $9.52). The Anthropic lineage outperforms the OpenAI lineage on MathNet-style olympiad problems in our setup.
- **GPT-5.4 Mini at 36.7% is the realistic peer for fine-tuned small open models.** A 1.5B QLoRA isn't going to beat Opus; the meaningful comparison is against the commodity API tier.
- **The 4-layer grading pipeline (exact → normalized → sympy → LLM judge) reduced LLM-judge spend by ~40%** vs judge-everything. 41% of correct grades resolve on the objective non-judge layers; the judge is reserved for set-valued answers, notation synonyms, and prose-wrapped solutions.
- **GPT-5 family has elevated `miss` rates even after the judge runs.** Investigated on a pre-registered 40-sample manual audit: 85% are genuine model errors, only 10% are grader artifacts — below the 15% pre-registered fix threshold. Numbers stand. See [docs/gpt-missrate-analysis.md](docs/gpt-missrate-analysis.md).
- **Methodology caveats are load-bearing.** Opus is a 100-problem spot-check (95% CI ≈ ±8pp). Gemini ran with `thinking_budget=4096` (capped) and N=239 after hitting a preview-model daily quota cap. OpenAI filtered 7 prompts (`invalid_prompt`) across GPT-5.4 and GPT-5.4 Mini. Denominators are `n_scored`, not 500.

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

# 5. QLoRA training on the UW Hyak cluster (A40 GPU)
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
