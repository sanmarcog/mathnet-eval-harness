# Day 1 report — 2026-04-23

## Goal

Stand up the eval harness and run a small smoke test to validate the
problem → API → response → grader pipeline end-to-end. This is the
foundation for the Week-1 through Week-4 comparison of a fine-tuned
Qwen-2.5-1.5B-Instruct against four frontier models on MathNet.

## What shipped

- **Repo scaffolding**: package [src/mathnet_eval/](../src/mathnet_eval/) with `data`, `inference`, `grading`, `training`, `analysis` stubs; CLIs in [scripts/](../scripts/); Hyak clone wired up; `qlora` conda env augmented with `anthropic`, `python-dotenv`, `sympy`, `pytest`.
- **MathNet loader** that bypasses the broken `datasets.load_dataset` path by reading the `data/all/` parquet files directly via pyarrow. Deduplicates against the per-country partitions (they're slices of the same data).
- **Stratified split builder** at [scripts/build_splits.py](../scripts/build_splits.py): English, text-only, has-final-answer. Funnel 27,817 → 4,096 → 500 eval / 3,596 train across 282+ competitions.
- **Inference harness** at [src/mathnet_eval/inference.py](../src/mathnet_eval/inference.py): Anthropic backend (Sonnet + Opus aliases), SHA-256 disk cache, dataclass-normalized `Response`, OpenAI / Gemini / local-HF stubbed for Day 2+.
- **Layered grader** at [src/mathnet_eval/grading.py](../src/mathnet_eval/grading.py): extract → exact → aggressive-normalize → sympy-symbolic → (opt-in) LLM-judge. 24 unit tests passing.
- **20-problem Sonnet 4.6 smoke test** under [results/smoke/sonnet-4-6/](../results/smoke/sonnet-4-6/): raw responses + grades + consolidated [summary.json](../results/smoke/sonnet-4-6/summary.json) + [NOTES.md](../results/smoke/sonnet-4-6/NOTES.md) + side-by-side [judge_review.md](../results/smoke/sonnet-4-6/judge_review.md).

## Headline numbers

| Metric | Value |
|---|---|
| Sonnet 4.6 accuracy (n=20) | **14/20 = 70.0%** |
| Errors | 0 / 20 |
| Grader paths (exact / normalized / symbolic / judge / miss) | 4 / 2 / 1 / 7 / 6 |
| Eval cost | $0.43 |
| Judge cost (est.) | ~$0.07 |
| Wall time | 349 s (~17 s/problem) |

## Judge-calibration finding

Manually eyeballed all 9 judge-accepted answers (before normalizer fixes) against gold. **9/9 verified correct, 0 judge errors.** Of those 9, 2 were cases the cheap string layer *should* have caught (LaTeX `\geq` vs `≥`; varname prefix `$A = -1$` vs `-1`). Post-fix these two now resolve at `normalized`, moving them off the judge path without changing accuracy.

**Takeaway:** the 70% Sonnet baseline is trustworthy. The judge is doing real semantic work, not rubberstamping.

## What surprised us

1. **Data/all is a duplicate view.** First inspection pass double-counted everything because I scanned both `data/all/*.parquet` and the per-country `data/<country>/*.parquet` files. Caught by noticing the `all` partition row count exactly matched the sum of the rest. Fix: load only `data/all/`.
2. **`HF_HOME` is set at Python import time for `huggingface_hub`.** Setting it via `os.environ.setdefault(...)` inside the script was too late, and the xet cache quietly filled the 3 GB home-dir quota on Hyak before failing. Fix: `export HF_HOME=...` before `python` runs, and I moved the existing `~/.cache/huggingface` to scratch with a symlink back to `~/.cache/huggingface` for any lingering defaults.
3. **`datasets 3.0.0` cannot parse the MathNet parquet metadata** — it carries a feature type the library doesn't recognize. Working around with pyarrow directly is fine; longer-term an `upgrade` of `datasets` in the `qlora` env would unlock HuggingFace tooling we'll want for training.
4. **Hyak cannot push to GitHub over HTTPS** (no stored credentials). Fix in place: `rsync` results to laptop, commit + push from there.

## Follow-ups before Day 2 full run

1. **OpenAI (GPT-5) and Google (Gemini 3 Pro) backends** in [inference.py](../src/mathnet_eval/inference.py) — stubs are already wired into the `MODELS` registry.
2. **Budget estimate for 500 × 4 models** — will present to user before the green-light (well above the $5 auto-confirm threshold).
3. **Track judge API cost** through the grade pipeline so [summary.json](../results/smoke/sonnet-4-6/summary.json) reflects true total cost.
4. **Parser-level grader improvements** are noted in [grader-todos.md](./grader-todos.md) but parked until we have 500-problem data to see how often each pattern shows up.

## Links

- Smoke-test results: [results/smoke/sonnet-4-6/](../results/smoke/sonnet-4-6/)
- Consolidated summary: [summary.json](../results/smoke/sonnet-4-6/summary.json)
- Manual judge review: [judge_review.md](../results/smoke/sonnet-4-6/judge_review.md)
- Day-1 NOTES: [NOTES.md](../results/smoke/sonnet-4-6/NOTES.md)
- Grader TODOs: [grader-todos.md](./grader-todos.md)
- Day-0 QLoRA spike preserved at [docs/spikes/day0_toy.py](./spikes/day0_toy.py)
