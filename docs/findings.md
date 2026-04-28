# Findings — frontier APIs, an open-weights base, and four QLoRA attempts

This document is the public methodology + results writeup. It complements the [scoreboard in the README](../README.md#scoreboard) with the full caveats, the headline parity finding, the negative result on QLoRA fine-tuning at 1.7B, and the operational lessons from running the evaluation.

The lead is the parity finding (Qwen3-1.7B ≈ GPT-5.4 Mini at 36.8% / 36.7%). The intellectual contribution is the negative-result diagnosis: across four QLoRA configurations spanning every sensible knob, none surpassed the Qwen3-1.7B base — which is itself the product of the Qwen team's full SFT + GRPO pipeline (per the [Qwen3 technical report 2505.09388](https://arxiv.org/abs/2505.09388), 3,995 query-verifier pairs were used in their Reasoning RL stage before release). So "the base" here is post-RL, not just post-SFT, which both raises the bar and reframes what "fine-tune past it" would actually require.

## Scoreboard

| Model | N scored | **Accuracy** | Eval cost |
|---|---|---|---|
| Claude Opus 4.7 *(spot-check)* | 100 | **84.0%** | $6.14 |
| Gemini 3 Pro *(partial)* | 240 / 300 | **73.3%** | $13.55 |
| Claude Sonnet 4.6 | 500 | **65.0%** | $10.35 |
| GPT-5.4 | 495 / 500 | **57.8%** | $9.52 |
| **Qwen3-1.7B base** *(open, thinking-on, vLLM, 16K)* | 500 | **36.8%** | — |
| GPT-5.4 Mini | 498 / 500 | **36.7%** | $1.51 |
| Qwen3-1.7B + Run 4 self-distill *(ours)* | 500 | **28.8%** | — |
| **API total spend** |  |  | **$41.06** |

Denominator is `n_scored`; missing problems on GPT are OpenAI safety-filter rejections (documented below). Gemini finished at N=240 due to a preview-model daily quota cap. Qwen3 cost is the academic-cluster compute on a single A40 GPU (no API billing).

**Eval cost is generation-side only.** The "$41.06 API total" figure does *not* include the Sonnet 4.6 LLM-judge API spend, which runs on every problem the cheap grader layers don't resolve (typically 30-40% of problems per model). Per-call judge cost is small (~$0.01-0.03), but across all five frontier models on n=500 it is real spend not currently tracked. The grade pipeline writes `"cost_notes": "excludes LLM-as-judge API spend (not yet tracked)"` to every `summary.json` to keep the omission visible. True total eval spend is approximately 10-20% higher than the figures shown.

## The headline: open-weights parity at the cheap-commercial waterline

The project was originally framed as: *can a fine-tuned 1.5B open model push past GPT-5.4 Mini at 36.7%?* That bar was set by the API scoreboard.

We then added Qwen3-1.7B as the open-weights baseline. **It scored 36.8% out of the box, with no fine-tuning** — at parity with the same Mini target.

We've kept that fact in the headline rather than rewriting history. The re-framed question is more current and more interesting: with the open ecosystem already at the cheap-commercial-tier waterline, **where does fine-tuning still add value?**

### Paired McNemar test on the n=498 intersection

The aggregate accuracy gap (36.8% vs 36.7%) is one problem in absolute terms; the headline framing should be defended with a proper paired test. On the 498-problem intersection:

| | Correct | Accuracy on n=498 |
|---|---|---|
| Qwen3-1.7B base | 184 | 36.9% |
| GPT-5.4 Mini    | 183 | 36.7% |

Transition matrix: 111 both correct, 73 Qwen3-only, 72 Mini-only, 242 neither. **McNemar exact two-sided p = 1.0000** (145 discordant pairs, smaller side 72). The discordant pairs are near-perfectly balanced 73 / 72 — the data is fully consistent with the two models being at the same expected accuracy.

But: the **parity is in aggregate, not per-problem.** They disagree on 145 / 498 problems, which is 29% of the eval — they reach the same headline rate by solving substantially different problem subsets. Reproducible via [`scripts/compute_parity_mcnemar.py`](../scripts/compute_parity_mcnemar.py).

### Caveat: not a constraint-matched comparison

The near-tie is "both models in their preferred inference mode": Qwen3 thinking-on at 16K tokens via vLLM, Mini with OpenAI's default reasoning configuration. Identical-constraint comparisons (e.g. capping Qwen3 to a comparable reasoning budget) would land somewhere different.

The headline finding isn't *"these are identical models"* (the per-problem disagreement rate of 29% rules that out) — it's *"the open ecosystem has caught up enough that the useful inference mode of a 1.7B base ties the useful inference mode of the cheap commercial tier in mean accuracy."*

### Same 63% miss rate, different causes

GPT-5.4 Mini and Qwen3-1.7B base both `miss` on 63% of problems. Looks similar; isn't.

**Mini misses are mostly genuine wrong answers.** From the [40-sample manual audit](./gpt-missrate-analysis.md): 85% of sampled GPT-5 misses were genuine model errors, only 10% grader artifacts. The model commits to an answer; the answer is just wrong.

**Qwen3-1.7B base misses are dominated by convergence failure.** From the 50-problem 16K-pilot:

- 35% of outputs hit the 16K-token ceiling
- 65% emit `\boxed{...}` somewhere in the output
- 33% saturate **and** never emit `\boxed{...}` — the model thinks itself out without ever committing

So Qwen3's 63% miss rate decomposes (roughly) into ~half "thought-loop failed to converge" and ~half "wrong but committed answer." This is the mechanistic setup for the fine-tuning question: *failure to converge* is exactly what fine-tuning on solution+answer pairs (with clean `\boxed{...}` conclusions) was supposed to fix.

It didn't. The fine-tuning attempts amplified the convergence-failure mode rather than reducing it. Why is the rest of this document.

## The negative finding: across four QLoRA configurations, none surpassed the SFT+GRPO-trained Qwen3-1.7B base

We're documenting the journey explicitly because it surfaces a useful research finding for anyone fine-tuning small instruction-tuned models on math.

### Run 1 — Qwen 2.5-1.5B-Instruct, default hyperparams

First attempt was a sensible starting point from common QLoRA tutorials:

- Base: `Qwen/Qwen2.5-1.5B-Instruct`
- LoRA r=16, alpha=32, all 7 target modules (q/k/v/o + gate/up/down)
- Effective batch 16, learning rate **2e-4**, 2 epochs
- Loss: standard SFT loss over the full sequence (no `completion_only_loss`)
- Training data: 3,596 English MathNet rows

**Result on n=150 paired vs base on the same IDs:**

| | Correct | Accuracy |
|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | 24/150 | 16.0% |
| + Run 1 LoRA adapter | 15/150 | 10.0% |
| Delta | | **-6.0 pp** |

13 problems regressed (base ✓ → adapter ✗), 4 improved (base ✗ → adapter ✓). McNemar exact two-sided **p = 0.049**. The fine-tune was meaningfully *worse* than the base.

### Run B — single-variable: enable `completion_only_loss`

Hypothesis: Run 1's loss was being computed on the full sequence including the system prompt and user problem, so ~20% of gradient was wasted reproducing the prompt instead of the answer. We enabled `completion_only_loss=True` (mask system + user tokens) and held everything else identical.

Pre-flight, we also wrote [scripts/verify_response_template.py](../scripts/verify_response_template.py) to ensure the response template `<|im_start|>assistant\n` actually appears as a contiguous subsequence in the tokenized data — TRL's collator silently fails to mask if it doesn't, and we wanted to rule that failure mode out before launching a 5-hour training run. Verification passed.

**Result on n=150 paired vs base:**

| | Correct | Accuracy |
|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | 24/150 | 16.0% |
| + Run B (= Run 1 + completion_only_loss) | 12/150 | 8.0% |
| Delta vs base | | **-8.0 pp** (p = 0.008) |
| Delta vs Run 1 | | -2 pp (p = 0.58, not significant) |

Run B was *not* better than Run 1. Loss-masking was not the load-bearing fix. The regression came from somewhere else.

### Pivot: deep-dive into the literature

After Run B failed to recover and a learning-rate ablation (D-LR, LR=5e-5) was cancelled mid-flight when mid-eval pause cost was projected to exceed walltime, we paused to read what works empirically for math fine-tuning at this scale. The most informative source was Alibaba's own [Qwen2.5-Math Technical Report](https://arxiv.org/html/2409.12122v1) — the only known successful math fine-tune of this model family at this size. Their published recipe for the 1.5B model:

| | Alibaba's Qwen2.5-Math 1.5B | Our Run 1 / B |
|---|---|---|
| **Learning rate** | **2 × 10⁻⁵** (decays to 7×10⁻⁷) | 2 × 10⁻⁴ |
| **Effective batch** | **128** | 16 |
| Epochs | 3 | 2 |
| Seq length | 4,096 | 2,048 |
| Data scale | 2.5M CoT problems, RM-curated | ~3.6K problems |
| Base | Qwen2.5-1.5B (the *base*, not Instruct) | Qwen2.5-1.5B-**Instruct** |

We were **10× too high on LR**, **8× too small on batch**, **700× too small on data**, and on the wrong base variant. Each contributes; together they explain the regression.

The [catastrophic forgetting literature](https://arxiv.org/abs/2512.13706) (Reynolds 2025) documents the failure mode directly: Flan-T5-Base loses 64.5 pp on NLI within 1,000 steps of math-only fine-tuning. The paper does not test LoRA specifically, but the mechanism it documents — aggressive narrow-domain training degrading general abilities the model needs — is what we observed in Runs 2 and 3.

We did *not* run an LR sweep to triangulate the right value. The literature converged on 2e-5 for this exact model family at this size. Adopting that directly was a faster path to a working result than re-deriving it experimentally.

### Run 2 — recipe-matched

Run 2's training sbatch ([slurm/train_qlora_run2.sbatch](../slurm/train_qlora_run2.sbatch)) matches the published recipe on every numerical hyperparameter:

- Learning rate **2e-5**
- Effective batch **128** (per-device 4 × grad-accum 32)
- **3 epochs**
- Sequence length **4,096**

Plus our additions: QLoRA r=64 / alpha=128 (we have ~10K rows vs Alibaba's 2.5M, and rank gives the adapter capacity to compensate); `completion_only_loss=True`; `enable_thinking=False` in the chat template (training data has no `<think>` traces); base `Qwen/Qwen3-1.7B` (paired with the 36.8% baseline that anchors the headline).

We also filtered the multilingual training data ([scripts/filter_train_by_solution_length.py](../scripts/filter_train_by_solution_length.py)) to drop the 13.7% of rows with empty `solutions_markdown` and the 6.5% with sub-100-token completions — those rows train the model to "after this prompt, emit boxed answer" with zero reasoning content. Filtered set is 11,648 rows.

**Result: 15/500 = 3.0%, paired delta -33.8 pp vs base, p < 10⁻⁴.**

Worse than the unfine-tuned base by 33 percentage points. Recipe-matching was not enough.

### Run 3 — adding boxed-answer augmentation

Hypothesis: Run 2's regression was due to MathNet's `solutions_markdown` field containing `\boxed{}` in only ~1.5% of rows. The QLoRA fine-tune unlearned the boxed-answer convention from the base because it was almost absent from training data.

Single change vs Run 2: each training row's solution gets an explicit `\n\nTherefore, the final answer is $\\boxed{<final_answer>}$` appended. A pre-launch safeguard ([scripts/verify_boxed_augmentation.py](../scripts/verify_boxed_augmentation.py)) verified the augmented text round-trips through `apply_chat_template` and `extract_answer` on 8 sampled rows.

**Result: 19/500 = 3.8%, paired delta -33.0 pp vs base, p < 10⁻⁴.**

Diagnosis post-mortem on Run 3's outputs: the augmentation *did* teach the boxing convention back — 41.8% of Run 3 responses emit `\boxed{}` (close to base's 65%). But the *content* inside the boxes is wrong nearly every time (~9% correct-among-boxed vs base's ~57%). So the fine-tune restored format but damaged math. Naive SFT on raw MathNet solutions degrades reasoning faster than format augmentation can rescue.

### Run 4 — self-distillation from base's own correct answers

Hypothesis (per [STaR 2203.14465](https://arxiv.org/abs/2203.14465) / [RFT 2308.01825](https://arxiv.org/abs/2308.01825), with loose support from [LIMO 2502.03387](https://arxiv.org/abs/2502.03387)): training on the *model's own* correct reasoning may elicit latent capability without the noise of MathNet's heterogeneous gold answers. The Run 2/3 training data had MathNet's prose-form gold (e.g. *"All n that are multiples of 4"*) inside a `\boxed{}` — confusing supervision. Run 4 replaces that with traces the base actually generated and got right. **LIMO caveat:** LIMO operates at 32B; the follow-up [LIMR (arxiv 2502.11886)](https://arxiv.org/abs/2502.11886) explicitly notes LIMO "significantly underperforms at 7B-scale through SFT." We expected our 1.7B-scale use to inherit this limitation, so the LIMO citation motivates attempting the recipe but does not predict success.

Critical caveat from [Why Does Self-Distillation (Sometimes) Degrade Reasoning](https://arxiv.org/abs/2603.24472) (Kim et al., 2603.24472). Appendix F.2 reports **-45.9% on Qwen3-1.7B with thinking-on** under their setup — our exact model and our exact inference mode. Their setup studies (a) off-policy SFT where the teacher conditions on the gold solution, and (b) on-policy SDPO; the identified mechanism is "conditioning the teacher on rich information suppresses uncertainty expression, hurting OOD." Run 4 is **rejection-sampled SFT**: the base generates traces without seeing gold, and we keep only the ones that happen to be correct. Solution-conditioning is structurally absent, so we expected a milder regression than -45.9% — and got it (our -8 pp). To minimize residual risk we preserve full `<think>...</think>` traces in the training targets, train with `enable_thinking=True` in the chat template (matches inference), `max_seq_length=8192` to fit the long traces, and use conservative LR `1e-5` with 1-2 epochs.

Pipeline:

1. Run base Qwen3-1.7B on ~430 train problems via vLLM, thinking-on, 16K
2. Filter for rows where `extract_answer` matches gold (cheap grader)
3. Save those as a self-distilled training set (146 rows kept)
4. SFT same base on the distilled set (LR 1e-5, eff batch 4, 2 epochs, ~75 opt steps)
5. Merge → eval n=500 thinking-on 16K, paired vs base 36.8%

**Pre-registered interpretations** (locked before launch):

| Run 4 result | Interpretation |
|---|---|
| ≥ 36.8% (≥ base) | Self-distillation works at our scale |
| 30-36% | Capability preserved; no improvement |
| 10-30% | Partial collapse despite trace preservation |
| ≤ 10% | Full collapse like Run 2/3 |

**Result: 144/500 = 28.8% (paired delta -8.0 pp vs base, McNemar exact two-sided p = 0.0001).** Lands in the "partial collapse despite trace preservation" bucket.

Self-distillation produced a *much milder* regression than Runs 2 and 3 (which sat at -33 to -34 pp), but the fine-tune still ended up below the Qwen3-1.7B base.

## Figure A: the diagnostic

![Outcome decomposition across base + Run 2/3/4](../results/figures/miss_mode_decomposition.png)

This is the central diagnostic of the writeup. Each bar is one of the four Qwen3-1.7B configurations on the same 500-problem eval, with identical inference settings (vLLM, thinking-on, `max_new_tokens=16384`, `temperature=0`). The bars decompose into four buckets:

- **Correct** — graded by any of the 4 grader layers
- **Wrong, committed** — `miss`, output was under 16K tokens (the model finished and was wrong)
- **Saturated, boxed but wrong** — `miss`, output hit the 16K cap, model emitted a `\boxed{}` somewhere
- **Saturated, never boxed** — `miss`, output hit the 16K cap, model never emitted a final answer (convergence failure)

The deep-red segment grows from base to Run 4. The fine-tune trained the model to *think longer*, not to *think better*. Specific numbers: base saturated 157 outputs (31% of all problems); Run 4 saturated **198** (40%). Of Run 4's 356 misses, **53% are saturated AND never boxed** — versus a smaller share for the base.

### The mechanism: trained on long traces, the model thinks longer

The mechanism is mechanical. Base produces long reasoning traces (median ~14K tokens with `<think>` blocks). Run 4 trained on those long traces — so the resulting model thinks *longer*. On problems Run 4 can solve, this is fine. On problems it can't, the model spirals into recomputation loops past the 16K ceiling without converging.

The mechanism we observed is not the one [Kim et al. (2603.24472)](https://arxiv.org/abs/2603.24472) identify (uncertainty-expression suppression from solution-conditioned teacher distillation). That mechanism is structurally absent from rejection-sampled SFT — we don't condition the teacher on the gold solution. The much gentler -8 pp we measured (vs Kim et al.'s -45.9% on the same Qwen3-1.7B + thinking-on setup with off-policy SFT or SDPO) is consistent with avoiding their failure mode. But a different failure mode bit us, which is our own diagnosis from the eval data: training on base's long reasoning traces taught the model to *think longer*, amplifying the convergence-failure mode the base was already prone to. The base sat at a precarious equilibrium between useful long thinking and runaway thinking; even a soft, faithful retraining tipped it past the cliff.

### A concrete illustration: problem `0ai2`

> *2014 lines are given in a plane, arranged in three groups of pairwise parallel lines. What is the greatest possible number of triangles formed by the lines?* (gold answer: **302561952**)

**Qwen3-1.7B base solved this in 5,271 tokens.** Clean reasoning: maximize $a \cdot b \cdot c$ subject to $a + b + c = 2014$ → pick $(671, 671, 672)$ → compute $671 \cdot 671 \cdot 672 = 302{,}561{,}952$ → `\boxed{302561952}`. Done.

**Run 4 saturated at 16,384 tokens with no answer.** It picked the wrong distribution (672, 672, 670 — slightly off) and got 302,561,280 (wrong by 672). Then second-guessed itself — *"Wait, earlier I had 755,728,512..."* — and spent the remaining ~10,000 tokens re-factoring the expression in different ways:

> *"Total = 672 × [672 × 670 + (672 × 671 + 671 × 670 + 670 × 669)/2]<br>
> Which is what I had before, leading to 672 × 1,124,596 ="*

— and runs out of tokens mid-arithmetic, never committing to a final answer.

This is the failure mode generalized: the fine-tune produced a model that *can* arrive at correct intermediate values, but lost the base's discipline of **picking one approach and finishing**. Trained on long-trace data, it learned to *keep thinking* past the point where the base would have boxed an answer and stopped.

## Figure B: the paired-comparison view

![Run 4 vs base transition matrix on 500 paired problems](../results/figures/transition_matrix_run4_vs_base.png)

The 2×2 transition matrix on n=500 paired problems makes the asymmetry concrete. Both ✓ and neither cells are large; the off-diagonal cells show that **regressions outnumber improvements roughly 2:1**. The net delta of -40 problems = -8 pp is what shows up on the scoreboard, but the underlying picture is "30 things became right, 70 things became wrong" — not "everything got slightly worse."

This matters for interpretation: Run 4 isn't "uniformly weaker base." It's "shifted distribution" — the fine-tune helped some problems, broke others, and the broken outnumbered the helped.

McNemar exact two-sided p ≈ 0.0001 — the discordant-pair imbalance is well past statistical significance.

## Summary of fine-tune attempts

The runs split into two distinct experiment families with **different base models**, so the deltas are NOT directly comparable across the dividing line. Run 1 / Run B used **Qwen2.5-1.5B-Instruct** (the original API-scoreboard anchor), n=150 paired vs Qwen2.5-1.5B base. Runs 2/3/4 use **Qwen3-1.7B** (the open-base anchor after the parity finding pivoted the project), n=500 paired vs Qwen3-1.7B base. The two halves of the table are connected only by methodology lessons (LR ablation, completion-only-loss, recipe-matching), not by a shared baseline.

**Early experiments — Qwen2.5-1.5B-Instruct base, n=150 paired:**

| Run | Recipe / single change | Train data | Eval acc (paired Δ vs Qwen2.5 base) |
|---|---|---|---|
| Run 1 | default LR 2e-4, no completion-only-loss | English MathNet (3,596) | 10.0% (-6 pp paired vs Qwen2.5 base 16%, p=0.049) |
| Run B | + `completion_only_loss=True` | (same as Run 1) | 8.0% (-8 pp paired, p=0.008) |

**Project anchor runs — Qwen3-1.7B base, n=500 paired:**

| Run | Base | Recipe / single change | Train data | Eval acc (paired Δ vs Qwen3 base 36.8%) |
|---|---|---|---|---|
| **Base** | Qwen3-1.7B | (no fine-tune) | — | **36.8% (anchor)** |
| Run 2 | Qwen3-1.7B | recipe-match Alibaba's Qwen2.5-Math 1.5B (LR 2e-5, 3 epochs, eff batch 128, multilingual filtered) | 11,648 rows | **3.0% (-33.8 pp)** |
| Run 3 | Qwen3-1.7B | + boxed-answer augmentation (every row gets `\\boxed{X}` appended) | 11,648 rows | **3.8% (-33.0 pp)** |
| Run 4 | Qwen3-1.7B | + self-distilled training data (base's own correct outputs, traces preserved) | 146 rows | **28.8% (-8.0 pp paired, p < 10⁻³)** |

D-LR (LR=5e-5 midpoint ablation under the early-experiment family) was started and cancelled mid-flight when mid-eval pause cost was projected to exceed walltime; no held-out accuracy. Loss curve was numerically stable.

## What it would take to actually beat the open base

At 1.7B, the Qwen3 base appears to sit at a local optimum hard to disturb without breaking — and that optimum was reached via the Qwen team's own SFT + GRPO pipeline ([2505.09388](https://arxiv.org/abs/2505.09388)). Self-distillation reduced the *damage* of fine-tuning (Runs 2/3 = -34 pp; Run 4 = -8 pp), but didn't push above. **Surpassing the open base at this size likely requires methods structurally different from any we tested.** Four candidates:

1. **Bias-corrected RL — Dr. GRPO is the load-bearing change, not the base.** Vanilla GRPO has its own length-amplification pathology — exactly the failure mode that bit Run 4 — documented in [Dr. GRPO (Liu et al., 2503.20783)](https://arxiv.org/abs/2503.20783): *"optimization bias in GRPO ... artificially increases response length, especially for incorrect outputs."* The bias-corrected variant addresses this. The base choice is a real trade-off:
    - **Qwen3-1.7B** — already GRPO-trained per the [Qwen3 tech report (2505.09388)](https://arxiv.org/abs/2505.09388). Applying Dr. GRPO on top tests whether the bias-correction extracts marginal lift from an already-RL-tuned base. Expected magnitude is smaller (iterative RL has documented diminishing returns), but it preserves the paired comparison against our existing n=500 eval — same problem, same harness, McNemar against base + Run 4 directly comparable.
    - **[Qwen2.5-Math-1.5B base](https://arxiv.org/abs/2409.12122)** — math-specialized, hasn't absorbed RL. Dr. GRPO reports at 1.5B scale: AIME24 16.7% → 20.0%, MATH500 61.8% → 74.2%, OlympiadBench 28.4% (post-GRPO not stated), average ~36% → ~42%. Higher expected magnitude with published replication signal, but a different problem (new baseline eval, not paired with the Qwen3 work in this writeup). Ready recipes via [SimpleRL-Zoo (2503.18892)](https://arxiv.org/abs/2503.18892) and HuggingFace [Open-R1](https://github.com/huggingface/open-r1).

    Both are defensible. The trade-off is paired-comparison continuity vs. expected-magnitude.
2. **Test-time MCTS search with self-evolved data — [rStar-Math](https://arxiv.org/abs/2501.04519).** Distinct mechanism from GRPO: Monte Carlo Tree Search at inference plus a process preference model trained on self-evolved traces. Reports Qwen2.5-Math-7B 58.8% → 90.0% on MATH and Phi3-mini-3.8B 41.4% → 86.4%; demonstrated only at 7B and 3.8B scales — not 1.7B — so transfer to our setting is open.
3. **Distillation from a stronger external teacher.** Sonnet 4.6 emits decisive ~5K-7K-token solutions on these same problems; the base emits noisy ~14K-token solutions. Training on Sonnet's trace distribution (or DeepSeek-R1's) would relabel the supervision target with cleaner, more decisive reasoning — addressing exactly the "trained to think longer" mechanism. Cost-prohibitive for the project budget but the most direct fix.
4. **Continued pretraining on a larger math corpus** (e.g. [Llemma](https://arxiv.org/abs/2310.10631) and its Proof-Pile-2 dataset of scientific papers, math web data, and mathematical code). Different scale entirely. Useful as an upstream step *before* any of the above; not a replacement for them.

## Frontier-tier findings

1. **Opus 4.7 is clearly the strongest**, but the spot-check sample size means we treat the 84% as indicative of the ceiling, not as a precise number.
2. **Sonnet 4.6 beats GPT-5.4 by 7 pp.** Aggregate: 65.0% vs 57.8%. **Paired McNemar on the n=495 intersection: 324 vs 286 correct, exact two-sided p = 0.0019** (144 discordant pairs, 91 Sonnet-only vs 53 GPT-only). The Anthropic lineage outperforms the OpenAI lineage on MathNet-style olympiad problems in our setup, by an effect well past noise, at comparable eval cost ($10.35 vs $9.52). Reproducible via [`scripts/compute_parity_mcnemar.py`](../scripts/compute_parity_mcnemar.py).
3. **Gemini 3 Pro at 73.3% (N=240 of 300 target) with capped thinking.** Ran with `thinking_budget=4096` to fit budget; would plausibly score 1-3 pp higher with default (unbounded) thinking.
4. **The GPT-5 family has a large `miss` rate even with the LLM judge enabled** (209 and 315 misses across GPT-5.4 / Mini). Investigated on a pre-registered 40-sample manual audit: [docs/gpt-missrate-analysis.md](./gpt-missrate-analysis.md). 85% of sampled misses are genuine model errors; only 10% are grader artifacts (`extractor_failure` or `judge_false_negative`), below the [pre-registered 15% fix threshold](./gpt-missrate-preregistration.md). **Conclusion: numbers stand; Mini at 37% is the real peer-tier target, not a grader-inflated figure.**

## Where Qwen3-1.7B base is strongest and weakest

Stratifying by `topics_flat` top-level prefix ([full breakdown](./qwen3_base_topic_breakdown.md)):

| Topic | n | accuracy |
|---|---|---|
| Geometry | 56 | **44.6%** |
| Algebra | 275 | 41.1% |
| Number Theory | 177 | 35.0% |
| Discrete Mathematics | 155 | **23.9%** |

A problem can be tagged with multiple top-level prefixes, so the n's sum to more than 500. The ~20 pp gap between Geometry/Algebra and Discrete Mathematics is real signal. Within Discrete Mathematics, **Graph Theory** is the model's hard floor: **1/18 = 5.6%**. Within Algebra, by contrast, prealgebra runs at 46.6% and equations/inequalities at 42.9%. The open-base 1.7B does well on procedural/computational topics and worse on topics requiring combinatorial case analysis.

For per-competition Run 4 vs base deltas (where the fine-tune hurt vs helped on a competition basis), see [results/figures/per_competition_delta_run4.png](../results/figures/per_competition_delta_run4.png) (figure C, supplementary).

## Methodology caveats

These are prominent on purpose — the headline numbers mean very different things with vs. without them.

- **Opus 4.7 is N=100 by design, a spot-check not a full 500.** The $60 budget ceiling drove this choice. 95% CI on 84% at N=100 is roughly ±8 pp — treat the number as indicative of the ceiling, not as a precise accuracy measure.
- **Gemini 3 Pro ran with `thinking_budget=4096`**, other models ran with default reasoning settings. Cost-control decision based on a 15-problem calibration showing median 5,454 thoughts / max 15,730 per problem under default thinking. A default-thinking run would plausibly score 1-3 pp higher.
- **Gemini 3 Pro is N=300 by design** (rescoped from 500 during calibration to fit budget) and currently N=239 in practice due to a preview-model daily quota cap. The remaining 61 are deferred to a future run.
- **OpenAI filtered 5 / 500 GPT-5.4 and 2 / 500 GPT-5.4 Mini prompts** with `invalid_prompt` 400s. Accuracy denominators are `n_scored` (495 and 498) rather than 500 for those two models. See [results/full/openai-flagged-problems.md](../results/full/openai-flagged-problems.md).
- **Judge model = Claude Sonnet 4.6.** The judge's job is pairwise equivalence, not problem-solving; the initial 9-problem calibration found **0 false positives on the 9 accepted answers tested** — false negatives were not assessed in that calibration, and the later 40-sample GPT miss-rate audit did surface 2 judge false negatives ([docs/gpt-missrate-analysis.md](./gpt-missrate-analysis.md)). So the calibration bounds the false-positive risk, not the false-negative risk; a second-judge cross-check would be a reasonable follow-up. Same family as the #3 model on the scoreboard, which is its own caveat.
- **Saturation cutoff is identical across all four Qwen3-1.7B runs.** All four sbatches (`eval_qwen3_base.sbatch`, `eval_qwen3_run2.sbatch`, `eval_qwen3_run3.sbatch`, `eval_qwen3_run4.sbatch`) pass `--max-new-tokens 16384` to the same eval script with `temperature=0` on the same vLLM backend. The single source of truth for the analysis-side definition is `mathnet_eval.SATURATION_CUTOFF` in [src/mathnet_eval/__init__.py](../src/mathnet_eval/__init__.py); both `scripts/make_diagnostic_figures.py` and `scripts/analyze_finetune_vs_base.py` import it. If anyone re-runs at a different cap, update the constant and the sbatches together — figure A's saturated-vs-not labels would silently disagree otherwise.
- **Single-seed runs.** All Qwen3 evals run at `temperature=0` (greedy), and all training runs use `--seed 0`. Greedy generation is deterministic so the eval-side seed is moot for the base run, but the QLoRA-trained adapters depend on training-data shuffle order, and Run 4 specifically picked a *single* sample of 146 base-correct rows (no resampling). The paired McNemar tests we report are valid for the comparison as run; they don't bound the variance over alternative training-data samples.
- **Run 4 self-distill data is N=146.** That is small enough that a different sample of base-correct rows could plausibly produce a meaningfully different result. The McNemar test catches that the difference vs base is real (p ≈ 10⁻⁴), but does not bound the *direction* under resampling. A robust follow-up would resample the distillation set 3-5 times and report distribution of paired deltas.

### Grader path breakdown (Sonnet 4.6, n=500)

Correctness is assigned by a 4-layer grading pipeline that tries cheap, deterministic checks before falling back to an LLM judge. On the Sonnet full run:

| Path | Count | Share |
|---|---|---|
| `exact` | 87 | 17.4% |
| `normalized` | 30 | 6.0% |
| `symbolic` (sympy) | 16 | 3.2% |
| `judge` (LLM) | 192 | 38.4% |
| `miss` | 175 | 35.0% |

133 / 325 = **40.9% of correctly-graded outputs resolve on the objective (non-judge) layers** (the cheap layers run first; if any returns equal, the judge isn't called). The LLM judge catches the other 59% — set-valued answers, notation synonyms, prose-wrapped solutions. The initial judge calibration on 9 accepted answers found 0 false positives; the calibration's scope was false-positive only (false negatives weren't assessed in those 9 — the later 40-sample audit found 2 judge false negatives, see [docs/gpt-missrate-analysis.md](./gpt-missrate-analysis.md)). See [results/figures/grader_paths.png](../results/figures/grader_paths.png) for the per-model distribution.

## Operational findings

Things we did not expect, and which cost us time. Documenting them here so the next person doesn't repeat them.

1. **Gemini 3.x preview models have a 250 RPD cap even on paid billing.** Not in the public rate-limits documentation that we saw. The run hit the cap at exactly 239 successful + 61 failing = 250 daily + some retries.
2. **AI Studio and Google Cloud billing are parallel lanes.** Credits added via AI Studio do not show up in Cloud-console's "$300 free trial credits" view; they appear in a separate "Gemini API Billing" view. Confusing UX; tripped us up for an hour diagnosing.
3. **Our cost estimator was under-reporting Gemini by ~6×** because `thoughts_tokens` weren't summed. Found by cross-checking our `summary.json` against the billing dashboard (thought we had spent $2.05, the dashboard said $13.49). Fix in [scripts/grade_results.py](../scripts/grade_results.py); the $13.55 figure in the scoreboard matches the dashboard.
4. **Opus 4.7 rejects the `temperature` parameter** (reasoning-style model); our preflight script caught this before launch. Worth its weight in gold — a 500-problem failure at cost would have been painful.
5. **GPT-5.4 emitted zero reasoning tokens on our prompts** — every smoke-test problem showed `reasoning_tokens=0`, so its cost matched "visible output" estimates exactly. Gemini, by contrast, used thinking heavily.
6. **Slurm `ckpt-all` partition preempts long jobs aggressively.** A 500-problem vLLM eval batched as one `llm.generate` call is not preemption-survivable. Fixed by chunking the batched generation in groups of 50 and writing per-problem JSONs after each chunk, so a preempt loses at most one chunk's worth of work. Patch in [scripts/eval_qwen_hf.py](../scripts/eval_qwen_hf.py).

## Methodology notes

A few things we got wrong on the first attempt and fixed in flight.

1. **Mid-training eval was too narrow on Run 1.** The QLoRA training loop logs a held-out 50-problem accuracy at 25%/50%/75% of training plus each epoch boundary. Run 1 was capped at 1024 output tokens with greedy decoding and the cheap-grader-only path — no LLM judge. All four checkpoints reported 1/50 = 2%, which read as "training is broken." It wasn't. 1024 tokens is under 10% of the median convergent output length on this benchmark; the model hadn't yet emitted a `\boxed{...}` by the time the cap hit, so `extract_answer` returned `None` on 49/50. **Bumped for Run 2:** mid-eval token cap raised to 4096 in `src/mathnet_eval/training.py`. **But that is still too small to give a meaningful Qwen3 accuracy signal** — Qwen3-1.7B's median final-answer length is ~14K tokens with thinking-on. Treat mid-eval at 4096 as a *training-loop sanity check* (is the model still emitting parseable output at all, has it numerically diverged?), not as an accuracy estimate. The real Run 2/3/4 accuracy signal comes from the post-training n=500 vLLM-served eval at the full 16K cap.
2. **Pre-registered ranges turned out to be too pessimistic on Qwen3 base.** Range was 14-22% (anchored to Qwen 2.5 numbers). Actual landed at 36.8%, well outside. We're treating pre-registration the way it's meant to be treated — **as a flag for when reality differs from your prior, not as a target to defend.** When the prior misses, document the miss and update the model.
3. **vLLM does not load PEFT adapters in our eval-script wiring.** Post Run 2 training, we merge the LoRA adapter into bf16 base weights ([scripts/merge_adapter.py](../scripts/merge_adapter.py)) and serve the merged checkpoint via vLLM, which keeps the post-fine-tune eval on the same fast inference path the base used (≈500 tok/s aggregate under 500-sequence batching). Without this step, fine-tuned eval would fall back to HF generate at ~30 tok/s, i.e. a 10+ hour run.

## Infrastructure

The frontier evaluation ran from a single sbatch on the UW Hyak Klone cluster (`ckpt-all` partition, CPU-only, 4 cpus / 8 GB / 2h54m wall, all API calls). The Qwen3 base evaluation ran on the same cluster on a single A40 GPU via vLLM (3h25m wall, 6.2M output tokens generated). Pre-launch readiness: preflight gate that hits each provider with one smoke problem; `tenacity` retry (4 attempts, exponential backoff on 429/5xx); per-problem JSON writes to disk + SHA-256-keyed inference cache so preemption or interruption is survivable; per-model log files for independent tailing.

## Links

- Per-model raw + graded JSONs: [results/full/](../results/full/)
- Consolidated NOTES: [results/full/NOTES.md](../results/full/NOTES.md)
- OpenAI filter artifact: [results/full/openai-flagged-problems.md](../results/full/openai-flagged-problems.md)
- GPT-5 miss-rate investigation: [docs/gpt-missrate-analysis.md](./gpt-missrate-analysis.md) (pre-registration: [docs/gpt-missrate-preregistration.md](./gpt-missrate-preregistration.md))
- Smoke + judge-review report: [day1_report.md](./day1_report.md)
- Run 4 paired analysis: [docs/run4_analysis.md](./run4_analysis.md)
- Multi-config comparison: [docs/run4_full_comparison.md](./run4_full_comparison.md)
