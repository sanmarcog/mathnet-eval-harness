# Findings — frontier APIs, an open-weights base, and four QLoRA attempts

This document is the public methodology + results writeup. It complements the [scoreboard in the README](../README.md#scoreboard) with the full caveats, the headline parity finding, the negative result on QLoRA fine-tuning at 1.7B, and the operational lessons from running the evaluation.

The lead is the parity finding (Qwen3-1.7B ≈ GPT-5.4 Mini at 36.8% / 36.7%). The intellectual contribution is the negative-result diagnosis: across four QLoRA configurations spanning every sensible knob, none surpassed the post-trained base, and we have a clean mechanistic explanation for why.

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

## The headline: open-weights parity at the cheap-commercial waterline

The project was originally framed as: *can a fine-tuned 1.5B open model push past GPT-5.4 Mini at 36.7%?* That bar was set on Day-2 with the API scoreboard.

On Day-3 we added Qwen3-1.7B as the open-weights baseline. **It scored 36.8% out of the box, with no fine-tuning** — at parity with the same Mini target.

We've kept that fact in the headline rather than rewriting history. The re-framed question is more current and more interesting: with the open ecosystem already at the cheap-commercial-tier waterline, **where does fine-tuning still add value?**

### Caveat: not a constraint-matched comparison

The 36.8% / 36.7% near-tie is "both models in their preferred inference mode": Qwen3 thinking-on at 16K tokens via vLLM, Mini with OpenAI's default reasoning configuration. Identical-constraint comparisons (e.g. capping Qwen3 to a comparable reasoning budget) would land somewhere different.

The headline finding isn't *"these are identical models"* — it's *"the open ecosystem has caught up enough that the useful inference mode of a 1.7B base ties the useful inference mode of the cheap commercial tier."*

### Same 63% miss rate, different causes

GPT-5.4 Mini and Qwen3-1.7B base both `miss` on 63% of problems. Looks similar; isn't.

**Mini misses are mostly genuine wrong answers.** From the Day-3 [40-sample manual audit](./gpt-missrate-analysis.md): 85% of sampled GPT-5 misses were genuine model errors, only 10% grader artifacts. The model commits to an answer; the answer is just wrong.

**Qwen3-1.7B base misses are dominated by convergence failure.** From the 50-problem 16K-pilot:

- 35% of outputs hit the 16K-token ceiling
- 65% emit `\boxed{...}` somewhere in the output
- 33% saturate **and** never emit `\boxed{...}` — the model thinks itself out without ever committing

So Qwen3's 63% miss rate decomposes (roughly) into ~half "thought-loop failed to converge" and ~half "wrong but committed answer." This is the mechanistic setup for the fine-tuning question: *failure to converge* is exactly what fine-tuning on solution+answer pairs (with clean `\boxed{...}` conclusions) was supposed to fix.

It didn't. The fine-tuning attempts amplified the convergence-failure mode rather than reducing it. Why is the rest of this document.

## The negative finding: across four QLoRA configurations, none surpassed the post-trained base

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

Hypothesis (per [LIMO 2502.03387](https://arxiv.org/abs/2502.03387) / [STaR 2203.14465](https://arxiv.org/abs/2203.14465) / [RFT 2308.01825](https://arxiv.org/abs/2308.01825)): training on the *model's own* correct reasoning may elicit latent capability without the noise of MathNet's heterogeneous gold answers. The Run 2/3 training data had MathNet's prose-form gold (e.g. *"All n that are multiples of 4"*) inside a `\boxed{}` — confusing supervision. Run 4 replaces that with traces the base actually generated and got right.

Critical caveat from [Why Does Self-Distillation (Sometimes) Degrade Reasoning](https://arxiv.org/abs/2603.24472) (Kim et al., 2603.24472), which reports up to -40% reasoning regression on Qwen3-**8B**, DeepSeek-Distill-Qwen-7B, and Olmo3-7B-Instruct (the paper does not test Qwen3-1.7B): self-distillation can hurt generalization. To minimize that risk we preserve full `<think>...</think>` traces in the training targets, train with `enable_thinking=True` in the chat template (matches inference), `max_seq_length=8192` to fit the long traces, and use conservative LR `1e-5` with 1-2 epochs.

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

Self-distillation produced a *much milder* regression than Runs 2 and 3 (which sat at -33 to -34 pp), but the fine-tune still ended up below the post-trained base.

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

The mechanism we observed is not the one [Kim et al. (2603.24472)](https://arxiv.org/abs/2603.24472) document for self-distillation degradation in larger Qwen3 / DeepSeek / Olmo models — they identify "conditioning the teacher on rich information suppresses uncertainty expression, hurting OOD." Our failure mode is different and is our own diagnosis from the eval data: training on base's long reasoning traces taught the model to *think longer*, amplifying the convergence-failure mode the base was already prone to. The base sat at a precarious equilibrium between useful long thinking and runaway thinking; even a soft, faithful retraining tipped it past the cliff.

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

| Run | Base | Recipe / single change | Train data | Eval acc (paired Δ vs base) |
|---|---|---|---|---|
| **Base** | Qwen3-1.7B | (no fine-tune) | — | **36.8% (anchor)** |
| Run 1 | Qwen2.5-1.5B-Instruct | default LR 2e-4, no completion-only-loss | English MathNet (3,596) | 10.0% (-6 pp paired n=150 vs Qwen2.5 base 16%) |
| Run B | Qwen2.5-1.5B-Instruct | + `completion_only_loss=True` | (same as Run 1) | 8.0% (-8 pp paired) |
| Run 2 | Qwen3-1.7B | recipe-match Alibaba's Qwen2.5-Math 1.5B (LR 2e-5, 3 epochs, eff batch 128, multilingual filtered) | 11,648 rows | **3.0% (-33.8 pp)** |
| Run 3 | Qwen3-1.7B | + boxed-answer augmentation (every row gets `\\boxed{X}` appended) | 11,648 rows | **3.8% (-33.0 pp)** |
| Run 4 | Qwen3-1.7B | + self-distilled training data (base's own correct outputs, traces preserved) | 146 rows | **28.8% (-8.0 pp paired, p < 10⁻³)** |

D-LR (LR=5e-5 midpoint ablation) was started and cancelled mid-flight when mid-eval pause cost was projected to exceed walltime; no held-out accuracy. Loss curve was numerically stable.

## What it would take to actually beat the open base

At 1.7B, the post-trained Qwen3 base appears to sit at a local optimum hard to disturb without breaking. Self-distillation reduced the *damage* of fine-tuning (Runs 2/3 = -34 pp; Run 4 = -8 pp), but didn't push above. **Surpassing the open base at this size likely requires methods structurally different from any we tested.** Three candidates, ordered by how directly they address the convergence-failure mechanism:

1. **RL — [GRPO](https://arxiv.org/abs/2402.03300) / [rStar-Math](https://arxiv.org/abs/2501.04519).** Avoids the supervision-length problem entirely. The model generates its own traces and the reward is on the final answer, so there is no teacher distribution biasing trace length upward. The 16K saturation issue fundamentally goes away because there is no demonstration to imitate.
2. **Distillation from a stronger external teacher.** Sonnet 4.6 emits decisive ~5K-7K-token solutions on these same problems; the base emits noisy ~14K-token solutions. Training on Sonnet's trace distribution (or DeepSeek-R1's) would relabel the supervision target with cleaner, more decisive reasoning — addressing exactly the "trained to think longer" mechanism. Cost-prohibitive for the project budget but the most direct fix.
3. **Continued pretraining on a larger math corpus** (e.g. [Llemma](https://arxiv.org/abs/2310.10631) and its Proof-Pile-2 dataset of scientific papers, math web data, and mathematical code). Different scale entirely. Useful as an upstream step *before* any of the above; not a replacement for them.

These are documented as Week 2-4 follow-on work. None are addressable in Week 1.

## Frontier-tier findings

1. **Opus 4.7 is clearly the strongest**, but the spot-check sample size means we treat the 84% as indicative of the ceiling, not as a precise number.
2. **Sonnet 4.6 beats GPT-5.4 by 7 pp** (65.0% vs 57.8%) on a sample large enough that this is not noise. The Anthropic lineage outperforms the OpenAI lineage on MathNet-style olympiad problems in our setup, and at comparable eval cost ($10.35 vs $9.52).
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
- **Judge model = Claude Sonnet 4.6.** The judge's job is pairwise equivalence, not problem-solving; a 9-problem Day-1 calibration found 0 false positives. Same family as the #3 model on the scoreboard, so a second-judge cross-check would be a reasonable follow-up.
- **Saturation cutoff is identical across all four Qwen3-1.7B runs.** All four sbatches (`eval_qwen3_base.sbatch`, `eval_qwen3_run2.sbatch`, `eval_qwen3_run3.sbatch`, `eval_qwen3_run4.sbatch`) pass `--max-new-tokens 16384` to the same eval script with `temperature=0` on the same vLLM backend. The figure A "saturated" comparison is uncontaminated.

### Grader path breakdown (Sonnet 4.6, n=500)

Correctness is assigned by a 4-layer grading pipeline that tries cheap, deterministic checks before falling back to an LLM judge. On the Sonnet full run:

| Path | Count | Share |
|---|---|---|
| `exact` | 87 | 17.4% |
| `normalized` | 30 | 6.0% |
| `symbolic` (sympy) | 16 | 3.2% |
| `judge` (LLM) | 192 | 38.4% |
| `miss` | 175 | 35.0% |

133 / 325 = **40.9% of correct grades resolve on the objective (non-judge) layers**. The LLM judge catches the other 59% — set-valued answers, notation synonyms, prose-wrapped solutions. Day-1 judge calibration on 9 accepted answers found 0 false positives. See [results/figures/grader_paths.png](../results/figures/grader_paths.png) for the per-model distribution.

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

1. **Mid-training eval was too narrow on Run 1.** The QLoRA training loop logs a held-out 50-problem accuracy at 25%/50%/75% of training plus each epoch boundary. Run 1 was capped at 1024 output tokens with greedy decoding and the cheap-grader-only path — no LLM judge. All four checkpoints reported 1/50 = 2%, which read as "training is broken." It wasn't. 1024 tokens is under 10% of the median convergent output length on this benchmark; the model hadn't yet emitted a `\boxed{...}` by the time the cap hit, so `extract_answer` returned `None` on 49/50. **Fixed for Run 2:** mid-eval token cap bumped to 4096 in `src/mathnet_eval/training.py`; default still no judge in mid-eval to avoid runaway API spend during long training jobs.
2. **Pre-registered ranges turned out to be too pessimistic on Qwen3 base.** Range was 14-22% (anchored to Qwen 2.5 numbers). Actual landed at 36.8%, well outside. We're treating pre-registration the way it's meant to be treated — **as a flag for when reality differs from your prior, not as a target to defend.** When the prior misses, document the miss and update the model.
3. **vLLM does not load PEFT adapters in our eval-script wiring.** Post Run 2 training, we merge the LoRA adapter into bf16 base weights ([scripts/merge_adapter.py](../scripts/merge_adapter.py)) and serve the merged checkpoint via vLLM, which keeps the post-fine-tune eval on the same fast inference path the base used (≈500 tok/s aggregate under 500-sequence batching). Without this step, fine-tuned eval would fall back to HF generate at ~30 tok/s, i.e. a 10+ hour run.

## Infrastructure

The frontier evaluation ran from a single sbatch on the UW Hyak Klone cluster (`ckpt-all` partition, CPU-only, 4 cpus / 8 GB / 2h54m wall, all API calls). The Qwen3 base evaluation ran on the same cluster on a single A40 GPU via vLLM (3h25m wall, 6.2M output tokens generated). Pre-launch readiness: preflight gate that hits each provider with one smoke problem; `tenacity` retry (4 attempts, exponential backoff on 429/5xx); per-problem JSON writes to disk + SHA-256-keyed inference cache so preemption or interruption is survivable; per-model log files for independent tailing.

## Links

- Per-model raw + graded JSONs: [results/full/](../results/full/)
- Consolidated NOTES: [results/full/NOTES.md](../results/full/NOTES.md)
- OpenAI filter artifact: [results/full/openai-flagged-problems.md](../results/full/openai-flagged-problems.md)
- GPT-5 miss-rate investigation: [docs/gpt-missrate-analysis.md](./gpt-missrate-analysis.md) (pre-registration: [docs/gpt-missrate-preregistration.md](./gpt-missrate-preregistration.md))
- Day-1 smoke + judge-review report: [day1_report.md](./day1_report.md)
- Run 4 paired analysis: [docs/run4_analysis.md](./run4_analysis.md)
- Multi-config comparison: [docs/run4_full_comparison.md](./run4_full_comparison.md)
