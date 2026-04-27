# Findings — frontier APIs and an open-weights base on 500 MathNet problems

This document is the public methodology + results writeup. It complements
the [scoreboard in the README](../README.md#scoreboard) with the full
caveats, the primary findings, and the operational lessons from running
the evaluation. Day-2 covered the five frontier APIs; Day-3 added the
open-weights Qwen3-1.7B base for context, which produced the project's
narrative pivot (see [Re-anchoring the project](#re-anchoring-the-project)).

## Scoreboard

| Model | N scored | **Accuracy** | Eval cost |
|---|---|---|---|
| Claude Opus 4.7 *(spot-check)* | 100 | **84.0%** | $6.14 |
| Gemini 3 Pro *(partial)* | 240 / 300 | **73.3%** | $13.55 |
| Claude Sonnet 4.6 | 500 | **65.0%** | $10.35 |
| GPT-5.4 | 495 / 500 | **57.8%** | $9.52 |
| **Qwen3-1.7B base** *(open, thinking-on, vLLM, 16K)* | 500 | **36.8%** | — |
| GPT-5.4 Mini | 498 / 500 | **36.7%** | $1.51 |
| Qwen3-1.7B + Run 4 self-distill *(ours)* | 500 | _TBD — pending eval_ | — |
| **API total spend** |  |  | **$41.06** |

Denominator is `n_scored`; missing problems on GPT are OpenAI safety-filter
rejections (documented below). Gemini finished at N=240 due to a preview-model
daily quota cap. Qwen3 cost is the academic-cluster compute on a single A40
GPU (no API billing).

## Re-anchoring the project

The project was originally framed as: *can a fine-tuned 1.5B open model push
past GPT-5.4 Mini at 36.7%?* That bar was set on Day-2 with the API
scoreboard.

On Day-3 we added Qwen3-1.7B as the open-weights baseline. **It scored 36.8%
out of the box, with no fine-tuning** — at parity with the same Mini target.

We've kept that fact in the headline rather than rewriting history. The
re-framed question is more current and more interesting: with the open
ecosystem already at the cheap-commercial-tier waterline, **where does
fine-tuning still add value?** The remaining `miss`-rate analysis (below)
gives a concrete hypothesis for what QLoRA on solution+answer training data
should target.

The original Run 1 attempt was on Qwen2.5-1.5B-Instruct with a smaller LoRA
(r=16) and English-only training data. Mid-training eval reported 2%, which
turned out to be a measurement artifact (1024-token cap was under 10% of
median convergent length on this benchmark — see
[mid-eval token bump methodology note](#methodology-notes)). Run 2 picks
up Qwen3-1.7B with the kitchen-sink config: r=64, multilingual training data
(14,585 rows), `completion_only_loss`, and `max_seq_length=4096`.

## Grader path breakdown (Sonnet 4.6, n=500)

Correctness is assigned by a 4-layer grading pipeline that tries cheap,
deterministic checks before falling back to an LLM judge. On the Sonnet full
run:

| Path | Count | Share |
|---|---|---|
| `exact` | 87 | 17.4% |
| `normalized` | 30 | 6.0% |
| `symbolic` (sympy) | 16 | 3.2% |
| `judge` (LLM) | 192 | 38.4% |
| `miss` | 175 | 35.0% |

133 / 325 = **40.9% of correct grades resolve on the objective (non-judge) layers**.
The LLM judge catches the other 59% — set-valued answers, notation synonyms,
prose-wrapped solutions. Day-1 judge calibration on 9 accepted answers found
0 false positives.

See [results/figures/grader_paths.png](../results/figures/grader_paths.png) for the
per-model distribution.

## Methodology caveats

These are prominent on purpose — the headline numbers mean very different
things with vs. without them.

- **Opus 4.7 is N=100 by design, a spot-check not a full 500.** The $60 budget
  ceiling drove this choice. 95% CI on 84% at N=100 is roughly ±8 pp — treat
  the number as indicative of the ceiling, not as a precise accuracy measure.
- **Gemini 3 Pro ran with `thinking_budget=4096`**, other models ran with
  default reasoning settings. Cost-control decision based on a 15-problem
  calibration showing median 5,454 thoughts / max 15,730 per problem under
  default thinking. A default-thinking run would plausibly score 1-3 pp
  higher.
- **Gemini 3 Pro is N=300 by design** (rescoped from 500 during calibration
  to fit budget) and currently N=239 in practice due to a preview-model
  daily quota cap. The remaining 61 are deferred to a future run.
- **OpenAI filtered 5 / 500 GPT-5.4 and 2 / 500 GPT-5.4 Mini prompts** with
  `invalid_prompt` 400s. Accuracy denominators are `n_scored` (495 and 498)
  rather than 500 for those two models. See
  [results/full/openai-flagged-problems.md](../results/full/openai-flagged-problems.md).
- **Judge model = Claude Sonnet 4.6.** The judge's job is pairwise
  equivalence, not problem-solving; a 9-problem Day-1 calibration found 0
  false positives. Same family as the #3 model on the scoreboard, so a
  second-judge cross-check would be a reasonable follow-up.
- **Qwen3 vs Mini are not constraint-matched.** The 36.8% / 36.7% near-tie
  is "both models in their preferred inference mode": Qwen3 thinking-on at
  16K tokens via vLLM, Mini with OpenAI's default reasoning configuration.
  Identical-constraint comparisons (e.g. capping Qwen3 to a comparable
  reasoning budget) would land somewhere different. The headline finding
  isn't "these are identical models" — it's "the open ecosystem has
  caught up enough that the *useful inference mode* of a 1.7B base ties
  the *useful inference mode* of the cheap commercial tier."

## The 63% miss rate: same number, different causes

GPT-5.4 Mini and Qwen3-1.7B base both `miss` on 63% of problems. Looks
similar; isn't.

**Mini misses are mostly genuine wrong answers.** From the Day-3
[40-sample manual audit](./gpt-missrate-analysis.md): 85% of sampled GPT-5
misses were genuine model errors, only 10% grader artifacts. The model
commits to an answer; the answer is just wrong.

**Qwen3-1.7B base misses are dominated by convergence failure.** From the
50-problem 16K-pilot:

- 35% of outputs hit the 16K-token ceiling
- 65% emit `\boxed{...}` somewhere in the output
- 33% saturate **and** never emit `\boxed{...}` — the model thinks itself
  out without ever committing

So Qwen3's 63% miss rate decomposes (roughly) into ~half "thought-loop
failed to converge" and ~half "wrong but committed answer." The two
failure modes need different fixes:

- *Wrong-but-committed* needs better reasoning (more parameters, more
  training, or both — not what QLoRA targets directly).
- *Failure to converge* needs the model to learn to wrap up and emit a
  final answer — exactly what fine-tuning on solution+answer pairs (with
  clean `\boxed{...}` conclusions) teaches.

This is the explicit hypothesis for Run 2: **QLoRA on MathNet should
disproportionately reduce convergence-failure misses, not wrong-answer
misses.** If the run lifts accuracy, expect the lift to come mostly from
the saturation-driven half of the miss bucket.

## Where Qwen3-1.7B base is strongest and weakest

Stratifying by `topics_flat` top-level prefix
([full breakdown](./qwen3_base_topic_breakdown.md)):

| Topic | n | accuracy |
|---|---|---|
| Geometry | 56 | **44.6%** |
| Algebra | 275 | 41.1% |
| Number Theory | 177 | 35.0% |
| Discrete Mathematics | 155 | **23.9%** |

A problem can be tagged with multiple top-level prefixes, so the n's sum
to more than 500.

The ~20pp gap between Geometry/Algebra and Discrete Mathematics is real
signal. Within Discrete Mathematics, **Graph Theory** is the model's hard
floor: **1/18 = 5.6%**. Within Algebra, by contrast, prealgebra runs at
46.6% and equations/inequalities at 42.9%. So the open-base 1.7B does
well on procedural/computational topics and worse on topics requiring
combinatorial case analysis. This is a useful baseline pattern to track
through Run 2.

## Findings

1. **Opus 4.7 is clearly the strongest**, but the spot-check sample size
   means we treat the 84% as indicative of the ceiling, not as a precise
   number.
2. **Sonnet 4.6 beats GPT-5.4 by 7 pp** (65.0% vs 57.8%) on a sample large
   enough that this is not noise. The Anthropic lineage outperforms the
   OpenAI lineage on MathNet-style olympiad problems in our setup, and at
   comparable eval cost ($10.35 vs $9.52).
3. **The current-gen open ecosystem is at the cheap-commercial waterline.**
   Qwen3-1.7B in its preferred inference mode (thinking-on, 16K, vLLM)
   scores 36.8%, statistically tied with GPT-5.4 Mini at 36.7%. This is
   the project's narrative pivot: the original target ("can a 1.5B
   fine-tune beat the cheap commercial tier?") is already met by the open
   base. The new question is **what value-add fine-tuning still
   provides** — see the miss-rate decomposition above.
4. **Gemini 3 Pro at 73.3% (N=240 of 300 target) with capped thinking.** Ran
   with `thinking_budget=4096` to fit budget; would plausibly score 1-3 pp
   higher with default (unbounded) thinking.
5. **The GPT-5 family has a large `miss` rate even with the LLM judge
   enabled** (209 and 315 misses across GPT-5.4 / Mini). Investigated on a
   pre-registered 40-sample manual audit:
   [docs/gpt-missrate-analysis.md](./gpt-missrate-analysis.md). 85% of
   sampled misses are genuine model errors; only 10% are grader artifacts
   (`extractor_failure` or `judge_false_negative`), below the
   [pre-registered 15% fix threshold](./gpt-missrate-preregistration.md).
   **Conclusion: numbers stand; Mini at 37% is the real peer-tier target,
   not a grader-inflated figure.**

## Operational findings

Things we did not expect, and which cost us time. Documenting them here so
the next person doesn't repeat them.

1. **Gemini 3.x preview models have a 250 RPD cap even on paid billing.**
   Not in the public rate-limits documentation that we saw. The run hit
   the cap at exactly 239 successful + 61 failing = 250 daily + some retries.
2. **AI Studio and Google Cloud billing are parallel lanes.** Credits
   added via AI Studio do not show up in Cloud-console's "$300 free trial
   credits" view; they appear in a separate "Gemini API Billing" view.
   Confusing UX; tripped us up for an hour diagnosing.
3. **Our cost estimator was under-reporting Gemini by ~6×** because
   `thoughts_tokens` weren't summed. Found by cross-checking our
   `summary.json` against the billing dashboard (thought we had spent $2.05,
   the dashboard said $13.49). Fix in
   [scripts/grade_results.py](../scripts/grade_results.py); the $13.55
   figure in the scoreboard matches the dashboard.
4. **Opus 4.7 rejects the `temperature` parameter** (reasoning-style
   model); our preflight script caught this before launch. Worth its weight
   in gold — a 500-problem failure at cost would have been painful.
5. **GPT-5.4 emitted zero reasoning tokens on our prompts** — every
   smoke-test problem showed `reasoning_tokens=0`, so its cost matched
   "visible output" estimates exactly. Gemini, by contrast, used thinking
   heavily. Anthropic defaults split the middle: Opus 4.7 runs long without
   emitting a separate reasoning-token stream.

## Infrastructure

The frontier evaluation ran from a single sbatch on the UW Hyak Klone
cluster (`ckpt-all` partition, CPU-only, 4 cpus / 8 GB / 2h54m wall, all
API calls). The Qwen3 base evaluation ran on the same cluster on a single
A40 GPU via vLLM (3h25m wall, 6.2M output tokens generated).
Pre-launch readiness: preflight gate that hits each provider with one
smoke problem; `tenacity` retry (4 attempts, exponential backoff on
429/5xx); per-problem JSON writes to disk + SHA-256-keyed inference cache
so preemption or interruption is survivable; per-model log files for
independent tailing.

## Methodology notes

A few things we got wrong on the first attempt and fixed in flight.
Documenting them so the trajectory is honest and so the next person
running this experiment doesn't repeat them.

1. **Mid-training eval was too narrow on Run 1.** The QLoRA training loop
   logs a held-out 50-problem accuracy at 25%/50%/75% of training plus
   each epoch boundary. Run 1 was capped at 1024 output tokens with
   greedy decoding and the cheap-grader-only path — no LLM judge. All
   four checkpoints reported 1/50 = 2%, which read as "training is
   broken." It wasn't. 1024 tokens is under 10% of the median convergent
   output length on this benchmark; the model hadn't yet emitted a
   `\boxed{...}` by the time the cap hit, so `extract_answer` returned
   `None` on 49/50. **Fixed for Run 2:** mid-eval token cap bumped to
   4096 in `src/mathnet_eval/training.py`; default still no judge in mid-eval
   to avoid runaway API spend during long training jobs.
2. **Pre-registered ranges turned out to be too pessimistic on Qwen3
   base.** Range was 14-22% (anchored to Qwen 2.5 numbers). Actual
   landed at 36.8%, well outside. We're treating pre-registration the
   way it's meant to be treated — **as a flag for when reality differs
   from your prior, not as a target to defend.** When the prior misses,
   document the miss and update the model.
3. **vLLM does not load PEFT adapters in our eval-script wiring.** Post
   Run 2 training, we merge the LoRA adapter into bf16 base weights
   ([scripts/merge_adapter.py](../scripts/merge_adapter.py)) and serve
   the merged checkpoint via vLLM, which keeps the post-fine-tune eval
   on the same fast inference path the base used (≈500 tok/s aggregate
   under 500-sequence batching). Without this step, fine-tuned eval
   would fall back to HF generate at ~30 tok/s, i.e. a 10+ hour run.

## Why our first QLoRA attempts regressed: a recipe story

The fine-tune side of this project did not work on the first few attempts.
We're documenting the journey explicitly because it surfaces a useful
lesson for anyone fine-tuning small instruction-tuned models on math.

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

13 problems regressed (base ✓ → adapter ✗), 4 improved (base ✗ → adapter ✓).
McNemar exact two-sided **p = 0.049**. The fine-tune was meaningfully
*worse* than the base.

### Run B — single-variable: enable `completion_only_loss`

Hypothesis: Run 1's loss was being computed on the full sequence including
the system prompt and user problem, so ~20% of gradient was wasted
reproducing the prompt instead of the answer. We enabled
`completion_only_loss=True` (mask system + user tokens) and held everything
else identical.

Pre-flight, we also wrote
[scripts/verify_response_template.py](../scripts/verify_response_template.py)
to ensure the response template `<|im_start|>assistant\n` actually appears
as a contiguous subsequence in the tokenized data — TRL's collator
silently fails to mask if it doesn't, and we wanted to rule that failure
mode out before launching a 5-hour training run. Verification passed.

**Result on n=150 paired vs base:**

| | Correct | Accuracy |
|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | 24/150 | 16.0% |
| + Run B (= Run 1 + completion_only_loss) | 12/150 | 8.0% |
| Delta vs base | | **-8.0 pp** (p = 0.008) |
| Delta vs Run 1 | | -2 pp (p = 0.58, not significant) |

Run B was *not* better than Run 1. Loss-masking was not the load-bearing
fix. The regression came from somewhere else.

### Run D-LR — cancelled mid-flight

Next single-variable hypothesis was that the **2e-4 learning rate was too
aggressive** for an instruction-tuned base on a small dataset, causing
catastrophic forgetting. We submitted Run D-LR identical to Run B with
LR cut to 5e-5 (a midpoint) and started training.

We cancelled it 1h15m in. The mid-eval callback (50 problems × up to
4096 output tokens via 4-bit + LoRA HF generate, on the slow inference
path with the unmerged adapter) was running at ~2.5 minutes per problem,
so a single mid-eval callback was projected at 2 hours. With four
callbacks per training run plus the actual training, total wall time
exceeded our 6-hour walltime allocation. The job would have been
slurm-killed before producing the held-out accuracy signal we wanted.

What we kept from D-LR: the training loss curve over the first ~25% of
training (380 steps total, got to 94). Loss dropped 1.15 → ~0.80 then
stabilized around 0.83-0.85. Confirms training at LR=5e-5 was numerically
stable; no held-out accuracy signal collected.

### Deep dive into the literature

After Run B failed to recover and D-LR was cancelled, we paused to read
what works empirically for math fine-tuning at this scale. The most
informative source was Alibaba's own
[Qwen2.5-Math Technical Report](https://arxiv.org/html/2409.12122v1) —
the only known successful math fine-tune of this model family at this
size. Their published recipe for the 1.5B model:

| | Alibaba's Qwen2.5-Math 1.5B | Our Run 1 / B |
|---|---|---|
| **Learning rate** | **2 × 10⁻⁵** (decays to 7×10⁻⁷) | 2 × 10⁻⁴ |
| **Effective batch** | **128** | 16 |
| Epochs | 3 | 2 |
| Seq length | 4,096 | 2,048 |
| Data scale | 2.5M CoT problems, RM-curated | ~3.6K problems |
| Base | Qwen2.5-1.5B (the *base*, not Instruct) | Qwen2.5-1.5B-**Instruct** |

We were **10× too high on LR**, **8× too small on batch**, **700× too
small on data**, and on the wrong base variant. Each contributes; together
they explain the regression. Other sources converge on the same range:
math fine-tuning lives at 10⁻⁵ to 10⁻⁴, with instruction-tuned bases at
the low end.

The
[catastrophic forgetting literature](https://arxiv.org/html/2512.13706)
documents the failure mode directly: Flan-T5-Base loses 64.5pp on NLI
within 1,000 steps of math-only fine-tuning. LoRA mitigates relative to
full fine-tuning but does not eliminate the problem; aggressive LR on a
narrow domain still degrades general abilities the model needed to solve
the problem.

### Decision: skip the LR ablation, adopt the published recipe

We did *not* run a learning-rate sweep to triangulate the right value.
The literature converged on 2e-5 for this exact model family at this size.
Adopting that directly was a faster path to a working result than
re-deriving it experimentally. The blog-post framing: *we deferred to
published expertise rather than re-running the ablation that produced it.*

### Run 2 — recipe-matched

Run 2's training sbatch
([slurm/train_qlora_run2.sbatch](../slurm/train_qlora_run2.sbatch))
matches the published recipe on every numerical hyperparameter:

- Learning rate **2e-5**
- Effective batch **128** (per-device 4 × grad-accum 32)
- **3 epochs**
- Sequence length **4,096**

Plus our additions: QLoRA r=64 / alpha=128 (the parameter-efficient
wrapper around the recipe; we have ~10K rows vs Alibaba's 2.5M, and rank
gives the adapter capacity to compensate); `completion_only_loss=True`;
`enable_thinking=False` in the chat template (training data has no
`<think>` traces); base `Qwen/Qwen3-1.7B` (paired with the 36.8% baseline
that anchors the headline).

We also filtered the multilingual training data
([scripts/filter_train_by_solution_length.py](../scripts/filter_train_by_solution_length.py))
to drop the 13.7% of rows with empty `solutions_markdown` and the 6.5%
with sub-100-token completions — those rows train the model to "after
this prompt, emit boxed answer" with zero reasoning content. Filtered set
is 11,648 rows, well above the 8K floor we'd set for the
filter-or-keep-unfiltered call.

The post-training eval is split into a separate sbatch
([slurm/eval_qwen3_run2.sbatch](../slurm/eval_qwen3_run2.sbatch)) so
train+merge fits comfortably in one walltime window and the n=500
thinking-on 16K-token eval runs separately.

### Run 3 — adding boxed-answer augmentation

Hypothesis: Run 2's regression was due to MathNet's `solutions_markdown`
field containing `\boxed{}` in only ~1.5% of rows. The QLoRA fine-tune
unlearned the boxed-answer convention from the base because it was almost
absent from training data.

Single change vs Run 2: each training row's solution gets an explicit
`\n\nTherefore, the final answer is $\\boxed{<final_answer>}$` appended.
A pre-launch safeguard
([scripts/verify_boxed_augmentation.py](../scripts/verify_boxed_augmentation.py))
verified the augmented text round-trips through `apply_chat_template`
and `extract_answer` on 8 sampled rows.

Result: **19/500 = 3.8%, paired delta -33.0 pp vs base, p < 10⁻⁴.**

Diagnosis post-mortem on Run 3's outputs: the augmentation *did* teach
the boxing convention back — 41.8% of Run 3 responses emit `\boxed{}`
(close to base's 65%). But the *content* inside the boxes is wrong
nearly every time (~9% correct-among-boxed vs base's ~57%). So the
fine-tune restored format but damaged math. Naive SFT on raw MathNet
solutions degrades reasoning faster than format augmentation can rescue.

### Run 4 — self-distillation from base's own correct answers

Hypothesis (per [LIMO 2502.03387](https://arxiv.org/abs/2502.03387) /
[STaR 2203.14465](https://arxiv.org/abs/2203.14465) /
[RFT 2308.01825](https://arxiv.org/abs/2308.01825)): training on the
*model's own* correct reasoning may elicit latent capability without
the noise of MathNet's heterogeneous gold answers. The Run 2/3
training data had MathNet's prose-form gold (e.g. *"All n that are
multiples of 4"*) inside a `\boxed{}` — confusing supervision. Run 4
replaces that with traces the base actually generated and got right.

Critical caveat from [Why Does Self-Distillation Degrade Reasoning
(arxiv 2603.24472)](https://arxiv.org/html/2603.24472), which
specifically observes -40% on Qwen3-1.7B with naive SFT when training
data has shorter reasoning than the model's natural depth: we preserve
full `<think>...</think>` traces in the training targets, train with
`enable_thinking=True` in the chat template (matches inference),
`max_seq_length=8192` to fit the long traces, conservative LR `1e-5`
and 1-2 epochs.

Pipeline:
1. Run base Qwen3-1.7B on ~430 train problems via vLLM, thinking-on, 16K
2. Filter for rows where `extract_answer` matches gold (cheap grader)
3. Save those as a self-distilled training set (~150 expected correct)
4. SFT same base on the distilled set (LR 1e-5, eff batch 4, 2 epochs,
   ~75 opt steps)
5. Merge → eval n=500 thinking-on 16K, paired vs base 36.8%

**Pre-registered interpretations** (locked before launch):

| Run 4 result | Interpretation | Writeup direction |
|---|---|---|
| ≥ 36.8% (≥ base) | Self-distillation works at our scale | "Curated correctness teaches without damage" |
| 30-36% | Capability preserved; no improvement | "We can preserve, can't add — needs RL or stronger teacher" |
| 10-30% | Partial collapse despite trace preservation | "Documents Qwen3-1.7B-specific failure mode" |
| ≤ 10% | Full collapse like Run 2/3 | "Strongest possible negative; literature-backed via 2603.24472" |

Result: _TBD — pending eval_.

### Summary of fine-tune attempts

| Run | Base | Recipe / single change | Train data | Eval acc (paired Δ vs base) |
|---|---|---|---|---|
| **Base** | Qwen3-1.7B | (no fine-tune) | — | **36.8% (anchor)** |
| Run 1 | Qwen2.5-1.5B-Instruct | default LR 2e-4, no completion-only-loss | English MathNet (3,596) | 10.0% (-6 pp paired n=150 vs Qwen2.5 base 16%) |
| Run B | Qwen2.5-1.5B-Instruct | + `completion_only_loss=True` | (same as Run 1) | 8.0% (-8 pp paired) |
| Run 2 | Qwen3-1.7B | recipe-match Alibaba's Qwen2.5-Math 1.5B (LR 2e-5, 3 epochs, eff batch 128, multilingual filtered) | 11,648 rows | **3.0% (-33.8 pp)** |
| Run 3 | Qwen3-1.7B | + boxed-answer augmentation (every row gets `\\boxed{X}` appended) | 11,648 rows | **3.8% (-33.0 pp)** |
| Run 4 | Qwen3-1.7B | + self-distilled training data (base's own correct outputs, traces preserved) | ~150 rows | _TBD_ |

D-LR (LR=5e-5 midpoint ablation) was started and cancelled mid-flight
when mid-eval pause cost was projected to exceed walltime; no held-out
accuracy. Loss curve was numerically stable.

## Links

- Per-model raw + graded JSONs: [results/full/](../results/full/)
- Consolidated NOTES: [results/full/NOTES.md](../results/full/NOTES.md)
- OpenAI filter artifact:
  [results/full/openai-flagged-problems.md](../results/full/openai-flagged-problems.md)
- GPT-5 miss-rate investigation:
  [docs/gpt-missrate-analysis.md](./gpt-missrate-analysis.md)
  (pre-registration: [docs/gpt-missrate-preregistration.md](./gpt-missrate-preregistration.md))
- Day-1 smoke + judge-review report: [day1_report.md](./day1_report.md)
