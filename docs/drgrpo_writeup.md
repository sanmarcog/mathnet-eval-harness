# Run 5: Dr. GRPO on Qwen3-1.7B — bias-corrected RL on a saturation-prone base

Pre-registration: [docs/dr_grpo_plan.md](dr_grpo_plan.md). The pre-reg locked the hypothesis, recipe, and four interpretation buckets (with prose for each) before the run. The result lands in the **"-5 to 0 pp = no transfer"** bucket; that bucket's prose is reproduced verbatim below.

## Headline

| | Correct | Accuracy on n=500 |
|---|---|---|
| Qwen3-1.7B base | 184/500 | **36.8%** |
| Qwen3-1.7B + Dr. GRPO *(this run)* | 163/500 | **32.6%** |
| **Paired delta** | | **-4.2 pp** |

Paired McNemar exact two-sided **p = 0.024** (79 discordant pairs; 50 base-only-correct vs 29 drgrpo-only-correct). Statistically significant regression — Dr. GRPO does *not* reach base parity.

## Five-way comparison

| config | acc on n=500 | Δ vs base | paired p |
|---|---|---|---|
| Qwen3-1.7B base | 36.8% | — | — |
| **Qwen3-1.7B + Dr. GRPO** | **32.6%** | **-4.2** | **0.024** |
| Qwen3-1.7B + Run 4 self-distill | 28.8% | -8.0 | <0.001 |
| Qwen3-1.7B + Run 3 (Run 2 + boxed augmentation) | 3.8% | -33.0 | <0.001 |
| Qwen3-1.7B + Run 2 (Alibaba recipe-match) | 3.0% | -33.8 | <0.001 |

Dr. GRPO is the **best of the four trained runs** — the smallest regression — but still doesn't beat the unmodified base.

## Pre-registered interpretation (locked before run)

> **"-5 to 0 pp = No transfer."** Dr. GRPO at this scale, on this base, doesn't extract additional lift. Multiple plausible reasons: (a) the saturation-amplification mechanism we diagnosed in SFT isn't the dominant failure mode under RL; (b) the base is close enough to a local optimum that no method we tested can move it; (c) our recipe (group size 4, LR 1e-6, 200 steps) is undertuned for this base. We can't distinguish (a)–(c) from one run.

The result was stronger than that prose anticipates on one specific dimension: Dr. GRPO **partially recovered from Run 4's regression** (-8 pp → -4.2 pp), and the mechanistic data shows the bias correction did its job on saturation rate (below). So (a) is partially refuted: the bias correction *did* address part of the failure mode, just not enough to push past base.

## Mechanistic reading: bias correction reduced saturation rate but not to parity

Saturation (rollouts hitting the 16,384-token cap without ever emitting `\boxed{...}`) was the diagnosed failure mode in Run 4. Dr. GRPO is the bias-corrected RL variant that addresses exactly that bias. So the question worth asking: *did saturation actually go down?*

| | Total misses | Saturated at 16K | % of misses saturated |
|---|---|---|---|
| Qwen3-1.7B base | 316 | 157 | **50%** |
| Qwen3-1.7B + Run 4 SFT | 356 | 198 | **56%** |
| Qwen3-1.7B + Dr. GRPO | 337 | 151 | **45%** |

Run 4 saturated 6 pp more than base (56% vs 50%); Dr. GRPO saturates 5 pp less than base (45% vs 50%). The bias correction *did* address the length-amplification mechanism. Relative to Run 4, Dr. GRPO cut the saturation rate by 11 pp.

But the total miss count went *up* relative to base (337 vs 316). Dr. GRPO trades **saturated-without-answer** failures for **wrong-but-committed** failures. The model is now more decisive — it terminates more often — but its confident answers are wrong on 50 problems where the base was confidently right.

## Paired transition matrix vs base

|   | base ✓ | base ✗ |
|---|---|---|
| **drgrpo ✓** | 134 | 29 *(improved)* |
| **drgrpo ✗** | 50 *(regressed)* | 287 |

29 problems gained, 50 lost. Net -21. The regression is paired and statistically significant (McNemar p = 0.024).

## vs Run 4: Dr. GRPO recovers about half the SFT regression

| | Correct | Accuracy |
|---|---|---|
| Qwen3-1.7B + Run 4 | 144/500 | **28.8%** |
| Qwen3-1.7B + Dr. GRPO | 163/500 | **32.6%** |
| **Delta** | | **+3.8 pp** |

Paired McNemar **p = 0.0558** (89 discordant; 35 vs 54). Borderline-significant by conventional thresholds. Of the 54 problems Dr. GRPO got right that Run 4 missed, the recovery is concentrated in the judge-resolved category (+15 vs Run 4 on the judge path, +7 on exact-match). Dr. GRPO produces semantically correct answers — sometimes in formats only the LLM judge accepts — that Run 4 wasn't producing.

## What this confirms (and what it doesn't)

**Confirms:** the length-amplification mechanism diagnosed in Run 4 was real and the bias-corrected RL variant *partially* addresses it. Saturation rate dropped from 56% (Run 4) to 45% (Dr. GRPO) of misses; the model recovered 19 net problems vs Run 4.

**Does not confirm:** that Dr. GRPO alone is sufficient to push past the base. The bias correction is a real but undersized intervention. The *direction* matches the diagnosis; the *magnitude* doesn't reach parity.

**The natural next inference**, supported by the post-hoc literature check (DeepSeek-R1, Qwen3 official recipe, Red Hat's small-model R1 reproduction): **bias-corrected RL needs cold-start SFT first**. The canonical small-model recipe is curated short-trace SFT → GRPO. Our Run 2/3/4 SFT phase used the *wrong* kind of cold-start data (long teacher traces), which amplified the failure mode instead of preparing the model for RL. Dr. GRPO inherited a partially-amplified base and did what it could.

## Recipe deviations from pre-registration

The pre-reg locked the recipe in [docs/dr_grpo_plan.md](dr_grpo_plan.md). Three deviations from the locked spec, documented for reproducibility:

1. **`max_completion_length=12000` instead of pre-reg's 3000.** Reason: with the chat template properly applied (see deviation 2) and thinking enabled, the rollout distribution naturally fits in ~2-4K tokens, well under 12000. The pre-reg's 3000 would have produced 100% clipping on the few rollouts that genuinely needed more. The pre-reg listed cuts to 2000 as the documented OOM-fallback; the increase to 12000 was not pre-anticipated. We arrived here after observing 100% clipping with 4000 (the model couldn't terminate inside the budget) and confirming 12000 fit on the A100 80GB without OOM.

2. **Chat template applied via `tokenizer.apply_chat_template(..., enable_thinking=True)` before TRL.** Pre-reg did not specify chat-template handling. TRL 0.19.0 GRPOTrainer does *not* apply the tokenizer's chat template to plain-string `{"prompt": str}` rows ([trl/data_utils.py L93](https://github.com/huggingface/trl/blob/v0.19.0/trl/data_utils.py#L93) — `is_conversational` returns False for strings). Without explicit chat templating, the model received raw text continuations and never opened a `<think>` block, generating undirected continuations until the cap. Pre-formatting the prompt with `apply_chat_template` (Unsloth's documented pattern) is what made the run produce well-formed thinking traces.

3. **Reward-extraction regex extended for nested braces.** The pre-reg's `_BOXED_RE` regex `r"\\boxed\{([^{}]+)\}"` chokes on nested braces like `\boxed{\frac{1}{2}}`. Replaced with a balanced-brace scanner during calibration. Without this fix, ~5-10% of correct-but-fraction-formatted answers would silently get `pred=None` and reward=0, biasing the gradient against fraction-formatted outputs.

Two attempts that were rolled back during calibration (each ran one calibration job, then was reverted as the wrong fix for our specific failure mode):

- **`mask_truncated_completions=True`**: TRL flag that zeros truncated rollouts in the loss. Documented as the fix for "noisy rewards on truncated completions" — but our reward function is *robust* to truncation (it scans the whole text for `\boxed{}`), so masking strictly threw away signal. Rolled back after observing zero-only training with `num_tokens=216` per step.
- **`/no_think` prompt suffix**: rolled back after research confirmed `/no_think` is a chat-template directive, not a base-model behavior. Without TRL applying the chat template (see deviation 2), the suffix was inert text.

Calibration log (for the curious): [logs/dr_grpo_calib_*.log](../logs/) on the Hyak side.

## What's next

The natural follow-on is the canonical small-model recipe the literature points to: **cold-start SFT (the right kind) → GRPO**.

- "The right kind" of cold-start SFT means *short, in-distribution* traces — rejection-sampled from the model itself (or a same-tier model). This is exactly what RFT (rejection-sampled fine-tuning) is, and Week 2 of this project is the RFT chapter.
- A second Dr. GRPO run on top of an RFT-cold-started base would test whether the failure was "Dr. GRPO doesn't work on this base" (current data) or "Dr. GRPO doesn't work *without proper cold-start* on this base" (hypothesis).

Within the current chapter, two ablations would tighten the conclusion if compute permits:

- **`num_generations=8` + format-credit reward** (anirudhmsu Qwen3-1.7B GRPO pattern). Boots within-group reward variance. Format-credit reward branch is implemented and behind a flag in [scripts/train_dr_grpo.py](../scripts/train_dr_grpo.py) (`--format-credit`); not yet run.
- **More steps** (200 → 500). The reward trajectory plot (steps 71-180) was roughly stationary; we cannot distinguish "trained to convergence" from "would have improved with more steps." Cheap to run if A100 queue cooperates.

These would refine, not redirect, the result. The dominant chapter conclusion — *bias correction alone, on this base, falls short of parity* — would not change.

## Artifacts

- Pre-registration: [docs/dr_grpo_plan.md](dr_grpo_plan.md)
- Paired analysis vs base: [docs/drgrpo_vs_base_analysis.md](drgrpo_vs_base_analysis.md)
- Paired analysis vs Run 4: [docs/drgrpo_vs_run4_analysis.md](drgrpo_vs_run4_analysis.md)
- Diagnostic figures: [results/figures/miss_mode_decomposition.png](../results/figures/miss_mode_decomposition.png), [transition_matrix_run4_vs_base.png](../results/figures/transition_matrix_run4_vs_base.png), [per_competition_delta_run4.png](../results/figures/per_competition_delta_run4.png)
- Per-problem result JSONs: [results/full/qwen3-1.7b-drgrpo/](../results/full/qwen3-1.7b-drgrpo/)
- Training script: [scripts/train_dr_grpo.py](../scripts/train_dr_grpo.py)
- Slurm scripts: [slurm/calibrate_dr_grpo.sbatch](../slurm/calibrate_dr_grpo.sbatch), [slurm/train_dr_grpo.sbatch](../slurm/train_dr_grpo.sbatch), [slurm/eval_dr_grpo.sbatch](../slurm/eval_dr_grpo.sbatch), [slurm/postprocess_dr_grpo.sbatch](../slurm/postprocess_dr_grpo.sbatch)
