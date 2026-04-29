# Dr. GRPO on Qwen3-1.7B — pre-registration

Pre-registered before launching the training run. Locks the hypothesis, recipe, and interpretation language for each possible outcome. Treats pre-registration the same way the Run 4 plan did: language committed before measurement, so that the post-result writeup is honest about what each magnitude actually means.

## Hypothesis

The QLoRA-SFT runs (Run 2/3/4) failed because training on long teacher traces amplified the convergence-failure mode the base was already prone to (saturation at the 16K token cap without committing to a final answer). [Dr. GRPO (Liu et al., 2503.20783)](https://arxiv.org/abs/2503.20783) documents that vanilla GRPO has a structurally identical length-amplification bias and proposes a bias-corrected variant. **Hypothesis: applying Dr. GRPO directly to Qwen3-1.7B addresses the failure mode that bit our SFT runs and produces a non-negative paired delta vs. the base.**

This is a direct test of the diagnosis from the Run 4 writeup. It is not a prediction about how *much* lift Dr. GRPO produces — only about whether the bias-corrected RL variant avoids the SFT failure mode. Magnitude is what we measure.

## Recipe (locked before run)

- **Base**: `Qwen/Qwen3-1.7B` (same as Run 4; preserves paired comparison vs. existing n=500 eval).
- **Method**: Dr. GRPO via TRL `GRPOConfig(loss_type="dr_grpo")`, LoRA wrapper.
- **LoRA**: r=64, alpha=128, target = q/k/v/o + gate/up/down.
- **Optimization**: LR `1e-6`, KL `beta=0.04`, group size `num_generations=4` (cut from default 8 for 48GB feasibility), `max_completion_length=3000` during training (smaller than the 16K eval cap to fit memory), per-device batch 1 × grad-accum 4.
- **Steps**: 200 (target). Save every 25.
- **Reward**: cheap-grader correctness only — `1.0` if `extract_answer(rollout)` matches gold (normalized), else `0.0`. No LLM judge during training (judge is for post-training eval, same as the SFT runs).
- **Prompts**: `data/splits/train_english.jsonl` (3,596 problems). On-policy: model generates rollouts on these; reward signal comes from gold-answer match.
- **Inference at eval time**: identical to base + Run 2/3/4 evals — vLLM, thinking-on, `max_new_tokens=16384`, `temperature=0` (greedy). Paired n=500 vs base.

## Pre-flight gate

`slurm/calibrate_dr_grpo.sbatch` runs 5 steps on 50 prompts to verify 48GB feasibility and measure tokens/sec. Full-run launch is conditional on calibration completing without OOM. Expected calibration outcome: ~3-5 min/step, no OOM. If OOM at group size 4, fallback options in order: drop to `num_generations=2`; cut `max_completion_length` to 2000; request 2×48GB.

## Pre-registered interpretation buckets

Locked before measurement. Result will land in exactly one bucket; the prose for that bucket goes into the writeup verbatim.

| Paired Δ vs. Qwen3-1.7B base on n=500 | Bucket | Interpretation |
|---|---|---|
| ≥ +5 pp | **Dr. GRPO works** | The bias-corrected RL variant clearly addresses the saturation-amplification mechanism. The result is a clean lift over a base that was already SFT+GRPO-trained via distillation. Closes the project's main loop: SFT failed → diagnosed length-amplification → bias-corrected RL fixed it. |
| 0 to +5 pp | **Modest improvement, mechanism partially confirmed** | Dr. GRPO produces a measurable but small lift. Consistent with the bias-correction helping, but the small magnitude is also consistent with the iterative-RL diminishing-returns argument (Qwen3-1.7B already inherited RL behavior via distillation, so headroom is limited). The cleanest read is: bias correction matters, but the base is close to its small-model ceiling. |
| -5 to 0 pp | **No transfer** | Dr. GRPO at this scale, on this base, doesn't extract additional lift. Multiple plausible reasons: (a) the saturation-amplification mechanism we diagnosed in SFT isn't the dominant failure mode under RL; (b) the base is close enough to a local optimum that no method we tested can move it; (c) our recipe (group size 4, LR 1e-6, 200 steps) is undertuned for this base. We can't distinguish (a)-(c) from one run. |
| < -5 pp | **RL collapse** | Dr. GRPO actively damaged the model. Strongest plausible cause: KL coefficient too low, model drifted off-distribution. Less likely: bias correction doesn't generalize beyond the math-specialized bases tested in the original paper. Either way, a notable failure worth documenting. |

## What goes in the writeup regardless of outcome

- The locked recipe + hyperparameters above.
- The calibration output (per-step wall time, GPU utilization).
- The paired McNemar p-value on the n=498 / 500 intersection.
- The transition matrix vs. base.
- Figure A (miss-mode decomposition) regenerated with the Dr. GRPO bar included — does saturation rate go up, down, or stay flat compared to the base?
- The bucket-prose above for the bucket the result lands in.

## What this experiment does NOT test

- **Vanilla GRPO vs. Dr. GRPO ablation.** That would require a second training run with `loss_type="grpo"` instead of `"dr_grpo"`. Out of scope for this run; would be a follow-up.
- **Different base.** No Qwen2.5-Math-1.5B comparison run; sticking with Qwen3-1.7B for paired-comparison continuity with the existing eval.
- **Larger compute / more steps.** 200 steps × group 4 × eff batch 4 is what fits in the wall-time window. Bigger budgets might produce different magnitudes.
