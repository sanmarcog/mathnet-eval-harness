# Blog draft notes — Day 5 me, read this first

These are notes-to-self for the eventual blog post. Not the post itself,
not a public document, but a place to keep the *thinking* coherent so that
future-me doesn't have to reconstruct it from commits and chat logs.

## The new headline

> **Current-gen 1.7B open-weights models match the cheap commercial tier
> out of the box. The interesting question for practitioners isn't whether
> small open models are competitive — it's where fine-tuning still adds
> value on top.**

This is the post's spine. Everything else hangs off it.

## The narrative pivot — own it

The project was originally framed as "fine-tune a small open model to
beat GPT-5.4 Mini at 36.7%." On Day-3 we ran Qwen3-1.7B as the open-weights
baseline and it scored 36.8% out of the box, with no fine-tuning.

Two ways to write this up:

- **Defensive.** Hide the original framing, claim we always meant to study
  the marginal value of fine-tuning. *Don't do this.* Hiring managers
  reading the post will sniff it out and the post becomes weaker, not
  stronger.
- **Honest.** Lead with the original target, show that the open base
  already met it, name the pivot explicitly, then ask the better question.
  *This is the move.* The pivot is evidence of process: pre-registration
  flagged that reality differed from prior, and we updated the prior.

Specific words that work:

> "We initially targeted GPT-5.4 Mini at 36.7% as the bar a fine-tuned
> 1.5B model needed to clear. On Day-3 we ran Qwen3-1.7B as the open-weights
> baseline. It scored 36.8% — already at the bar — without any fine-tuning.
> So we re-scoped the question: where does fine-tuning still help when the
> open base is already commercial-tier?"

## The technical hypothesis for Run 2

GPT-5.4 Mini and Qwen3-1.7B base both miss on 63% of problems. **Same
rate, different causes.** This is the cleanest content the post has.

**Mini misses are mostly genuine wrong answers.** From the Day-3
40-sample manual audit: 85% genuine model errors, 10% grader artifacts.
The model commits to an answer; the answer is just wrong.

**Qwen3 misses are dominated by convergence failure.** From the 50-problem
16K-pilot: 35% of outputs hit the 16K-token ceiling, and 33% saturate
**without ever emitting** `\boxed{...}` — the model thinks itself out
without committing to a final answer. The other half of Qwen3's misses
are wrong-but-committed, comparable to Mini.

So Qwen3's 63% miss rate decomposes (roughly) into:

- ~33% **convergence failure** (saturated, no boxed) → fine-tuning on
  solution+answer pairs with clean conclusions should target this
  directly
- ~30% **wrong but committed** → not what QLoRA addresses well

Run 2's explicit hypothesis: **QLoRA on MathNet should disproportionately
reduce convergence-failure misses, not wrong-answer misses.** If accuracy
goes up, expect the lift to come mostly from teaching the model to
wrap up and emit, not from teaching it new math.

If this is right, the post-Run-2 grader-path chart will show:
- saturation rate dropping (`miss` shrinks)
- `exact` and `judge` segments growing
- not much movement in the "model knew the answer" structure

## Methodology caveat I keep being tempted to gloss over

The 36.8% / 36.7% near-tie is **both models in their preferred inference
mode** — Qwen3 thinking-on at 16K via vLLM, Mini at OpenAI's default
reasoning settings. They're not constraint-matched. A careful reader
might point this out. Be clear in the post:

> "These aren't identical-constraint comparisons; they're 'each model in
> the inference mode you'd actually use it in.' That's what makes the
> finding interesting — it's a deployment-mode comparison, not a
> capability comparison."

Hiring managers won't care about this distinction; thoughtful readers
will. Get ahead of it.

## Three framings the post must lock in

These came out of the s1+GRPO synthesis + manager review on 2026-04-25.

**Data scale is a feature, not an apology.** Our 11,648 filtered multilingual rows is well within the small-curated regime that's been published to work. [s1 used 1,000 examples](https://arxiv.org/abs/2501.19393) and beat o1-preview at 32B. The relevant comparison is *not* Alibaba's 2.5M (frontier-scale); it's the small-curated end. If Run 2's lift is modest, the framing is "consistent with small-scale SFT literature," not "we didn't have enough data." This is also why distillation from Sonnet (which would balloon "data" via stronger CoT traces) is deferred — at our scale it wouldn't tell us whether QLoRA helped, only whether *better data* helped (which is uninteresting).

**Recipe-scaling caveat.** The SFT → RFT → GRPO pipeline is what DeepSeek-R1 and Qwen3-Math used, but at 7B+ with much more compute. Whether the recipe transfers to 1.7B at our compute is empirical and *part of what the project tests.* Don't write "this is the established recipe at small scale" — write "this is the playbook from larger models; we're testing whether it scales down."

**Week 2-4 sequencing changed.** Original plan had Week 2 = RAG + reranking. The literature pass made the continuous SFT → BoN → RFT story noticeably stronger than the SFT-then-pivot-to-RAG story. **Locked sequence:** Week 2 = best-of-N (in Week 1 if Run 2 lands acceptably) + RFT (Week 2 main project), Week 3 = RAG, Week 4 = GRPO **or** distillation as the "maximum-impact layer-on-top." Week 2 is locked regardless of Week 1's final number — it's about demonstrating closed-loop self-improvement, not chasing a specific delta.

**Important caveat for the test-time-scaling table.** Qwen3-1.7B base scored 36.8% at thinking-on, 16K tokens — that already *is* a form of test-time compute scaling. Adding best-of-N or budget forcing on top asks "does sampling diversity (or extended reasoning) help on a model that's already thinking hard?" That's a legitimate question, but it's not "test-time scaling vs no test-time scaling." The post should phrase the eval table as "additional test-time techniques on an already-thinking-on baseline," not as discovery of test-time scaling.

## What's evidence of process to keep visible

- The Qwen 2.5 first attempt and the pivot to Qwen3
- The 2% mid-eval that turned out to be a 1024-token measurement
  artifact, not a training failure (and the fix: bump to 4096)
- The pre-registered range that turned out to be too pessimistic, and
  the explicit acknowledgment that pre-registration is a flag, not a
  cage
- The vLLM-doesn't-load-adapters discovery and the merge-step fix that
  unblocked fast post-fine-tune eval
- **The Run 1 → Run B → cancelled D-LR sequence and what we learned.**
  Run 1 (default 2e-4) regressed -6pp vs base; Run B (single-variable
  completion-only-loss) didn't recover; the literature deep-dive
  surfaced Alibaba's published Qwen2.5-Math 1.5B recipe (LR 2e-5,
  batch 128, 3 epochs). We started a midpoint-LR ablation (D-LR at
  5e-5) but cancelled it once we noticed (a) it was going to exceed
  walltime due to the 4096-token mid-eval pause cost, and (b) the
  diagnostic value was marginal — we were going to adopt 2e-5 in
  Run 2 either way, and 5e-5 vs 2e-5 are close enough that
  interpolating from a 5e-5 point doesn't change the decision. So
  we skipped the LR ablation and went directly to the published
  recipe. Worth saying explicitly in the post: *we did not run an LR
  sweep; we adopted the literature recipe.* Hiring readers value
  knowing when an experimentalist chooses to defer to published
  expertise rather than re-derive it.

These belong in the post — they make it more credible, not less.

## What to avoid

- Don't claim QLoRA's value is "proven" until Run 2 lands. The
  hypothesis is clear; the result is not yet in.
- Don't get attached to a specific Run 2 accuracy number. If Run 2
  hits 45%, that's a real result. If it hits 38%, that's also a
  real result and still a story (ablate further: which change moved
  the needle?). Both narratives are valid.
- Don't bury the cost numbers. $41 for a five-frontier-model 500-problem
  eval is concrete and useful for anyone budgeting their own evals.

## TODO before the post

- Run 2 result + post-fine-tune scoreboard row
- A miss-rate decomposition chart for Qwen3 base + Qwen3 fine-tune
  side-by-side, showing which buckets moved
- A short stratified breakdown by competition / topic if interesting
  patterns emerge
