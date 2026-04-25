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

## What's evidence of process to keep visible

- The Qwen 2.5 first attempt and the pivot to Qwen3
- The 2% mid-eval that turned out to be a 1024-token measurement
  artifact, not a training failure (and the fix: bump to 4096)
- The pre-registered range that turned out to be too pessimistic, and
  the explicit acknowledgment that pre-registration is a flag, not a
  cage
- The vLLM-doesn't-load-adapters discovery and the merge-step fix that
  unblocked fast post-fine-tune eval

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
