# Findings — five frontier models on 500 MathNet problems

This document is the public methodology + results writeup for the Day-2
frontier-model evaluation. It complements the [scoreboard in the README](../README.md#scoreboard)
with full caveats, the five primary findings, and the operational lessons
from running the full eval.

## Scoreboard

| Model | N scored | **Accuracy** | Eval cost |
|---|---|---|---|
| Claude Opus 4.7 *(spot-check)* | 100 | **84.0%** | $6.14 |
| Gemini 3 Pro *(partial)* | 240 / 300 | **73.3%** | $13.55 |
| Claude Sonnet 4.6 | 500 | **65.0%** | $10.35 |
| GPT-5.4 | 495 / 500 | **57.8%** | $9.52 |
| GPT-5.4 Mini | 498 / 500 | **36.7%** | $1.51 |
| **Total spend** |  |  | **$41.06** |

Denominator is `n_scored`; missing problems on GPT are OpenAI safety-filter
rejections (documented below). Gemini finished at N=240 due to a preview-model
daily quota cap.

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

## Findings

1. **Opus 4.7 is clearly the strongest**, but the spot-check sample size
   means we treat the 84% as indicative of the ceiling, not as a precise
   number.
2. **Sonnet 4.6 beats GPT-5.4 by 7 pp** (65.0% vs 57.8%) on a sample large
   enough that this is not noise. The Anthropic lineage outperforms the
   OpenAI lineage on MathNet-style olympiad problems in our setup, and at
   comparable eval cost ($10.35 vs $9.52).
3. **GPT-5.4 Mini at 36.7% is the realistic peer for fine-tuned small open
   models.** The bar a 1.5B QLoRA needs to clear isn't Opus; it's Mini.
   This reframes "is the fine-tune good?" from an almost-impossible
   comparison to a meaningful one.
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

The evaluation ran from a single sbatch on the UW Hyak Klone cluster
(`ckpt-all` partition, CPU-only, 4 cpus / 8 GB / 2h54m wall, all API calls).
Pre-launch readiness: preflight gate that hits each provider with one
smoke problem; `tenacity` retry (4 attempts, exponential backoff on 429/5xx);
per-problem JSON writes to disk + SHA-256-keyed inference cache so preemption
or interruption is survivable; per-model log files for independent tailing.

## Links

- Per-model raw + graded JSONs: [results/full/](../results/full/)
- Consolidated NOTES: [results/full/NOTES.md](../results/full/NOTES.md)
- OpenAI filter artifact:
  [results/full/openai-flagged-problems.md](../results/full/openai-flagged-problems.md)
- GPT-5 miss-rate investigation:
  [docs/gpt-missrate-analysis.md](./gpt-missrate-analysis.md)
  (pre-registration: [docs/gpt-missrate-preregistration.md](./gpt-missrate-preregistration.md))
- Day-1 smoke + judge-review report: [day1_report.md](./day1_report.md)
