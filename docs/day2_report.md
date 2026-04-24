# Day 2 report — 2026-04-23

## Goal

Run the five frontier-model evaluation at scale on the 500-problem MathNet
eval split, validate the harness under load, and produce the baselines
that Week-4's fine-tuned Qwen-2.5-1.5B will be compared against.

## What shipped

- **One sbatch run** on Hyak Klone (`ckpt-all`, CPU-only, 4 cpus / 8 GB / 2h54m wall) producing raw responses + graded outputs for:
  - Sonnet 4.6 × 500
  - Opus 4.7 × 100 (spot-check)
  - GPT-5.4 × 500
  - GPT-5.4 Mini × 500
  - Gemini 3 Pro × 300 *(239 completed Day-2 due to a preview-model daily cap; remaining 61 filling in after quota reset)*
- **Pre-launch readiness infrastructure**: preflight gate, tenacity retry (4 attempts, exp backoff on 429/5xx), per-problem JSON writes + inference cache so preemption or interruption is survivable, per-model log files for independent tailing.
- **Cost-tracking fix** after the run: the cost estimator was systematically under-reporting Gemini by ~6× because it wasn't summing `thoughts_tokens`. Fixed and re-run; numbers below match the Google billing dashboard to within sales tax.
- **OpenAI safety-filter artifact** ([results/full/openai-flagged-problems.md](../results/full/openai-flagged-problems.md)) capturing the 7 MathNet problems OpenAI's reasoning API rejected with `invalid_prompt` — 5 on GPT-5.4 (1.0%), 2 on GPT-5.4 Mini (0.4%), 0 elsewhere.

## Headline numbers

| Model | N scored | **Accuracy** | Eval cost |
|---|---|---|---|
| Opus 4.7 *(spot-check)* | 100 | **84.0%** | $6.14 |
| Gemini 3 Pro *(partial, final for Day-2)* | 240 / 300 | **73.3%** | $13.55 |
| Sonnet 4.6 | 500 | **65.0%** | $10.35 |
| GPT-5.4 | 495 / 500 | **57.8%** | $9.52 |
| GPT-5.4 Mini | 498 / 500 | **36.7%** | $1.51 |
| **Total eval spend** | | | **$41.06** |

*Denominator is `n_scored`; missing-from-500 problems on GPT are OpenAI safety-filter rejections (documented). Gemini finished at N=240 due to a preview-model daily quota cap; the remaining 60 are deferred to a future run.*

## Findings (preliminary)

1. **Opus 4.7 is clearly the strongest**, but the spot-check-only sample (N=100) means a ~±8 pp 95% CI. Treat as indicative.
2. **Sonnet 4.6 beats GPT-5.4 by 7pp** (65.0% vs 57.8%) on a sample large enough that this is not noise. The Anthropic lineage outperforms the OpenAI lineage on MathNet-style olympiad problems in our setup.
3. **GPT-5.4 Mini at 36.7%** is the critical baseline for the QLoRA-Qwen comparison: the real head-to-head for a fine-tuned 1.5B model isn't Opus, it's Mini. The bar is lower than I'd expected.
4. **Gemini 3 Pro at 73.3% (N=240 of a 300 target)** with capped thinking. Ran with `thinking_budget=4096` to fit budget; would plausibly score 1–3 pp higher with default (unbounded) thinking. Caveat is prominent in methodology.
5. **GPT-5 family has a large `miss` rate even with the LLM judge enabled** (209 and 315 misses respectively). **Investigated** via 40-sample manual categorization ([gpt-missrate-analysis.md](./gpt-missrate-analysis.md)): 85% of sampled misses are genuine model errors; only 10% are grader artifacts (`extractor_failure` or `judge_false_negative`), below the pre-registered 15% fix threshold. **Conclusion: numbers stand; Mini at 37% is the real Week-4 target, not a grader-inflated figure.**

## Methodology caveats (bullet form — prominent)

- **Gemini 3 Pro ran with `thinking_budget=4096`**, other models ran with default reasoning settings. Cost-control decision based on a 15-problem calibration showing median 5,454 thoughts / max 15,730 per problem under default thinking.
- **OpenAI filtered 5/500 GPT-5.4 and 2/500 GPT-5.4-Mini prompts** with `invalid_prompt` 400s. Accuracy denominators are `n_scored` (495 and 498) rather than 500 for those two models.
- **Opus 4.7 is N=100 by design** as a spot-check, not a full 500. The $60 budget ceiling drove this choice.
- **Gemini 3 Pro is N=300 by design** (rescoped from 500 during calibration to fit budget) and currently N=239 in practice due to a preview-model daily quota cap; fill-in of the remaining 61 is tracked.
- **Judge model = Claude Sonnet 4.6**. The judge's job is pairwise equivalence, not problem-solving, and a 9-problem calibration on Day-1 found 0 false positives. Still, it's the same family as the #3 model on the scoreboard; a second-judge cross-check would be a reasonable follow-up.

The full list of caveats lives in [results/full/NOTES.md](../results/full/NOTES.md).

## What surprised us

1. **Gemini 3.x preview models have a 250 RPD cap even on paid billing.** Not in the public rate-limits docs that I saw. The run hit the cap at exactly 239 successful + 61 failing = 250 daily + some retries.
2. **AI Studio and Google Cloud billing are parallel lanes.** Credits added via AI Studio don't show up in Cloud-console's "$300 free trial credits" view; they appear in a separate "Gemini API Billing" view. Confusing UX; tripped us up for an hour diagnosing.
3. **Our cost estimator was under-reporting Gemini by 6×** because `thoughts_tokens` weren't summed. Found by cross-checking our `summary.json` against the billing dashboard — we thought we'd spent $2.05 on Gemini, the dashboard said $13.49. Fix now in [scripts/grade_results.py](../scripts/grade_results.py).
4. **Opus 4.7 rejects the `temperature` parameter** (reasoning-style model); our preflight script caught this before launch. Worth its weight in gold.
5. **GPT-5.4 didn't use any reasoning tokens on our prompts** — every smoke-test problem showed `reasoning_tokens=0`, so its cost matched "visible output" estimates exactly. Gemini, by contrast, used thinking heavily. Anthropic defaults split the middle: Opus 4.7 runs long without emitting a separate reasoning-token stream.

## Follow-ups before Week 3 / 4

- [ ] Fill in the remaining 60 Gemini problems after its daily quota resets, re-grade Gemini (optional — current N=240 is sufficient for the headline finding)
- [x] ~~Investigate the GPT-5 high `miss` rate~~ — **done** ([gpt-missrate-analysis.md](./gpt-missrate-analysis.md)); 10% grader-artifact rate below threshold; numbers stand
- [ ] Blog-post side observation: pattern-spot the 7 OpenAI-filtered problems (LaTeX? game-theoretic language? specific topics?)
- [ ] **Week 3:** build per-topic-prefix accuracy breakdown from `topics_flat` (stratified analysis directive from project kickoff)
- [ ] **Week 4:** run fine-tuned QLoRA Qwen-2.5-1.5B against same 500-problem eval; compare primarily to GPT-5.4 Mini (the cheapest comparable API tier), secondarily to the frontiers

## Links

- Per-model raw + graded JSONs: [results/full/](../results/full/)
- Consolidated NOTES: [results/full/NOTES.md](../results/full/NOTES.md)
- OpenAI filter artifact: [results/full/openai-flagged-problems.md](../results/full/openai-flagged-problems.md)
- Day-1 smoke + judge-review report: [day1_report.md](./day1_report.md)
- Grader TODOs from Day-1 review: [grader-todos.md](./grader-todos.md)
