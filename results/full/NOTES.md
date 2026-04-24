# Day-2 full eval — notes

**Date:** 2026-04-23
**Sbatch job:** 34793163 on `ckpt-all` / `n3263` (CPU-only, 4 cpus, 8 GB, ~2h54m wall)
**Models + scope:** Sonnet 4.6 / GPT-5.4 / GPT-5.4 Mini × 500; Gemini 3 Pro × 300 *(partial 239, fill-in underway)*; Opus 4.7 × 100 spot-check
**Stratification:** by `competition` on `data/splits/eval.jsonl` (seed 0, 282 strata)

## Scoreboard

| Model | N scored | **Accuracy** | exact | normalized | symbolic | judge | miss | Eval cost |
|---|---|---|---|---|---|---|---|---|
| **Opus 4.7** | 100 | **84.0%** | 20 | 5 | 3 | 56 | 16 | $6.14 |
| **Gemini 3 Pro** *(partial, final for Day-2)* | 240 | **73.3%** | 75 | 1 | 7 | 93 | 64 | $13.55 |
| **Sonnet 4.6** | 500 | **65.0%** | 87 | 30 | 16 | 192 | 175 | $10.35 |
| **GPT-5.4** | 495 | **57.8%** | 53 | 24 | 6 | 203 | 209 | $9.52 |
| **GPT-5.4 Mini** | 498 | **36.7%** | 20 | 23 | 4 | 136 | 315 | $1.51 |

Total eval spend: **$41.06**. Judge calls (Sonnet 4.6) are shared across models; rough estimate ~$5 not itemised here. Full run final: **~$46** under the $60 ceiling.

## Methodology caveats — read before citing any number

1. **Sample sizes differ by design and by incident.** The accuracy denominator is `n_scored`, not the nominal target (500 or 300 or 100).
    - Sonnet 4.6, GPT-5.4, GPT-5.4 Mini nominally targeted N=500. Actual `n_scored`: 500, 495, 498. Missing problems are OpenAI safety-filter rejections (see below).
    - Gemini 3 Pro targeted N=300 (rescoped down from 500 during calibration to fit the $60 ceiling — see §"Gemini rescoping"). Actual final `n_scored` for Day-2: **240**. The preview model enforces a 250 requests-per-day cap (rolling 24 hours) which we hit on the sbatch run (239) and an attempted fill-in window (1 more). Remaining **60 problems are deferred** until we have a reason to run Gemini again (Week 3 topic-stratified analysis is a natural re-run trigger). The 1-problem difference between 239 and 240 does not change any finding.
    - Opus 4.7 was always a **100-problem spot-check** — we deliberately did not run the full 500 because Opus 4.7 is ~5× the price of Sonnet 4.6 and the cross-model headline didn't need it at full N. Wider confidence interval on Opus is expected; treat it as indicative, not definitive.

2. **OpenAI safety-filter rejections adjust the GPT denominators** (*"filter rate"* in the Day-1-reviewed sense):
    - GPT-5.4: **5 of 500** prompts rejected (1.0%) with `invalid_prompt` 400s. Accuracy reported as 286 / 495.
    - GPT-5.4 Mini: **2 of 500** (0.4%). Accuracy reported as 183 / 498.
    - Other backends (Sonnet, Opus, Gemini) accepted every prompt.
    - These are not model errors; they are prompt-level moderation rejections by the OpenAI reasoning API. The prompts themselves are plain olympiad math from MathNet — see [openai-flagged-problems.md](./openai-flagged-problems.md) for the full list of 7 flagged problems with gold answers and error messages. Worth a one-paragraph side observation in the writeup about what trips the filter (LaTeX-heavy problems, partition/allocation language); sample is small so pattern-spotting is loose.

3. **Gemini 3 Pro ran with `thinking_budget=4096`; the other models ran with their default reasoning settings.** This asymmetry is the result of a cost-control decision, not a design preference:
    - A 15-problem Gemini calibration showed median 5,454 thoughts tokens per problem (max 15,730) under default thinking, which extrapolated to ~$38 on 500 problems and blew the $60 ceiling.
    - Capping `thinking_budget=4096` bounded wall time from an observed 567s tail down to 63s and brought the projected cost into range.
    - Sonnet 4.6, Opus 4.7, GPT-5.4 ran in their default (non-opt-in extended thinking) modes. GPT-5.4 reported 0 reasoning tokens on every problem in the smoke test (we did not opt into reasoning mode), so its "visible output only" behavior mirrors Sonnet. Opus 4.7 is a Claude reasoning model that uses extended thinking by default and reported higher output-token counts per problem (~2,400 avg vs ~1,250 for Sonnet).
    - **Bottom line: the Gemini accuracy number is probably a slight underestimate of what it would do with unbounded thinking, particularly on hardest problems.** Directionally worth noting if Gemini is close to another model in the final comparison.

4. **Gemini rescoping from 500 → 300.** Driven by the calibration above; documented in [docs/day2_report.md](../../docs/day2_report.md). The competition-strata representation in the Gemini sample is still proportional to the 4,096-problem filtered pool, just with smaller per-stratum counts than the other four models.

5. **Opus 4.7 temperature.** The Anthropic API rejects custom `temperature` for Opus 4.7 (reasoning-style behavior). Opus runs with the provider default; Sonnet and Gemini ran with `temperature=0.0` explicitly; GPT-5.x also omits temperature (reasoning-model restriction). Non-uniform but consistent with each provider's defaults.

6. **Judge model = Sonnet 4.6** (our cheapest strong frontier model). The judge sees the *problem*, the *gold answer*, and the *candidate*. It cannot grade itself — but since Sonnet shows at 65% here (middle of the pack) and the judge's job is pairwise equivalence rather than problem-solving, self-judging risk is low. Day-1 review of 9 judge-accepted answers validated 9/9 correct. See [../smoke/sonnet-4-6/judge_review.md](../smoke/sonnet-4-6/judge_review.md).

## Headline findings (preliminary, pending Gemini fill-in)

- **Opus 4.7 leads at 84%**, clean separation. Small N; wide CI — treat as indicative.
- **Gemini 3 Pro second at 73% (partial)**. Even with capped thinking. Expect to stay second once the 61 remaining problems fill in.
- **Sonnet 4.6 (65.0%) beats GPT-5.4 (57.8%)** by ~7pp — not within noise. That's genuinely interesting for the blog.
- **GPT-5.4 Mini at 36.7%** is the critical datapoint for the QLoRA-Qwen comparison coming in Week 4. That's our real "small-model baseline" target, not Opus or any other flagship.
- **GPT-5 family shows a strikingly high `miss` rate** (209 and 315 respectively) even after the LLM judge layer. Worth investigating whether this is a formatting issue (GPT produces answers the extractor can't parse), a grader-judge gap (judge is conservative on GPT outputs), or genuine errors. Hypothesis to check: GPT's tendency to wrap answers in prose rather than the `Final answer:` marker.

## Operational notes / what surprised us

- **Gemini 3.x preview models have a 250 RPD cap that applies even on paid billing tier.** This is not documented in the rate-limits page we read; we discovered it experientially when the sbatch ran into 429s after 239 successful calls. The cap is per-model-per-day; upgrading the tier doesn't raise it. Worth flagging for anyone else doing frontier-scale evals on Google preview models.
- **Google AI Studio billing ≠ Google Cloud billing.** Credits added via AI Studio show up on the "Gemini API Billing" view (visible at the preview/upgrade flow). `0 out of $300 Cloud trial credits used` on the main Cloud console can coexist with paid tier active on the specific project. Confusing UX; cost paid $14.92 shows up on the Gemini billing view, not the Cloud credits view.
- **Our cost estimator systematically under-reported Gemini by ~6×** before a mid-Day-2 fix because `summary.json` only summed input + visible output tokens. `thoughts_tokens` are billed at output rate. [grade_results.py](../../scripts/grade_results.py) now sums thinking/reasoning/thoughts across all responses before computing cost. Correct total matches the Google billing dashboard to within the tax line item.
- **Opus 4.7 rejects the `temperature` parameter**; other reasoning-style models (GPT-5.x) do too. [scripts/preflight.py](../../scripts/preflight.py) caught this before the main run — exactly what it exists for.

## Reproducing

```bash
# On Hyak
cd /gscratch/scrubbed/sanmarco/mathnet-eval-harness
git checkout <this-commit>
sbatch slurm/full_eval.sbatch     # reproduces the 5-model run
# After completion:
for m in sonnet-4-6 opus-4-7 gpt-5.4 gpt-5.4-mini gemini-3-pro; do
    python scripts/grade_results.py --dir results/full/$m --use-judge
done
python scripts/collect_openai_flags.py    # rebuilds the flagged-problems artifact
```

Splits are deterministic (seed=0 in `build_splits.py`). Inference has a SHA-256 disk cache keyed on `(model, prompt, params)` so re-runs are free. Grader is deterministic given fixed judge responses (and judge responses cache the same way).

## Links

- Per-model consolidated summaries: `sonnet-4-6/summary.json` etc.
- Per-problem raw + graded: `<model>/{id}.json`, `<model>/{id}.graded.json`
- OpenAI filter artifact: [openai-flagged-problems.md](./openai-flagged-problems.md)
- Day-1 operational validation: [../smoke/sonnet-4-6/NOTES.md](../smoke/sonnet-4-6/NOTES.md)
- Day-2 report: [../../docs/day2_report.md](../../docs/day2_report.md)
