# Day-1 smoke test — notes

**Model:** Claude Sonnet 4.6 (`claude-sonnet-4-6`)
**N:** 20 problems, stratified by competition, sampled from [data/splits/eval.jsonl](../../../data/splits/eval.jsonl)
**Date:** 2026-04-23

## Headline

| Metric | Value |
|---|---|
| Accuracy | **14/20 = 70.0%** |
| Errors | 0 / 20 |
| Wall time | 349 s (~17.4 s/problem) |
| Input tokens | 2,550 |
| Output tokens | 27,913 |
| Estimated eval-side cost | **$0.43** |
| Judge cost (est.) | ~$0.07 |
| **Grader path breakdown** | **exact=4, normalized=2, symbolic=1, judge=7, miss=6** |

Eval-side cost excludes LLM-as-judge API calls (not yet tracked through the summary).

## Manual review of judge wins

All 9 judge-accepted cases (run before the normalizer improvements — now 7 under the updated grader) were eyeballed against gold. See [judge_review.md](./judge_review.md) for the full side-by-side.

**Result: 9/9 genuinely correct, 0 judge errors.** Categorization:

| Category | Cases | Count |
|---|---|---|
| Correct, cheap-layer should have caught (normalizer weakness) | `04sm`, `0db7` | 2 |
| Correct, legit judge work (semantic flex the judge earned) | `00jx`, `01s4`, `0cxc`, `0cyv`, `0cyy`, `0dhv` | 6 |
| Correct, parser could catch (medium effort) | `067y` | 1 |

The 2 "cheap-layer should have caught" cases were fixed post-review by the updated `normalize_for_exact()` (case-fold + LaTeX → unicode + strip `<var> =` prefix). After the fix, those two cases resolve at the `normalized` layer instead of `judge`.

**Conclusion: the 70% Sonnet 4.6 baseline is trustworthy for Day 2 comparison.**

## The 6 genuine misses

| ID | Predicted | Gold | Diagnosis |
|---|---|---|---|
| `01x2` | 11 | 11.2 | arithmetic error |
| `05cj` | *(empty)* | conditional on n | **truncated** — hit `max_tokens=4096`. Bumped default to 8192. |
| `09p0` | 38264 | 157152 | wrong counting |
| `0dce` | `P(x) = x^k` | conditional on n parity | partial (missed odd-n branch) |
| `0df4` | `{(1,1,-1), (1,-1,1), (-1,1,1)}` | `(1/2, 1/3, 1/6)` permutations | wrong answer |
| `0hbp` | Yes | No | flipped |

1/6 of the misses is our fault (truncation), the other 5 are genuine model errors. With 8192 tokens the truncation category goes away.

## Decisions made

1. **English-only, text-only scope confirmed for Week 1.** The funnel: 27,817 → 22,669 (text-only) → 6,708 (+English) → 4,096 (+has final_answer). This gave us a 500-problem eval + 3,596-problem train. The train ceiling is below the 5K target but sufficient for a first SFT pass. Revisit multilingual / proof-only scope in Week 3 if training signal is weak.
2. **Hybrid grader with LLM judge is required, not optional.** The string/sympy layers together score ≤ 50% of correct answers; the judge adds the other 50%. On the full 500 × 4-model run this matters.
3. **Commit results from laptop, not Hyak.** Hyak doesn't have GitHub credentials; pushing is an `rsync` + local commit away. Long-term fix is a GitHub PAT or SSH remote on Hyak; not blocking.

## Follow-ups before the full 500-problem run (Day 2)

- [x] Normalizer cheap-layer fixes (case-fold, LaTeX → unicode, strip `<var> =` prefix)
- [x] `max_tokens` default bumped 4096 → 8192
- [ ] OpenAI (GPT-5) backend
- [ ] Google (Gemini 3 Pro) backend
- [ ] Budget estimate for 500 × 4 models before the green-light
