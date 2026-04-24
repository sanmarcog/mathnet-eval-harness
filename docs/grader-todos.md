# Grader improvements noticed during Day 1 judge review

- [ ] **Normalize should case-fold.** `All integers k ≥ 2` vs `all integers k ≥ 2` falls through to judge (expensive) when a `.lower()` in normalize() would catch it.
- [ ] **Normalize should map LaTeX macros to unicode.** `\geq`/`\leq`/`\neq`/`\pm`/`\cdot`/`\times` → `≥`/`≤`/`≠`/`±`/`·`/`×`. Similar for trig/log macros if they appear in `final_answer` values.
- [ ] **Normalize should strip trailing punctuation** (`.`, `,`) — olympiad answers sometimes end with a sentence period.
- [ ] **After these: re-grade results/smoke/sonnet-4-6/** and see how many of the 9 judge-hits drop to `normalized`. Expected: 2-3. That saves ~$0.02/run × many future runs.

Noticed while the user eyeballed problem `04sm` (`All integers $k \geq 2$` vs `all integers k ≥ 2`).

- [ ] **Strip `<varname> = ` prefix from predicted answers.** `$A = -1$` vs `-1` — common pattern when the problem asks for the value of a labeled variable and the model dutifully re-labels it in the final answer. Regex like `^\s*[A-Za-z]\w*\s*=\s*` applied to the *predicted* side only.
  - Noticed on problem `0db7` during Day 1 judge review.

## Harder (parser-level) improvements

- [ ] **Tuple-set parser.** Gold `[(4,8,0), (35,70,1), (14,77,1)]` vs pred `{(4,8,0), (14,77,1), (35,70,1)}` — same set of triples, different outer container + order. A parser that extracts bracketed tuples and compares as a multiset would catch this without the judge. Noticed on `067y`.
  - Scope: medium effort (handle `(`, `[`, `{`, `\{`, ordered vs unordered, nesting).
  - Sequencing: probably not worth building before we have data on *how often* this pattern shows up across the full 500-problem eval.

## Day-2 miss-rate analysis findings (2026-04-23)

- [ ] **Fix the `\[` extractor-failure bug.** `extract_answer()` grabs the opening LaTeX display delimiter `\[` as the predicted answer when the response uses a standalone `\[ … \]` block immediately after "Final answer:". Noticed on GPT-5.4 problems 0hd3 and 04p3 during the miss-rate investigation. Two of 20 sampled GPT-5.4 misses (10%) had this pattern. Below the 15% re-grade threshold so deferred, but a ~10-line regex fix would recover both problems and any similar cases in the other model runs.
- [ ] **Judge has two blind spots worth prompt-tuning:** (a) numerical equivalence that requires a one-step arithmetic ("is 2/65173330 the same as 1/32586665?"), and (b) geometric locus equivalences expressed with different-but-same-value coordinate formulas. Consider giving the judge a sympy-assisted "try to simplify both sides" step or a self-check loop. Below the fix threshold, but worth noting.

## Day-1 judge review findings (2026-04-23)
