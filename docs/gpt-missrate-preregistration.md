# GPT miss-rate investigation — pre-registration

**Written:** 2026-04-23, before looking at any Day-2 GPT miss responses.
**Investigator:** this is written before labeling to prevent threshold-creep.

## Question

Does the high `miss` rate on GPT-5.4 (42%) and GPT-5.4 Mini (63%) reflect genuine model errors, or does it include a meaningful fraction of cases the grader misclassified? If meaningful, does fixing the grader shift accuracy enough to change the Week-4 target for the fine-tuned Qwen comparison?

## Method

Sample **20 GPT-5.4 misses** and **20 GPT-5.4 Mini misses** from [results/full/gpt-5.4/*.graded.json](../results/full/gpt-5.4/) and [results/full/gpt-5.4-mini/*.graded.json](../results/full/gpt-5.4-mini/) using a fixed seed. Manually label each into one of five categories below.

## Pre-registered categories (5)

Each miss gets exactly one label.

1. **`genuine_wrong`** — model answered incorrectly. The candidate extracted by the grader is a real, final answer the model committed to, and it differs mathematically from the gold. Grader is correct to mark it wrong.

2. **`extractor_failure`** — model *had* the right answer somewhere in the response text, but `extract_answer()` regex grabbed a different string (earlier reasoning step, a side comment, an intermediate result). Fixable by improving the extractor.

3. **`judge_false_negative`** — `extract_answer()` correctly pulled the model's final answer, and that answer IS mathematically equivalent to gold, but the LLM judge said NO. Fixable (or at least surface-able) by judge prompt tuning or a stronger judge.

4. **`answer_not_produced`** — model never arrived at a final answer. Hit max_tokens mid-reasoning, or produced only scaffolding/clarifying text without committing. Not really a grading bug.

5. **`ambiguous`** — doesn't fit cleanly into the above. Examples: model gave a partial answer (correct on some cases, missing on others); model answered a slightly different question than asked; the gold answer itself is under-specified or ambiguous; the problem has multiple valid formulations and the model chose one that isn't isomorphic to gold. Added as a fifth bucket on 2026-04-23 to prevent forcing genuine edge cases into `genuine_wrong`.

## Pre-registered thresholds and decision rules

Let:
- `E = count(extractor_failure)`
- `J = count(judge_false_negative)`
- `A = count(ambiguous)`
- `N = 40` (total sample)

Decisions **locked now**:

1. **If `(E + J) / N ≥ 15%`** → there is a grader bug worth fixing. Fix the extractor and/or re-prompt the judge, re-grade just the GPT-5.4 and GPT-5.4-Mini dirs (cache makes this cheap), and **update all Day-2 artifacts** (NOTES, README, day2_report) with new accuracy numbers.

2. **If `(E + J) / N < 15%`** → Day-2 numbers stand. The high GPT miss rate is not a grader artifact; it's a real model behavior (likely that GPT-5 responds in prose without the `Final answer:` marker). Note this as a methodological finding in the blog's "surprise" section but don't re-run.

3. **If `A ≥ 3`** (`≥ 7.5%` of sample) → `ambiguous` is its own finding worth calling out separately in the analysis doc, independent of the E+J threshold. Note which problem types tend to trigger it.

4. **If `(E + J) / N` is within ±3% of the threshold (12%–18%)** → **do NOT creep the threshold.** Report the raw number, write both possible conclusions, and let the user decide whether to invest in grader fixes. No post-hoc threshold adjustment.

## Sample-selection procedure

Fixed, reproducible:

```python
import json, random
from pathlib import Path
rng = random.Random(20260423)  # today's date as seed
for model in ["gpt-5.4", "gpt-5.4-mini"]:
    misses = []
    for p in sorted(Path(f"results/full/{model}").glob("*.graded.json")):
        r = json.loads(p.read_text())
        if r.get("grade", {}).get("method") == "miss":
            misses.append(r)
    sample = rng.sample(misses, 20)
    # write to docs/gpt-missrate-analysis.md for labeling
```

## Deliverable

[docs/gpt-missrate-analysis.md](./gpt-missrate-analysis.md), to include:

1. Per-category counts and percentages for each of the 40 sampled misses.
2. **2–3 verbatim examples per category** (problem text, gold answer, extracted prediction, relevant slice of model response, judge verdict where applicable). Not just counts — a reader should be able to SEE what an `extractor_failure` looks like without having to open raw JSONs.
3. Conclusion: which decision rule fired, and what action was taken.
4. If action was taken (re-grade), a before/after accuracy comparison.

## Non-goals

- Not investigating Sonnet / Opus / Gemini miss rates — those were ≤35% and within expected range.
- Not trying new judge models or running new API calls — this is a pure pattern-spotting pass on existing data.
- Not retraining or re-running the eval — all 40 examples come from already-cached responses.
