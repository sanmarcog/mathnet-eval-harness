# TIR + RAG-of-TIR on Qwen2.5-Math-1.5B-Instruct — pre-registration

**Status: LOCKED 2026-04-29.** All citations verified per the verify-citations rule.

**Pre-reg corrections, 2026-04-30** (consistency fixes — not result-driven amendments; flagged before any production data lands):
- *Sandbox implementation.* Originally specified `smolagents.LocalPythonExecutor`. Switched to a **subprocess sandbox** (`subprocess.run([python, "-c", code], timeout=10)` with a 2000-char stdout cap). Reason: the pre-reg's load-bearing line under §Three eval conditions was already "subprocess-isolated, not E2B"; smolagents' executor is in-process, not subprocess-isolated, so the original wording was internally inconsistent. The subprocess path is simpler and matches the isolation guarantee we said we'd provide. Code: `src/mathnet_eval/tir.py::PythonSandbox`.
- *Smoke `print(2)` canary.* `scripts/eval_tir.py` injects a canned `print(2)` tool call on the first generation **only when `--smoke`** (the smoke uses Qwen2.5-0.5B-Instruct on CPU, which is too small to reliably emit the tool-call convention from prompt alone). The canary forces the sandbox + tool-output-feedback path to fire on every smoke problem so the integration is exercised. Production runs (Qwen2.5-Math-1.5B-Instruct, vLLM) never see this code path. Gated strictly behind `args.smoke and args.mode in ("tir","tir_rag")`.
- *Bank-builder smoke gating.* The bank-build sbatches require their respective `--smoke-real` to have been run on the current commit. Sentinels: `tests/.smoke_real_tir_passed_<commit>` (gates `slurm/build_tir_exemplar_bank.sbatch`) and `tests/.smoke_real_cot_passed_<commit>` (gates `slurm/build_cot_exemplar_bank.sbatch`). The eval sbatch (`slurm/eval_tir.sbatch`) requires only the 5-sandbox-sentinel pass + bank existence; it does NOT require either `--smoke-real` to have run, because the eval doesn't generate banks.
- *CoT exemplar bank source.* The 18-cell ablation's CoT-exemplar arm reads from a **dedicated CoT-rollout bank** (`results/tir/exemplar_bank_cot.jsonl`), built by `scripts/build_cot_exemplar_bank.py` running `Qwen2.5-Math-1.5B-Instruct` in CoT mode and filtered for correct-answer rollouts. It is NOT TIR rollouts re-formatted to drop the code blocks. Reason: the headline secondary question is "do tool-using rollouts as exemplars beat CoT-only rollouts as exemplars at 1.5B?" — a question about the *content* of retrieved exemplars, not the formatting. A `--bank-cot results/tir/exemplar_bank.jsonl` fallback path exists in the ablation runner for dev convenience (so smoke + early integration tests can run without two production banks); the headline run uses the dedicated CoT bank.

**Pre-reg correction, 2026-05-01** (caught by in-flight diagnostic on the first cot sbatch — 4.5% accuracy on n=220 with 39% trailing-loop pattern, far below Alibaba's 38.1% on OlympiadBench, gap too large to be distribution shift):
- *System prompt swap to Alibaba's canonical wording.* Replaced the original wordier prompts with the verbatim wording from the Qwen2.5-Math tech report (2409.12122):
  - **CoT**: `"Please reason step by step, and put your final answer within \boxed{}."`
  - **TIR**: `"Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \boxed{}."`
  Reason: Qwen2.5-Math-Instruct is format-sensitive — it was trained on a specific short prompt and deviating elicited greedy-decoding repetition loops on ~39% of CoT rollouts, not a capability deficit. A 10-problem login-node smoke on `Qwen2.5-Math-1.5B-Instruct` with the canonical prompts produced loop rate **0/6** (vs 39% under the prior prompt) and boxed-emission rate **5/6 = 83%** (vs 28%), confirming the diagnosis. Code: `src/mathnet_eval/tir_prompts.py` (commit `0bd0661`). All three eval conditions (cot, tir, tir_rag) now use the canonical prompts; the original 4.5% cot run (job 34979430) is **discarded** as an instrument-malfunction measurement, not data.

**Pre-reg correction, 2026-05-02** (yield-driven bank model swap; reframes the experiment):
- *TIR / CoT exemplar banks generated with `Qwen2.5-Math-7B-Instruct`, headline eval still on `Qwen2.5-Math-1.5B-Instruct`.* The 1.5B bank-tir run completed at **26 / 3496 = 0.74% yield**, well below the pre-reg's 200-row target. Per the documented fallback ladder (§TIR exemplar bank construction), 7B is the next step. Resume-safety in the bank builders preserves the 1.5B-kept rows as a base; 7B appends.
  - **What this changes about the experiment.** The pre-reg's secondary headline question — *"do tool-using rollouts as exemplars beat CoT-only rollouts as exemplars at 1.5B?"* — was implicitly framed as **self-help via retrieval** (model retrieves its own prior successful rollouts). With banks built on 7B and inference on 1.5B, the question becomes **in-context distillation via retrieval**: does a 1.5B model do better on hard math when it can retrieve worked examples from a stronger model? That's a different mechanism (knowledge transfer, not self-elicitation) and a different story arc.
  - **Writeup commitment.** Both framings will be reported. The interpretation-matrix prose (§3×3 matrix below) still reads correctly under either framing — outcome cells are about whether retrieval helps, not about whether the helper is the same model. But the headline framing shifts: any positive RAG-of-TIR lift now reads as "distillation from 7B works in-context for 1.5B," not "the 1.5B benefits from its own prior successes."
  - *Multilingual fallback flag.* If 7B-on-English still falls below 200 rows, the next step is multilingual train (11,648 rows, pre-reg fallback step 1). **In that case, exemplars from non-English problems will mix into the bank while the headline eval is English-only.** Mitigation: filter the multilingual bank to keep only English-source rollouts (model's response language is what matters for retrieval similarity), OR cite the language-mismatch risk in the writeup with a per-language breakdown of retrieved exemplars in the audit. Decision deferred to when the multilingual step actually triggers.

Pre-registered before launching the eval. Locks the hypothesis, recipe, and interpretation language for each (TIR transfer × RAG-of-TIR lift) outcome cell. Treats pre-registration the way Run 4 and Dr. GRPO did: language committed before measurement, so the post-result writeup is honest about what each magnitude means.

## Hypothesis

**Headline.** Alibaba report a +11.1 pp lift from CoT → TIR on `Qwen2.5-Math-1.5B-Instruct` on OlympiadBench (38.1 → 49.2), per Table 3 of [2409.12122](https://arxiv.org/abs/2409.12122). MathNet is a different distribution (broader country/competition mix; longer published solutions in expectation). **Does Alibaba's TIR lift transfer to MathNet's distribution at the same model size?**

**Secondary.** At 1.5B scale, **does retrieving full TIR exemplars (problem + working Python + executed result + final answer) add measurable lift over plain TIR?** [Bangla MO 2501.04425](https://arxiv.org/abs/2501.04425) reports a related technique at Qwen2.5-32B: 70 → 71 / 100 with TIR + retrieved similar-problem-with-TIR-solution, **+1pp on N=100 — within noise.** Their retrieval is keyword-based and does not strictly retrieve (problem + working code + executed output + answer) tuples as we do. We are not testing replication; we are establishing the 1.5B effect size where no published number exists.

This pre-reg locks interpretation prose for nine outcome cells (3 × 3) so the writeup framing is decided before the data lands.

## Eval set: existing Week-1 500 (`data/splits/eval.jsonl`)

The MathNet paper ([2604.18584](https://arxiv.org/abs/2604.18584), Alshammari et al.) defines `MathNet-Solve-test-hard` (500 problems) but the IDs were never released. The lead author confirmed (personal correspondence, 2026-04-29) that the split was postponed and accidentally appeared in the arxiv. Beyond that, she made an observation that decides the methodology question for us:

> "When we looked at problems that most frontier LLMs failed on, they weren't necessarily the hardest for humans or from the toughest competitions. A lot of them instead had a strong visual component (e.g., maze or Sudoku-style problems), which makes it hard to use model performance alone as a clean signal."

I.e., **frontier-LLM-failure ≠ olympiad-hard.** If the dataset author cannot cleanly define a "hard" subset of MathNet, we should not hand-construct one ourselves either — any rule we picked (sol_len, competition-tier, mean-frontier-LLM-accuracy) would be a proxy for one of these dissociated axes, not a defensible operationalization of "olympiad-hard." Liu et al. ([2503.18069](https://arxiv.org/abs/2503.18069), "Long Is More Important Than Difficult for Training Reasoning Models") makes the related point that *length* and *difficulty* are dissociable; that's a piece of the same problem.

So the eval set is the **Week-1 stratified random sample (n=500)**, already committed to `data/splits/eval.jsonl`. Why this is the right call regardless of the hard-set question:

1. **The headline question is distribution-shift, not hardness-shift.** "Does Alibaba's +11.1pp TIR lift on OlympiadBench transfer to MathNet?" is well-formed on any random sample of MathNet. MathNet is a different distribution than OlympiadBench (different countries, different competitions, different topics) regardless of which 500 problems we pick.
2. **Floor and ceiling anchors are already paid for.** The existing 500 has frontier grades for Sonnet 4.6 (65.0%), GPT-5.4 (57.8%), GPT-5.4 Mini (36.7%), and open-base grades for Qwen3-1.7B base (36.8%). The B3 floor (Qwen3-1.7B base) and a frontier-ceiling reference (Sonnet 4.6) are *paired-McNemar-comparable* against our three TIR conditions on the same IDs.
3. **Saves $15 in fresh frontier-anchor spend** that would otherwise be needed for any newly derived 500.
4. **Avoids stacking a confounded "harder" claim on top of the transfer claim.** Cleaner experimental design.

**Universe characterization** (verified in `scripts/inspect_metadata_confidence.py`): the existing 500 was stratified-randomly drawn from a universe of 4,096 text-only English problems with non-empty `final_answer` from `ShadenA/MathNet`'s 27.8K total. Mean frontier accuracy across our three frontier models on these 500 is 0.531; spans 0 to 1 with bimodal mass at the extremes. Topic distribution roughly matches the universe (Algebra ~40%, Number Theory ~26%, Discrete ~24%, Geometry ~9%).

## Three eval conditions (run on the same 500 IDs from `data/splits/eval.jsonl`)

| Condition | Tool access | Retrieval | Token budget |
|---|---|---|---|
| `cot` | none | none | 4096 |
| `tir` | Python sandbox, ≤4 calls | none | 4096 (excl. tool I/O) |
| `tir_rag` | Python sandbox, ≤4 calls | k=3 from TIR exemplar bank, locked policy from §Retrieval ablation | 4096 (excl. tool I/O) |

**Base model:** `Qwen/Qwen2.5-Math-1.5B-Instruct` — the math-specialist Instruct tune that matches the model Alibaba publish their TIR numbers on, so the TIR-transfer question is well-formed.

**Backend:** vLLM 0.6+, `temperature=0` (greedy), `seed=0`, `max_new_tokens=4096` (matches Alibaba's published 4K eval budget on this model).

**Chat template:** model default. We do not add `<think>` blocks (the math-specialist Instruct is not a long-thinking model).

**Tool sandbox:** subprocess sandbox (per the 2026-04-30 correction above) — `subprocess.run([sys.executable, "-c", code], timeout=60, capture_output=True)`. Per-call timeout 60s (originally 10s; bumped 2026-04-30 after the Klone smoke gate caught that NFS-cold-start `import sympy` alone is ~10s, so every call timed out before its first print). Tool output truncated to 2000 chars before being fed back. Subprocess-isolated, not E2B. No OS-level network/filesystem sandboxing — relies on the model not emitting destructive code plus the 60s timeout. Allowed-import discipline (`sympy, numpy, math, fractions, itertools, functools, re`) is convention-only, not enforced; for stronger guarantees swap to E2B.

## Reference anchors on the same 500

Already in hand from Week 1, no fresh runs required:

| Anchor | What | Source |
|---|---|---|
| **Small-model floor (open)** | Qwen3-1.7B base, vLLM thinking-on, 16K | `results/full/qwen3-1.7b-base/` (n=500) |
| **Cheap-tier reference (closed)** | GPT-5.4 Mini | `results/full/gpt-5.4-mini/` (n=498) |
| **Frontier reference (closed, no RAG)** | Sonnet 4.6 | `results/full/sonnet-4-6/` (n=500) |

These give *paired* McNemar-comparable references on the same 500 IDs. Specifically: TIR-vs-base McNemar tests whether tools close any of the open-floor-to-frontier gap.

**Optional (budget-permitting): a frontier+RAG ceiling.** A fresh pass with Sonnet 4.6 + simple keyword-RAG (k=3 from MathNet train universe) on the same 500 would cost ~$10 generation + ~$4 judge ≈ $14. It changes the question from "did our small-model+tools+RAG close the gap to frontier-without-RAG?" to "...to frontier-with-RAG?" The fresh ceiling is the more honest framing for the C2 secondary RAG-of-TIR question. Decision deferred to launch time depending on budget headroom.

**Why we are not citing the MathNet paper's headline numbers as anchors:**
- Ministral-3B = 4.4% is real but on **MathNet-Solve-Test (6,400 problems)**, not on a paired 500-problem subset.
- DeepSeek-V3.2-Speciale + Expert-RAG = 97.3% is real but on **MathNet-RAG (35 problems)**, a different task entirely.
- No published numbers exist on any 500-problem MathNet subset for any model. The Week-1 500 is *our* paired-comparison baseline.

## Reference points (verified, used as context, not as anchors)

| Reference | Value | Source | Status |
|---|---|---|---|
| Alibaba TIR lift on Qwen2.5-Math-1.5B-Instruct, OlympiadBench | CoT 38.1 → TIR 49.2 (+11.1 pp) | [2409.12122](https://arxiv.org/abs/2409.12122) Table 3 | ✓ verified 2026-04-29 |
| Alibaba TIR on AIME24 | CoT 3/30 → TIR 7/30 (+4 problems) | same table | ✓ verified |
| Alibaba TIR on MATH | CoT 75.8 → TIR 79.9 (+4.1 pp) | same table | ✓ verified |
| Bangla MO @ 32B: TIR + retrieved-similar-problem-with-TIR-solution | 70 → 71 / 100 (+1 pp on N=100, within noise) | [2501.04425](https://arxiv.org/abs/2501.04425) | ✓ verified — their RAG retrieves similar problems' TIR solutions via keyword search; **not** strict (problem + code + executed output + answer) tuple retrieval as we do |

Bangla's +1pp at 32B is the **noise reference** for the secondary RAG-of-TIR question, not a precedent that "worked at 32B."

## Pre-flight gate (`--smoke`)

Mirrors the Dr. GRPO pattern (commit `1ead8cb`):

- `python scripts/eval_tir.py --mode {cot,tir,tir_rag} --smoke` → CPU-only on N=4 problems, ≤2 min wall-clock, catches sandbox / retrieval / JSON-write bugs without sbatch.
- `python scripts/build_tir_exemplar_bank.py --smoke` → bank generation on N=8 train rows; eyeball one or two filtered exemplars before launching full bank generation.
- **Sandbox sentinel set.** Five hand-graded problems (3 sympy, 2 numpy) committed to `tests/tir_sandbox_sentinels.json`. Smoke test runs each through the sandbox and asserts the executed result equals the hand-computed answer. Catches future smolagents / sympy version bumps that silently break the sandbox.

Full sbatch refuses to launch unless `logs/tir_smoke_passed_${commit}.txt` exists for the current git commit.

## TIR exemplar bank construction

- **Source:** `data/splits/train_english.jsonl` (3,596 problems, Week-1 English train set — disjoint from the eval 500 by Week-1 stratified split).
- **Generation:** `Qwen2.5-Math-1.5B-Instruct` runs in TIR mode on each train problem (vLLM, greedy, 4K cap, ≤4 tool calls).
- **Filter:** keep rows where (a) `extract_answer(rollout)` matches gold under cheap-grader (exact + normalized + sympy), AND (b) the model called Python at least once. Rows correct-but-no-Python are dropped.
- **Expected yield:** unknown until bank smoke runs. Pre-launch, commit a per-row breakdown of (model emitted Python? executed cleanly? answer matched gold?) on N=20 rows to set a yield prior.
- **Yield fallbacks (in order):**
  1. yield < 200 rows → expand source to multilingual train (11,648 rows; Week-1 Run 2/3 train set);
  2. still < 200 → regenerate with `Qwen2.5-Math-7B-Instruct` and document the swap.
- **Bank is committed** to `results/tir/exemplar_bank.jsonl` for reproducibility.

## Retrieval ablation (sub-grid → headline policy)

A separate ablation sub-run on **dev N=100** decides which policy advances to the headline `tir_rag` eval.

- **Dev N=100** is sampled from `train_english.jsonl` (so it is held out from the eval 500), stratified by `top_topic` (Algebra / NT / Discrete / Geometry) proportionally. Dev IDs are **excluded from the TIR exemplar bank** during bank construction so retrieval cannot trivially return the same row. Dev set is committed alongside the launch sbatch.

| Axis | Levels |
|---|---|
| Index | BM25 (`rank_bm25`) · dense (`BAAI/bge-small-en-v1.5`) · `topics_flat` exact-match |
| k | 1 · 3 · 5 |
| Exemplar type | TIR exemplars · CoT exemplars (control: isolates the *trace contribution* from the *retrieval contribution*) |

3 × 3 × 2 = **18 ablation cells**. Single best (index, k, exemplar-type) by mean accuracy on dev N=100 advances to the n=500 main eval. **The full 18-cell table is reported in the writeup**; headline `tir_rag` numbers come from the n=500 main eval only.

**Tie-break rule (locked before measurement):** if two cells are within 1 pp on dev N=100, prefer (1) smaller `k` (parsimony — fewer in-context exemplars at 1.5B), then (2) BM25 over dense (cheaper to ship, no embedding model dependency at inference time), then (3) TIR-exemplar over CoT-exemplar (matches the headline framing). Lock keeps post-hoc cell selection from drifting.

Separating the dev set avoids selected-on-dev contamination of the headline n=500 number.

## Pre-registered interpretation matrix (3 × 3)

Locked before measurement. The result will land in exactly one cell; the prose for that cell goes into the writeup verbatim.

**Outcome axes (paired McNemar on the n=500 eval set):**
- **TIR transfer Δ** = `acc(tir) − acc(cot)`
- **RAG-of-TIR lift Δ** = `acc(tir_rag) − acc(tir)`

**Bucket boundaries** (chosen so a paired delta on n=500 in the 5-30% accuracy regime sits past the ~±4 pp paired-CI noise floor):

| TIR transfer Δ | Bucket |
|---|---|
| ≥ +5 pp | **T1: Strong transfer** |
| 0 to +5 pp | **T2: Modest transfer** |
| < 0 pp | **T3: No / negative transfer** |

| RAG-of-TIR lift Δ | Bucket |
|---|---|
| ≥ +3 pp | **R1: Clear lift** |
| 0 to +3 pp | **R2: Marginal lift** |
| < 0 pp | **R3: Null / negative** |

### The 3 × 3 outcome matrix

|              | **R1: RAG clear lift (≥ +3 pp)** | **R2: RAG marginal (0 to +3 pp)** | **R3: RAG null / negative (< 0 pp)** |
|---|---|---|---|
| **T1: TIR strong (≥ +5 pp)** | **(T1, R1) — Both work.** Alibaba's +11pp TIR transfers to MathNet's distribution AND retrieval-of-tool-traces compounds the lift at 1.5B. Headline: "small-model TIR generalizes from MATH/Olympiad to MathNet, and retrieving working tool traces produces measurable additional lift at 1-3B scale — beyond the within-noise effect Bangla MO measured at 32B." | **(T1, R2) — TIR carries it, RAG marginal.** TIR transfers; retrieval adds within Bangla's 32B noise band. Headline: "TIR is the load-bearing piece on this distribution; retrieval-of-tool-traces is consistent with noise at 1.5B as it was at 32B." | **(T1, R3) — TIR transfers, RAG hurts.** TIR generalizes; retrieving exemplars actively hurts at 1.5B. Headline: "TIR transfers cleanly to MathNet's distribution; retrieval injects exemplars that distract a small-capacity model. Consistent with a working-capacity ceiling at 1-3B for in-context exemplars on hard math." |
| **T2: TIR modest (0 to +5 pp)** | **(T2, R1) — RAG carries it.** TIR transfers weakly; retrieval adds clearly. Headline: "tools alone partially transfer Alibaba's OlympiadBench result to MathNet, but retrieval-of-tool-traces produces clear additional lift on top of plain TIR at 1.5B." | **(T2, R2) — Honest small-lift report.** Both pieces help marginally; total lift is real but small. Per `target-honesty`: do not dress this up as a strong finding. Headline: "small but measurable contribution from both TIR and RAG-of-TIR on MathNet's distribution at 1.5B; bulk of the floor-to-ceiling gap remains, consistent with capacity ceiling at this scale." | **(T2, R3) — TIR is the only contribution.** TIR adds modestly; retrieval doesn't transfer to 1.5B. Headline: "tool use is the only technique that transfers cleanly down to 1.5B on MathNet; retrieval-of-tool-traces does not, consistent with a working-capacity ceiling at this scale." |
| **T3: TIR no / neg transfer (< 0 pp)** | **(T3, R1) — Surprising: RAG without TIR transfer.** Retrieval helps even though tool-use itself doesn't transfer. Argues the retrieved *content* (worked-out Python + executed answer) is doing the lift via in-context inference, *not* via tool execution. **Pre-register a follow-up audit on N=20 problems where TIR misses but TIR+RAG hits**: is the model copying the retrieved final answer when the problem is structurally similar? If yes, report two numbers in the writeup: with and without answer-blacklist on retrieval. | **(T3, R2) — RAG marginal, TIR null.** Both pieces produce small/no lift on MathNet at 1.5B. Headline: "neither tool-use nor retrieval-of-tool-traces transfers cleanly from published distributions to MathNet at 1.5B." | **(T3, R3) — Total null.** Headline: "Alibaba's published +11.1pp TIR lift on Qwen2.5-Math-1.5B-Instruct does not transfer to MathNet's distribution; retrieval-of-tool-traces does not change the picture. The 1-3B-class accuracy floor on this distribution sits where the small-model anchor (Qwen3-1.7B base 36.8%) already sits — neither technique perturbs it. Counter-evidence to two papers; legitimate negative result if it lands here." |

### Audit triggers (regardless of outcome cell)

- **Retrieval-leakage check.** If `tir_rag` shows ≥ +3 pp lift AND ≥ 50% of the lift comes from problems where the top-k retrieved exemplar's `final_answer` equals gold (excluding trivial answers like "0", "1", "2"): flag potential leakage. Rerun `tir_rag` with answer-string-blacklist filtering on retrieval and report both numbers.
- **Saturation check.** If any condition shows `extract_answer == None` rate > 25%, the model is failing to commit. Mirror of the Run-4 convergence-failure diagnostic. Document; do not silently re-prompt.
- **Tool-call rate check.** If `tir` shows tool-call rate < 40%, the prompt is not eliciting tool use; re-examine before reporting numbers.

## What goes in the writeup regardless of outcome

- The locked recipe + hyperparameters above.
- The 18-cell retrieval ablation table (BM25/dense/topic × k × exemplar-type) on dev N=100.
- Per-condition `n_scored`, accuracy, paired McNemar p-values: cot vs tir, tir vs tir_rag, cot vs tir_rag, plus tir vs Qwen3-1.7B base, tir_rag vs Sonnet 4.6.
- Per-condition saturation rate, tool-call rate, mean tool calls/problem.
- Per-topic accuracy stratification on `topics_flat` (mirroring `docs/qwen3_base_topic_breakdown.md`).
- Bucket prose for the cell the result lands in, **verbatim** from the matrix above.
- Brief methodology note on why we used the existing Week-1 500 rather than deriving a hard subset (the dataset author's "frontier-LLM-fail ≠ olympiad-hard" observation).

## Budget

Realistic estimate: **$15-25 total**, back inside the original envelope.

| Line | Estimate | Rationale |
|---|---|---|
| Headline judge spend (3 conditions × 500 problems) | $10-12 | Cheap-grader resolves ~40% per Week-1 finding; judge takes the remaining ~60%. **Judge is load-bearing for the headline pass** — without it the 60% objectively-correct-but-non-trivially-extracted answers are silently misgraded. |
| Floor + cheap-tier + frontier anchors on the 500 | $0 | Already paid for from Week 1 (Qwen3-1.7B base, GPT-5.4 Mini, Sonnet 4.6 grades exist). |
| 18-cell retrieval ablation on dev N=100 | <$3 | Cheap-grader only — 1,800 generations × cheap-grader. No judge spend. |
| Optional: frontier+RAG ceiling (Sonnet 4.6 + keyword-RAG, 500 fresh) | $14 | Decision deferred. Adds the more honest C2 secondary-question framing; not required for the headline T1/T2/T3 question. |
| Embedding for dense retrieval | <$1 | Local `bge-small-en-v1.5`, free. |
| Vast.ai contingency if Hyak is contested | $0-7 | ~5-10h of 48 GB inference if needed. |
| Buffer for re-runs / debug | $2-5 | Default allocation per the project policy. |

**Spend tracking:** `summary.json` per pass writes API spend exactly like Week-1 results. Project-total tracked against this budget; flag at 50% / 75% / 100% of upper bound.

## Risks

- **Concurrent-research risk.** MathNet paper is recent (April 2026); authors may publish their own test-hard split during our project window. Mitigation: our eval is the Week-1 500 (a stratified random sample, well-defined and committed), so the writeup remains a separately-defined measurement on a separately-defined set even if a different test-hard appears later.
- **Bank-yield risk.** TIR exemplar bank yield from `Qwen2.5-Math-1.5B-Instruct` may be < 200 rows. Three-step fallback ladder documented in §TIR exemplar bank construction.
- **Distribution-shift framing risk.** "MathNet's distribution is different from OlympiadBench's" is the headline claim's load-bearing fact. We support it qualitatively (different country/competition mix, different topic balance) but do not formally measure distributional distance. If a reviewer challenges this, the response is: MathNet draws from 60+ countries' olympiad problems, OlympiadBench from a CEE+Competition split dominated by Chinese sources — face-valid distribution shift.

## What this experiment does NOT test

- **Different base.** Sticking with `Qwen2.5-Math-1.5B-Instruct` to keep the TIR-transfer question well-formed against Alibaba's published numbers. No DeepSeek-Math or Llemma comparison.
- **A "hard" subset of MathNet.** Dataset author confirmed this is genuinely hard to define; we declined to hand-construct one.
- **Multi-step agentic planning.** TIR is single-loop tool use within a single response, not multi-turn agent planning.
- **Tools beyond Python.** No web search, no Wolfram, no theorem provers.
- **Other 1-3B math models.** No base sweep.
- **Larger compute / multi-rollout.** Greedy single-rollout. No best-of-N or self-consistency on TIR.
- **Pure-CoT-RAG vs RAG-of-TIR.** The retrieval ablation includes a CoT-exemplar control, but the *generation condition* is always TIR for `tir_rag`.
- **Paper's `MathNet-Solve-Test` (6,400).** Costs scale 13× and the paired-comparison story (against Week-1 500 numbers) holds on the existing 500.
