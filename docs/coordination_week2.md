# Week 2 coordination file (live)

This is the async communication channel between two Claude Code sessions while the user is offline (~36h starting 2026-05-02 21:00 PDT).

- **Tab A** = main session that did the Dr. GRPO chapter; has klone SSH access via the user's ControlMaster socket; will be polling cluster state via ScheduleWakeup.
- **Tab B** = Week 2 coder; built the TIR-RAG scaffolding; has deeper context on the pre-reg, the bank-build code, the ablation runner, and watcher v2/v3. Dormant unless invoked.

Both sides read this file at the start of any turn. Both sides append to the AUDIT log when they take action. Disagreements go in OPEN DECISIONS and wait for the user.

---

## Locked rules (no negotiation)

These were agreed with the user before they went offline. Either tab follows them; neither modifies them.

1. **Bank-clearance threshold: ≥40 rows in bank-tir.** If bank-tir lands ≥40 rows, proceed to ablation. If <40, pause and write to OPEN DECISIONS.
2. **Topic skew is acceptable; document, do not fix.** Heavily Algebra-skewed bank is expected. Write the methodology footnote describing it. Do not escalate to multilingual / BGE-M3 / Sonnet bank during this window.
3. **Competition skew is acceptable; document, do not fix.** South African Math Olympiad is ~31% of the bank. Same rule.
4. **Ablation winner cell = locked tie-break from `docs/tir_rag_plan.md`**: highest dev-100 accuracy → smaller k → BM25 over dense → TIR-exemplar over CoT-exemplar. Deterministic from the data. Either tab can compute and commit the winner without further consultation.
5. **Sonnet+RAG ceiling pass: REQUIRES USER OK.** Do not launch this. If ablation lands and ceiling pass is the natural next step, write to OPEN DECISIONS and wait for user return.
6. **Major pre-reg deviations beyond what's already documented: REQUIRES USER OK.** Log in OPEN DECISIONS, do not act.
7. **GPU/queue management is automatic per existing playbook**: when one of a hedge group enters RUNNING, scancel the others. When jobs preempt, the resume-safe builders handle it.
8. **Both tabs have all permissions and should troubleshoot autonomously.** User explicit directive (2026-05-02 13:38 PDT): don't punt fixable issues to OPEN DECISIONS. Restart killed orchestrators, re-create missing sentinels, fix obvious bugs, fall back from primary actions to alternates when primaries fail. Reserve OPEN DECISIONS for: Sonnet+RAG (>$5 spend), major pre-reg deviations, situations where you're genuinely uncertain what the right action is.

9. **Web search over guessing.** User explicit directive (2026-05-02 13:39 PDT): if you don't know something during troubleshooting (a TRL flag, a slurm directive, a Qwen2.5-Math behavior, a sympy quirk), websearch or webfetch the canonical source — don't guess and ship a fix that's structurally wrong. The cost of a wrong fix in this 36h window is higher than the cost of a 30-second search.

10. **Active collaboration over parallel monoculture.** Each tab has different context: Tab A has Dr. GRPO chapter context + general benchmarking judgment + cron polling; Tab B has TIR-RAG scaffolding + the orchestrator script + the sandbox/retriever code. When you encounter something that's clearly the other tab's specialty, write a clear hand-off in the audit log naming the question, not a vague flag. The other tab's next invocation should be able to act on it directly.

---

## Current state (last updated by Tab A 2026-05-02 13:30 PDT)

### Live klone state (squeue + bank file sizes)
- 34994520 (bank-tir 7B, h200): **RUNNING**, 5h 2m elapsed, time-limit 9h 5m
- 34994521 (bank-cot 7B, h200): PENDING, AssocMaxJobsLimit
- 34985530 (cot/tir headline 1.5B, l40s): PENDING, AssocMaxJobsLimit
- 34985531 (cot/tir headline 1.5B, a40): PENDING, AssocMaxJobsLimit
- 34985532 (cot/tir headline 1.5B, l40s): PENDING, AssocMaxJobsLimit
- 34985533 (cot/tir headline 1.5B, a40): PENDING, AssocMaxJobsLimit

`AssocMaxJobsLimit` = the demo-ckpt 1-running-job-at-a-time cap. Pending jobs unlock when bank-tir finishes.

### Bank file contents (live)
- bank-tir 7B: **66 rows** on disk (26 carry-over from 1.5B + 40 from 7B so far)
- bank-cot: 20 rows preserved from earlier 1.5B run; 7B run will append

### Topic distribution snapshot (at 51 rows)
- Algebra 52.9%, NT 25.5%, Discrete 15.7%, Geometry 3.9%, MWP 2.0%
- Competition concentration: South African Math Olympiad 31% (16/51)

### Headline runs
- 4 sbatches PENDING on 1.5B (jids 34985530-33). Order of operations: cot/tir can start the moment vLLM smoke sentinel exists; tir_rag waits for ablation winner.

### Watcher state
- Watcher v3 running on klone, no alerts since deployment per Tab B's last report.

---

## Open decisions

(Populated when either tab encounters something that needs the other tab or the user.)

*[empty as of 2026-05-02 21:00]*

---

## Audit log (append-only, newest at top)

### 2026-05-04 08:35 PDT — Tab B — executed Tab A 08:25 hand-off (1.5B → 7B headline pivot)

All 5 items from Tab A's hand-off executed.

**Code changes (commit `50cf9b2` pushed to main):**
- `slurm/eval_tir.sbatch`:
  - `--gpus-per-node=l40s:1` → `h200:1`
  - `--mem=48G` → `64G`
  - `--model Qwen/Qwen2.5-Math-1.5B-Instruct` → `Qwen/Qwen2.5-Math-7B-Instruct`
  - `OUT="results/tir/${MODE}"` → `OUT="results/tir/${MODE}_7b"` (preserves 1.5B results per item 4)
- `docs/tir_rag_plan.md`: added 2026-05-04 correction block — explains the pivot, notes it reverts the 2026-05-02 "in-context distillation" framing back to the originally-pre-registered "self-help via retrieval" since both bank construction and inference now run on the same model. The 1.5B headlines are documented as a noisy-baseline reference.

**Submitted (Klone, commit `50cf9b2` pulled, sentinel touched):**
- `35018779` — MODE=cot 7B headline (PENDING, AssocMaxJobsLimit)
- `35018780` — MODE=tir 7B headline (PENDING, AssocMaxJobsLimit)
- No hedges this round — h200 has been hitting the queue reliably, single-job-per-stage acceptable, and submitting fewer jobs respects the 1-running-job cap.

**tir_rag chain:** orchestrator (PID 622145) auto-submits `MODE=tir_rag sbatch ... slurm/eval_tir.sbatch` after ablation completes. Since the sbatch now defaults to 7B + `${MODE}_7b` output dir, the tir_rag run will land in `results/tir/tir_rag_7b/` automatically. Verified by reading the sbatch — no orchestrator update needed.

**Expected serial wall-clock (Hyak MaxJobs=1, FIFO modulo dependency):**
- bank-cot finishes ~14:30 PDT today (currently 3:11 elapsed, 5:53 left)
- cot 7B starts ~14:30, finishes ~20:30 (6h estimate, 7B is ~3-4× slower than 1.5B)
- tir 7B starts ~20:30, finishes ~02:30 May 5
- Ablation submitted by orchestrator ~14:30 (when bank-cot done + clearance set), enters queue ~02:30 May 5, finishes ~07:30 May 5
- tir_rag 7B starts ~07:30 May 5, finishes ~13:30 May 5
- **All-done estimate: Tuesday May 5 ~13:30 PDT.**

**Not changed:**
- Orchestrator GROUP map still references old 1.5B headline jobids (34985530-33). Harmless — those are terminal states (COMPLETED or CANCELLED), orchestrator polls them, marks DONE/FAILED, takes no further action. Adding 35018779/35018780 to GROUP would only matter if we'd submitted hedges (we didn't), since GROUP is just for scancel-the-losers-when-winner-runs.
- Watcher v3 will pick up the new 7B headlines automatically when they enter R; its checks are state-driven not jobid-driven.

**Tab A handoff back:** monitoring is yours from here per your 08:25 note ("Tab A staying available to monitor cluster state"). I'll wake on user invocation or if a TIR-RAG-specific issue surfaces (e.g. ablation cell-format issue, tir_rag chain breaks because orchestrator dies).

### 2026-05-04 14:25 PDT — Tab B — answered Tab A urgent review; ablation patched+resubmitted at 7B

**Q1 (ablation model):** Confirmed bug. `slurm/run_retrieval_ablation.sbatch` hardcoded 1.5B + l40s + 5h. Cell rankings would NOT reliably transfer cross-model — 1.5B saturates and may favor smaller k to preserve context budget; 7B has more context headroom and may favor richer k. Ablation must run on the deployment model.
- Patched in commit `d55ad8e`: `--model 1.5B → 7B`, `--gpus-per-node l40s → h200`, `--mem 48G → 64G`, `--time 5h → 12h` (7B slower per problem; 1800 dev generations could push 8-10h worst-case).
- Cancelled `35024821` (the 1.5B ablation Tab A submitted).
- Resubmitted as `35025478` (7B/h200).

**Q2 (orchestrator tir_rag uses 7B):** Confirmed correct. `scripts/hedge_orchestrator.sh:86` calls `MODE=tir_rag sbatch ... slurm/eval_tir.sbatch`. The sbatch hardcodes 7B + h200 + `${MODE}_7b` output dir per my 08:35 commit. tir_rag will land in `results/tir/tir_rag_7b/`. Verified by reading code.
- Since I cancelled+resubmitted ablation manually (orchestrator was dead), I also submitted tir_rag manually with `--dependency=afterok:35025478`. Jobid `35025479`.

**Q3 (resume bug in build_cot_exemplar_bank.py):** Confirmed. The `already_kept_ids` set only tracks SUCCESSFUL rows. Failed rows get retried on resume, and with greedy decoding (`temperature=0`) the failures are deterministic — every restart re-processes the same failed prefix forever. Bank-cot is moot now (58 rows accepted), but for future bank builds the fix is ~10 lines: also track attempted-failed IDs to a sidecar `.failed_ids` file, skip them on resume too. Not patching now (bank-cot is done) — flagging for the writeup's "lessons learned" section.

**Cleanup actions:**
- Cancelled `35018779` (cot_headline loser; `35024646` is the running winner on a40).
- Killed both orchestrators (`1465721`, `1475624`) and started a single fresh one (`1562597`) with stripped-down GROUP map = `tir_headline` only. Bank groups + cot_headline removed (terminal). New orchestrator can't trip the hard-fail-on-bank-CANCELLED branch since those groups aren't being polled anymore.
- Watcher v3 (`622277`) still alive from earlier.

**Current queue (14:25 PDT):**
- `35024646` RUNNING — cot 7B headline (a40, 45min in, 8:19 left)
- `35018780` PENDING — tir 7B headline (h200) [hedge w/ 35024647]
- `35024647` PENDING — tir 7B headline hedge (a40)
- `35025478` PENDING — ablation 7B (h200)
- `35025479` PENDING — tir_rag 7B (h200, dep on 35025478)

**Tab A handoff back:** monitoring is yours per usual. The orchestrator is now lightweight (only watches tir_headline group for hedge-loser-cancel). If anything in the chain breaks, I'm reachable. The most likely TIR-RAG-specific issue: ablation completes but writes to `results/tir/ablation/ablation_summary.json` with cells that don't pass the locked tie-break sanity check (e.g. all dev-100 cells at 0% accuracy if 7B can't handle the dev split well). I'll wake on that.

### 2026-05-04 14:10 PDT — Tab A — bank-cot was stuck in resume loop; broke the deadlock, submitted ablation, restarted orchestrator. URGENT REVIEW REQUEST FOR TAB B.

User came back, asked what was happening, was frustrated by lack of progress. I dug in. Diagnosis:

**Bank-cot was deterministically stuck.** First run (Sat) processed problems 1-1349 of 3476, kept 38 rows, was preempted at problem 1349. The May 4 re-run started from problem 1 again because the resume logic only tracks *kept* IDs, not *failed* ones. With T=0 greedy decoding, re-processing the failed prefix produces identical failures. 6+ hours of GPU time produced zero new rows. Each preemption + restart = stuck near problem 1349 forever.

**Actions I took (rule 8: troubleshoot autonomously):**
1. `scancel 34994521` — bank-cot, was making zero progress per restart
2. `scancel 35024673` — auto-resubmit of bank-cot
3. `sbatch slurm/run_retrieval_ablation.sbatch` → jobid 35024821 (orchestrator was dead, ablation never queued)
4. Restarted orchestrator (PID 1475624). Noted there's also a `1465721 hedge_orchestrator.sh` alive — unclear who started it, file-based state should make double-running safe but Tab B should clean this up if not.

**Live state right now (14:10 PDT):**
- 35024646 RUNNING — cot 7B headline (mode=cot, model=Qwen2.5-Math-7B-Instruct, output=results/tir/cot_7b/), 8 min elapsed, just finished vLLM init
- 35024821 PENDING — ablation (1.5B per current sbatch, runs on dev N=100)
- 35024647 PENDING — eval-tir hedge
- 35018780 PENDING — original tir 7B headline
- AssocMaxJobsLimit means these run one-at-a-time

**Bank state (final, won't grow further):** TIR 86 rows, CoT 58 rows. Both clear locked rule 1 (≥40). Topic skew documented per rule 2.

**REVIEW REQUESTS for Tab B (your specialty):**
1. **Is the ablation correctly configured for the 7B pivot?** The current `slurm/run_retrieval_ablation.sbatch` loads Qwen2.5-Math-1.5B-Instruct at the top of `scripts/run_retrieval_ablation.py`. Should the ablation also be running at 7B to match the headlines, or is the cell-selection on dev-100 expected to transfer cross-model? You wrote this code; you'd know.
2. **Verify the orchestrator submits tir_rag at 7B post-ablation.** When ablation finishes and the orchestrator picks the winner cell, does it submit `MODE=tir_rag sbatch slurm/eval_tir.sbatch` which now points at 7B per your 08:35 commit? Confirm by reading `scripts/hedge_orchestrator.sh` for the tir_rag submission command. If it hardcodes 1.5B somewhere, fix it.
3. **The resume bug in `scripts/build_cot_exemplar_bank.py`** — only kept-IDs are tracked, not failed-IDs. With greedy decoding this means restarts re-do the failed prefix deterministically. Bank-cot is moot now (we accepted 58 rows), but worth flagging for any future bank build that uses greedy + resume. ~10 line fix: also write attempted-failed IDs to a sidecar file, skip them on resume too.

**Tab A status post-action:** I'll keep monitoring via cron (resumes since Mac is awake again). User is back online and can adjudicate. Tab B needs to be invoked to address (1)-(2) above before tir_rag headline launches, otherwise we'll get a 1.5B winner cell selected for a 7B headline, which is methodologically muddy.

### 2026-05-04 08:25 PDT — Tab A — user decided Option B: pivot headlines to 7B. HAND-OFF to Tab B.

User reviewed the 1.5B headline results (CoT 4.6% / TIR 1.2% / 96% saturation) and decided **the 1.5B headline distribution is too noisy for any meaningful TIR-RAG-vs-CoT comparison.** The pre-reg locked 1.5B but the comparison can't be defended at that accuracy band (Wilson 95% CI ±1.8 pp on 4.6%; cell-to-cell deltas would all be in noise).

**Decision: rerun cot/tir/tir_rag headlines on Qwen2.5-Math-7B-Instruct** (the same model that built the bank). Pre-reg deviation, documented in writeup the same way the Dr. GRPO chapter handled its `max_completion_length` and chat-template deviations. The 1.5B results stay as a "reference baseline" that we report alongside, not as the headline.

**Hand-off to Tab B (your specialty: sbatches, orchestrator, the eval pipeline):**

1. Update `slurm/eval_tir.sbatch` (or whatever the parametric model arg is) to point at `Qwen/Qwen2.5-Math-7B-Instruct` instead of `Qwen/Qwen2.5-Math-1.5B-Instruct`. The cot/tir/tir_rag headlines should all use 7B going forward.
2. Bump GPU resource request as needed for 7B + vLLM KV cache. h200 preferred (you've validated 7B fits there during bank-builds); l40s/a40 likely tight depending on context budget.
3. Once bank-cot completes and ablation finishes, the orchestrator should auto-submit tir_rag at 7B; verify the model arg propagates through the dependency chain.
4. **DO NOT cancel the 1.5B 34985531 / 34985533 results** — those are preserved as the noisy-baseline reference. Just run new 7B headlines under different result-dir names so postprocess can pick them up separately.
5. Documentation: add a deviation note to `docs/tir_rag_plan.md` under a "2026-05-04 corrections" block. The 1.5B → 7B headline switch is the deviation; the rationale is the noise floor at 1.5B (4.6% with ±1.8 pp Wilson CI) is below the magnitude of any plausible cell-to-cell ablation effect.

**My current understanding of pipeline state:**
- bank-cot 7B: RUNNING ~2h 39m, expected finish today
- ablation: gated on bank-cot completion; orchestrator (PID 622145, fresh 24h budget) will pick it up
- new 7B headlines: pending Tab B's sbatch update + queue submission

User is back online; can adjudicate if anything ambiguous comes up. Tab A staying available to monitor cluster state and write up findings; Tab B owns the pipeline-config side.

### 2026-05-04 08:18 PDT — Tab B — recovered from 36h offline-window failure; ran headlines flagged with audit-trigger

User invoked me with "you didn't do anything." Correct from my perspective (was dormant per protocol). Tab A's automation also did not fire any audit entries during the 36h window. Reconstruction from sacct + log files on klone:

**What did happen during the window:**
- bank-tir 7B (34994520) COMPLETED 2026-05-02 15:39, final 86 rows. >40 threshold → satisfies locked rule 1.
- cot headline (34985531, a40 hedge) COMPLETED 2026-05-02 19:17. **23/500 = 4.6% accuracy** (use_judge=true; methods: miss 477, exact 17, judge 5, symbolic 1; n_saturated=0).
- tir headline (34985533, a40 hedge) COMPLETED 2026-05-03 04:59. **6/500 = 1.2% accuracy, 481/500 = 96.2% saturated** (methods: miss 494, exact 3, judge 3; total tool-calls 833 across 500 problems = 1.67/problem avg).
- bank-cot 7B (34994521) was running on g3128 May 2 19:18 → preempted before 09:10 May 3 (watcher hit 24h timeout while it was still running) → requeued by Slurm → re-started 2026-05-04 05:34 on g3126, currently 2:41 elapsed of 9:05 walltime.
- Orchestrator (started May 2 10:46 PID 1923813) ran 24h, hit POLL_MAX 2026-05-03 10:51, exited. Watcher v3 (started ~May 2 same time) hit its own 24h limit at 09:10 May 3, exited.
- `.hedge_state` contains `fatal_failure=bank_tir` (artifact from earlier orchestrator run before this one — bank_tir was actually COMPLETED at sacct level) AND `ablation_status=TIMEOUT` (POLL_MAX exit). Misleading but not actionable.

**What did NOT happen:**
- `.ablation_clearance` flag was NEVER created. Tab A's cron either did not fire or fired but did not act on the bank-tir-landed branch.
- Ablation never ran. Results dir does not exist.
- tir_rag headline never submitted (depends on ablation winner).

**Audit-trigger fired on tir headline (per pre-reg §Audit triggers):**
> "Saturation check. If any condition shows extract_answer == None rate > 25%, the model is failing to commit. Document; do not silently re-prompt."

96.2% >> 25% threshold. Per pre-reg discipline I am NOT re-running with adjusted budget; the saturation finding gets reported as part of the writeup. The TIR-vs-CoT delta of -3.4 pp lands in interpretation matrix bucket **T3 (No / negative transfer)**, with the saturation diagnostic reinforcing the pre-reg's note that "the convergence-failure mode the base was prone to" is being amplified rather than rescued by tools at this scale.

**Recovery actions (rule 8 autonomous fix):**
1. Touched `.ablation_clearance` with timestamp note (86 rows ≥ 40 satisfies rule 1).
2. Restarted orchestrator (PID 622145, fresh 24h budget). Re-poll discovered WINNER[bank_cot]=34994521 immediately; will detect WINNER_DONE[bank_tir] from sacct on next poll.
3. Restarted watcher v3 (PID 622277, fresh 24h budget). One duplicate process (622298) self-cleaned.
4. GROUP map verified correct (`bank_tir=34994520 bank_cot=34994521 cot_headline=34985530 34985531 tir_headline=34985532 34985533`).

**Expected timeline going forward (assuming no further preemption):**
- bank-cot finishes ~14:30 PDT today (May 4)
- Orchestrator detects both bank winners DONE + clearance set → submits ablation
- Ablation runs ~5h → finishes ~19:30 PDT
- Orchestrator auto-submits tir_rag with --dependency=afterok
- tir_rag runs ~5h → finishes ~00:30 PDT May 5
- Then I compute the winner cell from `results/tir/ablation/ablation_summary.json` per locked tie-break

**Tab A handoff** (per rule 10): if your cron is still scheduled, the recovered orchestrator + watcher will handle the rest. If your cron is dead too, please don't try to also touch `.ablation_clearance` — it's already set. The most useful Tab A action right now is investigating WHY your cron didn't act on the bank-tir-landed branch on 2026-05-02 evening (possible: stuck on auth, or the conditional logic to compare bank-row count never resolved). Worth a post-mortem entry for the writeup.

### 2026-05-02 13:39 PDT — Tab A — added rules 9 (websearch over guessing) and 10 (active collaboration)
User directive: web-search canonical sources before shipping fixes; explicit hand-offs between tabs (named question in the audit log, not a vague flag) so the other tab can act directly when invoked. These are now in Locked rules. Cron fires read the file each cycle so they apply automatically.

### 2026-05-02 13:38 PDT — Tab A — added rule 8 (troubleshoot autonomously), recreated cron
User directive: both tabs have all permissions and should fix issues directly rather than punting to OPEN DECISIONS. Cron job re-issued (2e544b8f) with troubleshooting branches: orchestrator-died-restart, ablation-fallback-submission, etc. Reserve OPEN DECISIONS for genuinely judgment calls (Sonnet+RAG spend, major deviations).

### 2026-05-02 13:36 PDT — Tab A — channel test successful, ack Tab B
Read Tab B's 13:35 entry. SSH still works post-extension (verified 13:35:36 PDT klone time). Snapshot:
- bank-tir 7B 5h 8m elapsed, 67 rows (up from 66 / 51-row dist snapshot earlier).
- Orchestrator PID 1923813 confirmed alive (2h 48m elapsed, `scripts/hedge_orchestrator.sh`).
- `.ablation_clearance` flag does NOT exist (correct, bank still running).

**Workflow update from Tab B's note:** the orchestrator auto-submits ablation when `.ablation_clearance` is touched. So my (Tab A) role at bank-tir-landing-time is **touch the clearance flag**, NOT submit the ablation sbatch directly. Updating my cron behavior accordingly:
- bank-tir finished AND ≥40 rows AND `.ablation_clearance` does not yet exist → `ssh ... 'touch /gscratch/scrubbed/sanmarco/mathnet-eval-harness/.ablation_clearance'`. Log to audit.
- bank-tir finished AND <40 rows → write to OPEN DECISIONS, do NOT touch flag.
- ablation finished → orchestrator handles tir_rag headline submission via `--dependency=afterok`. I just monitor. Tab B's fallback applies only if orchestrator dies.

Channel works in both directions. Stepping down to cron-fire cadence.

### 2026-05-02 13:35 PDT — Tab B — briefed, dormant, ready
Read the locked rules + audit log + open decisions. Verified ssh socket alive (job 34994520 bank-tir RUNNING 5:02 elapsed, bank file at 66 rows; matches Tab A's snapshot). No action required at briefing time. When invoked, my expected handoffs are:
- Bank-tir landing → re-check final row count vs locked-rule #1 (≥40); if pass, touch `/gscratch/scrubbed/sanmarco/mathnet-eval-harness/.ablation_clearance` to release the orchestrator's manual gate I added earlier today. The orchestrator (PID 1923813 last I checked) auto-submits ablation + tir_rag from there.
- Ablation landing → compute winner cell from `results/tir/ablation/ablation_summary.json` per the locked tie-break (already implemented in `scripts/run_retrieval_ablation.py::pick_winner`). The headline `tir_rag` job is auto-submitted by the orchestrator with `--dependency=afterok:$ablation_jid`, so no extra submission needed unless the orchestrator died — in which case I fall back to `MODE=tir_rag sbatch --parsable --dependency=afterok:$abl_jid slurm/eval_tir.sbatch` manually.
- Anything else outside locked rules → OPEN DECISIONS, no action.

### 2026-05-02 21:00 PDT — Tab A — coordination file initialized
Set up this file. Tab A will start polling cluster state every 30 min via ScheduleWakeup. Decisions made under the locked rules above will be logged here. Anything outside those rules waits for user or Tab B.

---

## Tab B briefing (read this first if you're Tab B)

Hey. The user is offline for ~36 hours. While you were dormant, I (Tab A) set this file up so we can coordinate.

When you're invoked, do this in order:
1. Read the **Locked rules** section. Don't negotiate.
2. Read the **Audit log** newest-first to see what happened while you were dormant.
3. Read **Open decisions** — that's where I might have written something that needs your TIR-RAG-specific judgment.
4. Take any action you need to. Append to the audit log when you do.
5. If you're making a decision the locked rules cover, you don't need to ask me — just do it and log.
6. If you're making a decision outside the locked rules, write to OPEN DECISIONS and don't act.

I have klone SSH access via the user's ControlMaster (`ssh -S /Users/sanmarco/.ssh/klone-ctrl sanmarco@klone.hyak.uw.edu '<cmd>'`). You have the same access. Use whichever is convenient.

Common commands you might want:
```bash
ssh -S /Users/sanmarco/.ssh/klone-ctrl sanmarco@klone.hyak.uw.edu 'squeue -u sanmarco -o "%.10i %.9P %.8T %.10M %.10l %R %b"'
ssh -S /Users/sanmarco/.ssh/klone-ctrl sanmarco@klone.hyak.uw.edu 'wc -l /gscratch/scrubbed/sanmarco/mathnet-eval-harness/results/tir/exemplar_bank.jsonl'
ssh -S /Users/sanmarco/.ssh/klone-ctrl sanmarco@klone.hyak.uw.edu 'tail -50 /gscratch/scrubbed/sanmarco/mathnet-eval-harness/logs/<latest>.log'
```

If the ControlMaster has died (you'll see `Permission denied` or `socket not found`), do NOT attempt re-auth — write to OPEN DECISIONS and stop.
