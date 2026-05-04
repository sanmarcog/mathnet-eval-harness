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
