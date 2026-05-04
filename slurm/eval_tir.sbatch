#!/bin/bash
# Week-2 TIR / TIR+RAG / CoT eval driver.
#
# Submit with MODE env var:
#   MODE=cot     sbatch slurm/eval_tir.sbatch
#   MODE=tir     sbatch slurm/eval_tir.sbatch
#   MODE=tir_rag sbatch slurm/eval_tir.sbatch
#
# Default MODE is `tir`. For tir_rag, expects results/tir/exemplar_bank.jsonl
# to exist (build it first via slurm/build_tir_exemplar_bank.sbatch).
#
# Pre-flight: this sbatch refuses to launch unless smoke has passed in the
# current git commit. Run on the login node first:
#
#   python scripts/check_tir_sandbox.py
#   python scripts/eval_tir.py --mode $MODE --smoke
#
# That writes logs/tir_smoke_passed_<commit>.txt; the script below checks
# for it.
#
# NOTE (deviation ruling 3, 2026-04-30): this sbatch does NOT require the
# --smoke-real sentinel. Only the bank-build sbatch needs it, because
# only bank construction runs the real-model TIR loop end-to-end at scale;
# eval_tir consumes a pre-built bank and is otherwise covered by the
# cheap smoke + sandbox sentinels.
#
# ----------------------------------------------------------------------
# Launch-order dependency graph (Hyak serializes our jobs — submission
# order = run order, modulo --dependency chains). Optimize for earliest
# wall-clock completion of the headline three conditions:
#
#   MODE=cot      ─┐
#   MODE=tir      ─┤ no dependencies — submit immediately after
#                 │ vLLM smoke passes
#                 │
#   bank-build-tir ─┐
#   bank-build-cot ─┤
#   ablation       ─┤ chain via --dependency=afterok:
#                  │   ablation depends on both bank builds
#   MODE=tir_rag   ─┘   tir_rag depends on ablation winner
#
# The cot/tir sbatches consume only the existing Week-1 anchors + the
# eval split — they do NOT need any TIR/CoT exemplar bank or ablation
# winner, so submitting them BEFORE the bank-build chain means they hit
# the queue earlier and finish earlier. Concretely:
#
#   vllm_jid=$(sbatch --parsable slurm/eval_tir_vllm_smoke.sbatch)
#   MODE=cot sbatch --dependency=afterok:$vllm_jid slurm/eval_tir.sbatch
#   MODE=tir sbatch --dependency=afterok:$vllm_jid slurm/eval_tir.sbatch
#   tir_bank_jid=$(sbatch --parsable --dependency=afterok:$vllm_jid slurm/build_tir_exemplar_bank.sbatch)
#   cot_bank_jid=$(sbatch --parsable --dependency=afterok:$vllm_jid slurm/build_cot_exemplar_bank.sbatch)
#   abl_jid=$(sbatch --parsable --dependency=afterok:$tir_bank_jid:$cot_bank_jid slurm/run_retrieval_ablation.sbatch)
#   MODE=tir_rag sbatch --dependency=afterok:$abl_jid slurm/eval_tir.sbatch
#
# Don't accidentally serialize cot/tir behind the bank builds — that
# costs hours of wall-time on a saturated queue.
#
#SBATCH --job-name=eval-tir
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h200:1
#SBATCH --time=14:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] eval-tir starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs

MODE="${MODE:-tir}"
COMMIT="$(git rev-parse --short HEAD)"
SENTINEL="logs/tir_smoke_passed_${COMMIT}.txt"
if [ ! -f "$SENTINEL" ]; then
    echo "[$(date)] FAIL: smoke sentinel $SENTINEL not found for commit $COMMIT"
    echo "Run on login node first:"
    echo "  python scripts/check_tir_sandbox.py"
    echo "  python scripts/eval_tir.py --mode $MODE --smoke"
    exit 2
fi
echo "[$(date)] smoke sentinel found: $SENTINEL"

# Output dir suffix _7b preserves the prior 1.5B headline results
# (results/tir/{cot,tir}/) as a noisy-baseline reference. Per the
# 2026-05-04 user decision: 1.5B accuracy 4.6%/1.2% with Wilson 95% CI
# ±1.8 pp made cell-to-cell ablation deltas noise-bound; 7B should land
# in a band where deltas are interpretable.
LOG="logs/eval_tir_${MODE}_7b_${SLURM_JOB_ID}.log"
OUT="results/tir/${MODE}_7b"
echo "[$(date)] mode=$MODE  model=7B  out=$OUT  log=$LOG"

EXTRA_ARGS=()
if [ "$MODE" = "tir_rag" ]; then
    if [ ! -f results/tir/exemplar_bank.jsonl ]; then
        echo "[$(date)] FAIL: results/tir/exemplar_bank.jsonl not found."
        echo "Build it first: sbatch slurm/build_tir_exemplar_bank.sbatch"
        exit 3
    fi
    EXTRA_ARGS+=(--bank results/tir/exemplar_bank.jsonl --retrieval-policy bm25 --k 3)
fi

$PY -u scripts/eval_tir.py \
    --mode "$MODE" \
    --eval-jsonl data/splits/eval.jsonl \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --backend vllm \
    --max-new-tokens 4096 \
    --max-tool-calls 4 \
    --use-judge \
    --out "$OUT" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[$(date)] eval_tir.py rc=$rc"
exit $rc
