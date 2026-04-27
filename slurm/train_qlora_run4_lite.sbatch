#!/bin/bash
# Run 4 LITE: train + merge directly on the existing
# data/splits/train_self_distill.jsonl (65 rows from the partial pilot
# data gen on Sunday night — pilot was preempted and never restarted, so
# we salvage the data we have rather than waiting for the full
# chain to clear).
#
# 65 rows × eff batch 4 × 2 epochs ≈ 32 opt steps. At LR 1e-5 with
# r=64 this is enough for a real signal — close to what the original
# pilot was designed to test.
#
# Pipeline:
#   1. verify response template alignment on existing self-distill data
#   2. train (LR 1e-5, eff batch 4, 2 epochs, max_seq 8192,
#      enable_thinking=True, completion_only_loss)
#   3. merge -> bf16 standalone
# (eval is in slurm/eval_qwen3_run4.sbatch — submit chained via afterok)
#
#SBATCH --job-name=qlora-run4-lite
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] qlora-run4-lite starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
ADAPTERS_ROOT=/gscratch/scrubbed/sanmarco/adapters
OUT=$ADAPTERS_ROOT/qwen3-mathnet-run4
MERGED=${OUT}-merged
DISTILL_DATA=$REPO/data/splits/train_self_distill.jsonl

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs "$ADAPTERS_ROOT"

LOG=logs/qlora_run4_lite_${SLURM_JOB_ID}.log
echo "[$(date)] qlora-run4-lite log -> $LOG"

n_rows=$(wc -l < "$DISTILL_DATA" 2>/dev/null || echo 0)
echo "[$(date)] $n_rows rows in $DISTILL_DATA"
if [ "$n_rows" -lt 30 ]; then
    echo "ERROR: too few self-distill rows ($n_rows < 30)."
    exit 3
fi

# === Stage 1: verify response template ===
echo "[$(date)] stage 1: verify response template"
$PY -u scripts/verify_response_template.py \
    --base-model Qwen/Qwen3-1.7B \
    --train-jsonl "$DISTILL_DATA" \
    --n 8 \
    2>&1 | tee -a "$LOG"
rc_v=${PIPESTATUS[0]}
[ "$rc_v" -eq 0 ] || exit $rc_v

# === Stage 2: train ===
echo "[$(date)] stage 2: train (LR=1e-5, eff batch 4, 2 epochs, max_seq 8192)"
$PY -u scripts/train_qlora.py \
    --base-model Qwen/Qwen3-1.7B \
    --out-dir "$OUT" \
    --train-jsonl "$DISTILL_DATA" \
    --eval-jsonl  data/splits/eval.jsonl \
    --mid-eval-log-path "$OUT/mid_eval.jsonl" \
    --lora-r 64 \
    --lora-alpha 128 \
    --max-seq-length 8192 \
    --num-train-epochs 2 \
    --learning-rate 1e-5 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --completion-only-loss \
    --enable-thinking \
    --mid-eval-n 0 \
    2>&1 | tee -a "$LOG"
rc_train=${PIPESTATUS[0]}
[ "$rc_train" -eq 0 ] || exit $rc_train

# === Stage 3: merge ===
echo "[$(date)] stage 3: merge"
$PY -u scripts/merge_adapter.py \
    --base-model Qwen/Qwen3-1.7B \
    --adapter-dir "$OUT" \
    --out-dir "$MERGED" \
    2>&1 | tee -a "$LOG"
rc_m=${PIPESTATUS[0]}
ls -la "$MERGED" || true
echo "[$(date)] Run 4 LITE train+merge complete. Submit slurm/eval_qwen3_run4.sbatch next."
exit $rc_m
