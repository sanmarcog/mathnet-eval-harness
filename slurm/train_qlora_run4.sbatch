#!/bin/bash
# Run 4 (full): self-distillation training. Submit AFTER pilot_run4.sbatch
# returns a clear-go signal (paired pilot accuracy >= 30%).
#
# Reuses pilot's 200-row self-distill file via --skip-existing, generates
# 1150 more (1350 total attempted -> ~500 expected correct examples).
#
# Pipeline:
#   1. self-distill data extension to n=1350 (skips pilot's 200)
#   2. verify response template alignment
#   3. train (LR 1e-5, eff batch 8, 1 epoch on ~500 examples = ~63 opt steps)
#   4. merge -> bf16 standalone
# (eval is in slurm/eval_qwen3_run4.sbatch -- separate sbatch to fit walltime)
#
#SBATCH --job-name=qlora-run4
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] qlora-run4 starting on $(hostname)  JOB=$SLURM_JOB_ID"
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

LOG=logs/qlora_run4_${SLURM_JOB_ID}.log
echo "[$(date)] qlora-run4 log -> $LOG"

# === Stage 1: extend self-distill data to n=1350 attempted ===
echo "[$(date)] stage 1: self-distill data extension, n_attempt=1350 (skip-existing)"
$PY -u scripts/build_self_distill_data.py \
    --source-jsonl data/splits/train_multilingual_filtered.jsonl \
    --base-model Qwen/Qwen3-1.7B \
    --out-jsonl "$DISTILL_DATA" \
    --n-attempt 1350 \
    --max-new-tokens 16384 \
    --seed 0 \
    --skip-existing \
    2>&1 | tee -a "$LOG"
rc_gen=${PIPESTATUS[0]}
echo "[$(date)] data gen rc=$rc_gen"
[ "$rc_gen" -eq 0 ] || exit $rc_gen

n_kept=$(wc -l < "$DISTILL_DATA")
echo "[$(date)] $n_kept correct examples in $DISTILL_DATA"
if [ "$n_kept" -lt 200 ]; then
    echo "ERROR: too few self-distill rows ($n_kept < 200); training would be too noisy."
    exit 3
fi

# === Stage 2: verify response template ===
echo "[$(date)] stage 2: verify response template alignment"
$PY -u scripts/verify_response_template.py \
    --base-model Qwen/Qwen3-1.7B \
    --train-jsonl "$DISTILL_DATA" \
    --n 16 \
    2>&1 | tee -a "$LOG"
rc_v=${PIPESTATUS[0]}
echo "[$(date)] verify rc=$rc_v"
[ "$rc_v" -eq 0 ] || exit $rc_v

# === Stage 3: train (full Run 4: eff batch 8, 1 epoch, ~63 opt steps) ===
echo "[$(date)] stage 3: train (LR=1e-5, eff batch 8, 1 epoch, max_seq 8192)"
$PY -u scripts/train_qlora.py \
    --base-model Qwen/Qwen3-1.7B \
    --out-dir "$OUT" \
    --train-jsonl "$DISTILL_DATA" \
    --eval-jsonl  data/splits/eval.jsonl \
    --mid-eval-log-path "$OUT/mid_eval.jsonl" \
    --lora-r 64 \
    --lora-alpha 128 \
    --max-seq-length 8192 \
    --num-train-epochs 1 \
    --learning-rate 1e-5 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --completion-only-loss \
    --enable-thinking \
    --mid-eval-n 0 \
    2>&1 | tee -a "$LOG"
rc_train=${PIPESTATUS[0]}
echo "[$(date)] train rc=$rc_train"
[ "$rc_train" -eq 0 ] || exit $rc_train

# === Stage 4: merge ===
echo "[$(date)] stage 4: merge"
$PY -u scripts/merge_adapter.py \
    --base-model Qwen/Qwen3-1.7B \
    --adapter-dir "$OUT" \
    --out-dir "$MERGED" \
    2>&1 | tee -a "$LOG"
rc_m=${PIPESTATUS[0]}
echo "[$(date)] merge rc=$rc_m"
ls -la "$MERGED" || true

echo "[$(date)] Run 4 train+merge complete. Submit slurm/eval_qwen3_run4.sbatch next."
exit $rc_m
