#!/bin/bash
# Run 2: QLoRA fine-tune of Qwen3-1.7B on multilingual MathNet, recipe
# matched to Alibaba's published Qwen2.5-Math 1.5B SFT (the only known
# successful math fine-tune of this model family at this size).
# Reference: https://arxiv.org/html/2409.12122v1
#
# Hyperparameters per their report:
#   learning_rate = 2e-5  (decay to 7e-7)
#   batch_size    = 128 effective
#   epochs        = 3
#   seq_length    = 4096
#
# Our additions on top of the recipe:
#   QLoRA r=64 / alpha=128 (we have ~10K rows, much less than their 2.5M;
#                           rank gives the adapter capacity to compensate)
#   completion_only_loss   (mask system+user tokens)
#   enable_thinking=False in chat template (training data has no <think>)
#
# Pipeline:
#   0. verify response template alignment (pre-flight)
#   1. train  -> adapters/qwen3-mathnet-run2/
#   2. merge  -> adapters/qwen3-mathnet-run2-merged/
# (eval is in a separate sbatch slurm/eval_qwen3_run2.sbatch so the
#  train+merge fits comfortably in a single ckpt-all walltime window)
#
# Submit:
#     sbatch slurm/train_qlora_run2.sbatch
#
#SBATCH --job-name=qlora-qwen3-run2
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=a40:1
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] qlora-qwen3-run2 starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
ADAPTERS_ROOT=/gscratch/scrubbed/sanmarco/adapters
OUT=$ADAPTERS_ROOT/qwen3-mathnet-run2
MERGED=${OUT}-merged
EVAL_OUT=$REPO/results/full/qwen3-1.7b-run2

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs "$ADAPTERS_ROOT" "$EVAL_OUT"

LOG=logs/qlora_run2_${SLURM_JOB_ID}.log
echo "[$(date)] launching Run-2 (recipe-matched); log -> $LOG"

# === Stage 0: verify response template alignment ===
echo "[$(date)] stage 0: verify response template alignment"
$PY -u scripts/verify_response_template.py \
    --base-model Qwen/Qwen3-1.7B \
    --train-jsonl data/splits/train_multilingual_filtered.jsonl \
    --n 16 \
    2>&1 | tee -a "$LOG"
rc_verify=${PIPESTATUS[0]}
echo "[$(date)] verify rc=$rc_verify"
[ "$rc_verify" -eq 0 ] || { echo "ABORTING: response template misaligned"; exit $rc_verify; }

# === Stage 1: train (Alibaba recipe + our QLoRA wrapper) ===
echo "[$(date)] stage 1: train, lr=2e-5, 3 epochs, effective batch 128"
$PY -u scripts/train_qlora.py \
    --base-model Qwen/Qwen3-1.7B \
    --out-dir "$OUT" \
    --train-jsonl data/splits/train_multilingual_filtered.jsonl \
    --eval-jsonl  data/splits/eval.jsonl \
    --mid-eval-log-path "$OUT/mid_eval.jsonl" \
    --lora-r 64 \
    --lora-alpha 128 \
    --max-seq-length 4096 \
    --num-train-epochs 3 \
    --learning-rate 2e-5 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 32 \
    --completion-only-loss \
    --mid-eval-n 10 \
    2>&1 | tee -a "$LOG"
rc_train=${PIPESTATUS[0]}
echo "[$(date)] train rc=$rc_train"
[ "$rc_train" -eq 0 ] || exit $rc_train
ls -la "$OUT" || true

# === Stage 2: merge LoRA into bf16 base ===
echo "[$(date)] stage 2: merge LoRA adapter into base -> $MERGED"
$PY -u scripts/merge_adapter.py \
    --base-model Qwen/Qwen3-1.7B \
    --adapter-dir "$OUT" \
    --out-dir "$MERGED" \
    2>&1 | tee -a "$LOG"
rc_merge=${PIPESTATUS[0]}
echo "[$(date)] merge rc=$rc_merge"
[ "$rc_merge" -eq 0 ] || exit $rc_merge
ls -la "$MERGED" || true

echo "[$(date)] train+merge complete. Submit slurm/eval_qwen3_run2.sbatch next."
exit 0
