#!/bin/bash
# Run 2: QLoRA fine-tune of Qwen3-1.7B on multilingual MathNet with
# completion-only loss masking. A40 GPU for the VRAM headroom needed
# by r=64 / alpha=128 / max_seq_length=4096. Training with thinking
# disabled in the chat template (our training data has no <think> traces).
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
#SBATCH --time=10:00:00
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

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs "$ADAPTERS_ROOT"

LOG=logs/qlora_run2_${SLURM_JOB_ID}.log
echo "[$(date)] launching Run-2 training; log -> $LOG"

# Pre-flight: verify the completion_only_loss response template aligns
# with this (model, tokenizer, dataset) triple. Silent-fail mode if not.
echo "[$(date)] pre-flight: verify response template alignment"
$PY -u scripts/verify_response_template.py \
    --base-model Qwen/Qwen3-1.7B \
    --train-jsonl data/splits/train_multilingual.jsonl \
    --n 16 \
    2>&1 | tee -a "$LOG"
rc_verify=${PIPESTATUS[0]}
echo "[$(date)] verify rc=$rc_verify"
[ "$rc_verify" -eq 0 ] || { echo "ABORTING: response template misaligned"; exit $rc_verify; }

# Run 2 hyperparameters:
#   base model: Qwen3-1.7B (up from Qwen2.5-1.5B)
#   LoRA r=64 alpha=128 (up from 16/32; math-recipe standard)
#   max_seq_length=4096 (up from 2048; captures long proofs)
#   train on data/splits/train_multilingual.jsonl (~13K rows vs ~3K English-only)
#   completion-only loss (mask system + user tokens)
#   enable_thinking=False in chat template (no <think> content in training data)
#   batch 4 / grad accum 4 -> effective batch 16 (A40 has the VRAM to go wider)
$PY -u scripts/train_qlora.py \
    --base-model Qwen/Qwen3-1.7B \
    --out-dir "$OUT" \
    --train-jsonl data/splits/train_multilingual.jsonl \
    --eval-jsonl  data/splits/eval.jsonl \
    --mid-eval-log-path "$OUT/mid_eval.jsonl" \
    --lora-r 64 \
    --lora-alpha 128 \
    --max-seq-length 4096 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --completion-only-loss \
    2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "[$(date)] Run-2 training rc=$rc"
ls -la "$OUT" || true

if [ "$rc" -eq 0 ]; then
    MERGED="${OUT}-merged"
    echo "[$(date)] merging LoRA adapter into base -> $MERGED"
    $PY -u scripts/merge_adapter.py \
        --base-model Qwen/Qwen3-1.7B \
        --adapter-dir "$OUT" \
        --out-dir "$MERGED" \
        2>&1 | tee -a "$LOG"
    merge_rc=${PIPESTATUS[0]}
    echo "[$(date)] merge rc=$merge_rc"
    ls -la "$MERGED" || true
fi

exit $rc
