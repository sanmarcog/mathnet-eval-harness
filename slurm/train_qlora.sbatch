#!/bin/bash
# QLoRA fine-tune of Qwen-2.5-1.5B-Instruct on MathNet.
# GPU allocation on ckpt-all; adapter drops to /gscratch/scrubbed/sanmarco/adapters/.
#
# Submit:
#     sbatch slurm/train_qlora.sbatch
# Monitor:
#     tail -f logs/qlora_train_<jobid>.log
#     tail -f /gscratch/scrubbed/sanmarco/adapters/qwen-mathnet-run1/mid_eval.jsonl
#
#SBATCH --job-name=qlora-mathnet
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --time=8:00:00
#
# OOM-fallback enabled by default: on a 2080 Ti (11GB VRAM), the default
# TrainConfig per_device_batch=2 OOMs on step 2 (verified 2026-04-23,
# job 34799557). Sbatch passes batch=1, grad_accum=16 so effective batch
# is still 16. Observed per-step wall is ~50s after warmup -> 378 steps
# ~ 5.3 hours, so walltime bumped to 8h for safety.
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail

echo "[$(date)] qlora-mathnet starting on $(hostname)  JOB=$SLURM_JOB_ID  NODE=$SLURM_NODELIST"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
ADAPTERS_ROOT=/gscratch/scrubbed/sanmarco/adapters
OUT=$ADAPTERS_ROOT/qwen-mathnet-run1

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"

cd "$REPO"
mkdir -p logs "$ADAPTERS_ROOT"

LOG=logs/qlora_train_${SLURM_JOB_ID}.log
echo "[$(date)] launching training; log -> $LOG"

$PY -u scripts/train_qlora.py \
    --out-dir "$OUT" \
    --mid-eval-log-path "$OUT/mid_eval.jsonl" \
    --train-jsonl "data/splits/train.jsonl" \
    --eval-jsonl  "data/splits/eval.jsonl" \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 16 \
    2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "[$(date)] training exit=$rc"
ls -la "$OUT" || true
exit $rc
