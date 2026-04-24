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
#SBATCH --time=4:00:00
#
# If OOM on the 2080 Ti (11GB VRAM): drop per_device_train_batch_size to 1
# and bump gradient_accumulation_steps to 16 to keep effective batch = 16.
# (Pass via --per-device-train-batch-size 1 --gradient-accumulation-steps 16
# on the train_qlora.py line below.)
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
    2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "[$(date)] training exit=$rc"
ls -la "$OUT" || true
exit $rc
