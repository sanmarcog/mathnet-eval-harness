#!/bin/bash
# Validate the Run 1 QLoRA training pipeline end-to-end by evaluating the
# Qwen2.5-1.5B-Instruct + Run-1 LoRA adapter on the first 50 eval problems.
# vLLM does not load adapters in our current eval script, so we use the HF
# backend. This is slower but fine at n=50.
#
# Compare the resulting per-problem grading to the same 50 problems from the
# existing base-only run at results/full/qwen-base/. If the adapter does not
# move any correctness bits, the training pipeline is suspect.
#
#SBATCH --job-name=qwen25-run1-eval
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a40:1
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] qwen25-run1-eval starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
ADAPTER=/gscratch/scrubbed/sanmarco/adapters/qwen-mathnet-run1
export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs results/full

LOG=logs/qwen25_run1_${SLURM_JOB_ID}.log
echo "[$(date)] eval Run1 adapter; log -> $LOG"

$PY -u scripts/eval_qwen_hf.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter "$ADAPTER" \
    --split data/splits/eval.jsonl \
    --out results/full/qwen25-1.5b-run1-n50 \
    --model-alias qwen25-1.5b-run1-n50 \
    --backend hf \
    --precision bf16 \
    --prompt-format boxed \
    --sampling greedy \
    --seed 0 \
    --max-new-tokens 4096 \
    --n 50 \
    2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "[$(date)] eval rc=$rc"
exit $rc
