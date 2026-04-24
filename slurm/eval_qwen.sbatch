#!/bin/bash
# Run base Qwen-2.5-1.5B and fine-tuned Qwen (base + QLoRA adapter) on the
# full 500-problem MathNet eval split, sequentially in one GPU allocation.
# Local inference (no API spend). Results go under results/full/<model>/.
#
# Submit:
#     sbatch slurm/eval_qwen.sbatch
#
#SBATCH --job-name=qwen-eval
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] qwen-eval starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
ADAPTER=/gscratch/scrubbed/sanmarco/adapters/qwen-mathnet-run1

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"

cd "$REPO"
mkdir -p logs results/full

BASE_LOG=logs/qwen_base_${SLURM_JOB_ID}.log
ADAPTER_LOG=logs/qwen_adapter_${SLURM_JOB_ID}.log

echo "[$(date)] === BASE Qwen eval (500 problems) ==="
$PY -u scripts/eval_qwen_hf.py \
    --split data/splits/eval.jsonl \
    --out results/full/qwen-base \
    --model-alias qwen-2.5-1.5b-instruct \
    --skip-existing \
    2>&1 | tee "$BASE_LOG"
BASE_RC=${PIPESTATUS[0]}
echo "[$(date)] base eval rc=$BASE_RC"

echo "[$(date)] === ADAPTER Qwen eval (base + qwen-mathnet-run1 adapter) ==="
$PY -u scripts/eval_qwen_hf.py \
    --split data/splits/eval.jsonl \
    --out results/full/qwen-mathnet-run1 \
    --adapter "$ADAPTER" \
    --model-alias qwen-mathnet-run1 \
    --skip-existing \
    2>&1 | tee "$ADAPTER_LOG"
ADAPTER_RC=${PIPESTATUS[0]}
echo "[$(date)] adapter eval rc=$ADAPTER_RC"

rc=0
[ "$BASE_RC" -ne 0 ] && rc=1
[ "$ADAPTER_RC" -ne 0 ] && rc=1
echo "[$(date)] qwen-eval done (overall rc=$rc)"
exit $rc
