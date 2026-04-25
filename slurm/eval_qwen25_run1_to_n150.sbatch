#!/bin/bash
# Extend the Run 1 adapter validation eval from n=50 to n=150 to firm up
# the comparison vs base. The first 50 are already cached at
# results/full/qwen25-1.5b-run1-n50/; --skip-existing means we only run
# the additional 100 fresh problems and the summary.json gets re-aggregated
# at n=150.
#
#SBATCH --job-name=qwen25-run1-n150
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a40:1
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] qwen25-run1-n150 starting on $(hostname)  JOB=$SLURM_JOB_ID"
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
mkdir -p logs

LOG=logs/qwen25_run1_n150_${SLURM_JOB_ID}.log
echo "[$(date)] extend Run1 adapter eval to n=150; log -> $LOG"

$PY -u scripts/eval_qwen_hf.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter "$ADAPTER" \
    --split data/splits/eval.jsonl \
    --out results/full/qwen25-1.5b-run1-n50 \
    --model-alias qwen25-1.5b-run1-n150 \
    --backend hf \
    --precision bf16 \
    --prompt-format boxed \
    --sampling greedy \
    --seed 0 \
    --max-new-tokens 4096 \
    --n 150 \
    --skip-existing \
    2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "[$(date)] eval rc=$rc"
exit $rc
