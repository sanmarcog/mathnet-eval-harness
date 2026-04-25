#!/bin/bash
# Best-of-N eval of the Run-2 merged model. Pair with the single-sample
# Run-2 result to measure how much test-time-scaling-via-sampling adds
# on top of the SFT lift.
#
# Submit AFTER slurm/train_qlora_run2.sbatch produces the merged model
# AND slurm/eval_qwen3_run2.sbatch produces the single-sample baseline.
#
#SBATCH --job-name=q3-run2-bon
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a40:1
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] q3-run2-bon starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
MERGED=/gscratch/scrubbed/sanmarco/adapters/qwen3-mathnet-run2-merged

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs results/full

if [ ! -d "$MERGED" ]; then
    echo "ERROR: merged Run-2 model not found at $MERGED"
    echo "Run slurm/train_qlora_run2.sbatch first."
    exit 2
fi

LOG=logs/q3_run2_bon_${SLURM_JOB_ID}.log
N=${BON_N:-8}

$PY -u scripts/eval_best_of_n.py \
    --base-model "$MERGED" \
    --split data/splits/eval.jsonl \
    --out results/full/qwen3-1.7b-run2-bon${N} \
    --model-alias qwen3-1.7b-run2-bon${N} \
    --n-samples $N \
    --prompt-format boxed \
    --enable-thinking \
    --max-new-tokens 16384 \
    --seed 0 \
    --skip-existing \
    2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "[$(date)] q3-run2-bon rc=$rc"
exit $rc
