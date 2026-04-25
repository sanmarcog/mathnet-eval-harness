#!/bin/bash
# Best-of-N eval of Qwen3-1.7B base on the full 500-problem eval split.
# Same model and prompt config as the existing slurm/eval_qwen3_base.sbatch
# (paired with the 36.8% single-sample baseline) but draws N samples per
# problem and majority-votes on the boxed answer.
#
# Submit AFTER the Run-2 train+merge+eval pipeline so we don't compete
# for the 1-concurrent-job slot.
#
#SBATCH --job-name=q3-base-bon
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
echo "[$(date)] q3-base-bon starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs results/full

LOG=logs/q3_base_bon_${SLURM_JOB_ID}.log
N=${BON_N:-8}

$PY -u scripts/eval_best_of_n.py \
    --base-model Qwen/Qwen3-1.7B \
    --split data/splits/eval.jsonl \
    --out results/full/qwen3-1.7b-base-bon${N} \
    --model-alias qwen3-1.7b-base-bon${N} \
    --n-samples $N \
    --prompt-format boxed \
    --enable-thinking \
    --max-new-tokens 16384 \
    --seed 0 \
    --skip-existing \
    2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "[$(date)] q3-base-bon rc=$rc"
exit $rc
