#!/bin/bash
# Budget-forcing eval of the Run-2 merged model. s1-style "Wait" loop
# adapted for Qwen3's </think> stop token. Pair with the single-sample
# Run-2 result + BoN result for the test-time-scaling table.
#
# Recommended: do the manual pre-check first by running with --n 5 and
# inspecting the bf.* fields in the per-problem JSONs to confirm Qwen3
# picks up the "Wait" cue (i.e. continues reasoning rather than echoing
# a final answer or repeating itself).
#
#SBATCH --job-name=q3-run2-bf
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
echo "[$(date)] q3-run2-bf starting on $(hostname)  JOB=$SLURM_JOB_ID"
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
    exit 2
fi

LOG=logs/q3_run2_bf_${SLURM_JOB_ID}.log
N_LIMIT=${BF_N_LIMIT:-}      # set to 5 for pre-check, unset for full 500
NUM_IGNORE=${BF_NUM_IGNORE:-1}
MAX_THINKING=${BF_MAX_THINKING:-16000}

ARGS=(
    --base-model "$MERGED"
    --split data/splits/eval.jsonl
    --out results/full/qwen3-1.7b-run2-bf
    --model-alias qwen3-1.7b-run2-bf
    --num-ignore $NUM_IGNORE
    --max-tokens-thinking $MAX_THINKING
    --max-tokens-answer 1024
    --prompt-format boxed
    --seed 0
    --skip-existing
)
if [ -n "$N_LIMIT" ]; then
    ARGS+=( --n "$N_LIMIT" )
fi

$PY -u scripts/eval_budget_forcing.py "${ARGS[@]}" 2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "[$(date)] q3-run2-bf rc=$rc"
exit $rc
