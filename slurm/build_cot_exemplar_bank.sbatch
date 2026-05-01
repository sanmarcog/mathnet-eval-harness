#!/bin/bash
# Build the CoT exemplar bank by running Qwen2.5-Math-1.5B-Instruct in CoT
# mode (no Python sandbox) across the English train set, keeping rollouts
# where the model produced both reasoning AND a correct boxed answer.
#
# Mirror of build_tir_exemplar_bank.sbatch, but the bank's exemplars carry
# `reasoning` (free-text) instead of `code`+`output`. The two banks
# together let the 18-cell ablation cleanly answer "do tool-using rollouts
# as exemplars beat CoT-only rollouts as exemplars at 1.5B?" (the headline
# secondary question).
#
# Pre-flight: smoke + smoke-real-cot MUST pass on the login node first.
#   python scripts/check_tir_sandbox.py
#   python scripts/eval_tir.py --mode cot --smoke
#   python scripts/build_cot_exemplar_bank.py --smoke-real --backend hf \
#       --device cpu --model Qwen/Qwen2.5-0.5B-Instruct --n 8 \
#       --out /tmp/bank_cot_smoke.jsonl
# Then submit:
#   sbatch slurm/build_cot_exemplar_bank.sbatch
#
# Output: results/tir/exemplar_bank_cot.jsonl + .summary.json
#
#SBATCH --job-name=build-cot-bank
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] build-cot-bank starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs results/tir

COMMIT="$(git rev-parse --short HEAD)"
SENTINEL="logs/tir_smoke_passed_${COMMIT}.txt"
if [ ! -f "$SENTINEL" ]; then
    echo "[$(date)] FAIL: smoke sentinel $SENTINEL not found for commit $COMMIT"
    echo "Run on login node first: python scripts/eval_tir.py --mode cot --smoke"
    exit 2
fi

SMOKE_REAL="tests/.smoke_real_cot_passed_${COMMIT}"
if [ ! -f "$SMOKE_REAL" ]; then
    echo "[$(date)] FAIL: smoke-real-cot sentinel $SMOKE_REAL not found for commit $COMMIT"
    echo "Run on login node first:"
    echo "  python scripts/build_cot_exemplar_bank.py --smoke-real --backend hf --device cpu --model Qwen/Qwen2.5-0.5B-Instruct --n 8 --out /tmp/bank_cot_smoke.jsonl"
    exit 3
fi
echo "[$(date)] smoke-real-cot sentinel found: $SMOKE_REAL"

LOG="logs/build_cot_bank_${SLURM_JOB_ID}.log"
echo "[$(date)] log -> $LOG"

$PY -u scripts/build_cot_exemplar_bank.py \
    --train-jsonl data/splits/train.jsonl \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --backend vllm \
    --max-new-tokens 4096 \
    --out results/tir/exemplar_bank_cot.jsonl \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[$(date)] build_cot_exemplar_bank.py rc=$rc"
exit $rc
