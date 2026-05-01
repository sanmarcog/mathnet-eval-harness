#!/bin/bash
# Build the TIR exemplar bank by running Qwen2.5-Math-1.5B-Instruct in TIR
# mode across the English train set, keeping rollouts where the model
# called Python AND got the answer right.
#
# Pre-flight: smoke MUST pass on the login node first.
#   python scripts/check_tir_sandbox.py
#   python scripts/eval_tir.py --mode tir --smoke
#   python scripts/build_tir_exemplar_bank.py --smoke   # fixture path
# Then submit:
#   sbatch slurm/build_tir_exemplar_bank.sbatch
#
# Output: results/tir/exemplar_bank.jsonl + .summary.json
#
#SBATCH --job-name=build-tir-bank
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
echo "[$(date)] build-tir-bank starting on $(hostname)  JOB=$SLURM_JOB_ID"
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
    echo "Run on login node first: python scripts/eval_tir.py --mode tir --smoke"
    exit 2
fi
echo "[$(date)] smoke sentinel found: $SENTINEL"

# Bank-build also requires the real-rollout smoke (--smoke-real) — exercises
# the full TIRRunner with an actual model, not just the canned canary path
# the cheap smoke uses. Per the 2026-04-30 deviation ruling 3.
SMOKE_REAL="tests/.smoke_real_tir_passed_${COMMIT}"
if [ ! -f "$SMOKE_REAL" ]; then
    echo "[$(date)] FAIL: smoke-real-tir sentinel $SMOKE_REAL not found for commit $COMMIT"
    echo "Run on login node first:"
    echo "  python scripts/build_tir_exemplar_bank.py --smoke-real --backend hf --device cpu --model Qwen/Qwen2.5-0.5B-Instruct --n 8 --out /tmp/bank_smoke.jsonl"
    exit 3
fi
echo "[$(date)] smoke-real-tir sentinel found: $SMOKE_REAL"

LOG="logs/build_tir_bank_${SLURM_JOB_ID}.log"
echo "[$(date)] log -> $LOG"

$PY -u scripts/build_tir_exemplar_bank.py \
    --train-jsonl data/splits/train.jsonl \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --backend vllm \
    --max-new-tokens 4096 \
    --max-tool-calls 4 \
    --out results/tir/exemplar_bank.jsonl \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[$(date)] build_tir_exemplar_bank.py rc=$rc"
exit $rc
