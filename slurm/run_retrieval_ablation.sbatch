#!/bin/bash
# 18-cell retrieval ablation on dev N=100. Loads Qwen2.5-Math-1.5B-Instruct
# once via vLLM, swaps retriever per cell, writes per-cell summary +
# top-level ablation_summary.json with the winner cell.
#
# Pre-flight (login node):
#   python scripts/check_tir_sandbox.py
#   python scripts/eval_tir.py --mode tir --smoke
#   python scripts/run_retrieval_ablation.py --smoke
#   sbatch slurm/eval_tir_vllm_smoke.sbatch         # validates the GPU/vLLM TIR path
#   sbatch slurm/build_tir_exemplar_bank.sbatch     # produces exemplar_bank.jsonl (TIR)
#   # also need a CoT exemplar bank: future TODO — for now the runner can use
#   # the TIR bank for both arms by passing --bank-cot results/tir/exemplar_bank.jsonl
#
# Submit:
#   sbatch slurm/run_retrieval_ablation.sbatch
#
# Wall-time budget: 4h. 18 cells × 100 problems × ~5-10s/gen on a hot vLLM
# engine ≈ 2.5-5h. Per-cell output is checkpointed; --skip-existing on
# resubmit lets a preempt resume from the last completed cell.
#
#SBATCH --job-name=tir-ablation
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --time=5:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] tir-ablation starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs results/tir/ablation

COMMIT="$(git rev-parse --short HEAD)"

# Required: cheap smoke + sandbox sentinel pass on this commit (matches
# eval_tir.sbatch's gating).
SENTINEL="logs/tir_smoke_passed_${COMMIT}.txt"
if [ ! -f "$SENTINEL" ]; then
    echo "[$(date)] FAIL: smoke sentinel $SENTINEL not found for commit $COMMIT"
    echo "Run on login node first: python scripts/eval_tir.py --mode tir_rag --smoke"
    exit 2
fi

# Required: vLLM smoke pass (so we know the bf16/CUDA path is healthy).
VLLM_SENTINEL="logs/tir_vllm_smoke_passed_${COMMIT}.txt"
if [ ! -f "$VLLM_SENTINEL" ]; then
    echo "[$(date)] FAIL: vLLM smoke sentinel $VLLM_SENTINEL not found for commit $COMMIT"
    echo "Run first: sbatch slurm/eval_tir_vllm_smoke.sbatch"
    exit 3
fi

# Required: TIR exemplar bank exists.
if [ ! -f results/tir/exemplar_bank.jsonl ]; then
    echo "[$(date)] FAIL: results/tir/exemplar_bank.jsonl not found."
    echo "Build first: sbatch slurm/build_tir_exemplar_bank.sbatch"
    exit 4
fi

# CoT exemplar bank: prefer a dedicated one; fall back to the TIR bank
# (which carries both code+output AND reasoning fields if built that way).
# A dedicated CoT bank is a future TODO; until then both arms read from
# the TIR bank and the formatter ignores the unused fields.
BANK_COT=results/tir/exemplar_bank_cot.jsonl
if [ ! -f "$BANK_COT" ]; then
    echo "[$(date)] note: $BANK_COT not found; using TIR bank for both arms."
    BANK_COT=results/tir/exemplar_bank.jsonl
fi

LOG="logs/run_retrieval_ablation_${SLURM_JOB_ID}.log"
echo "[$(date)] log -> $LOG"

$PY -u scripts/run_retrieval_ablation.py \
    --dev-jsonl data/splits/dev_100.jsonl \
    --bank-tir results/tir/exemplar_bank.jsonl \
    --bank-cot "$BANK_COT" \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --backend vllm \
    --max-new-tokens 4096 \
    --max-tool-calls 4 \
    --skip-existing \
    --out results/tir/ablation \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[$(date)] run_retrieval_ablation.py rc=$rc"
exit $rc
