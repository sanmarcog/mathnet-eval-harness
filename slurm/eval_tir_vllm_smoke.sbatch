#!/bin/bash
# vLLM smoke for the Week-2 TIR eval.
#
# Per the Dr. GRPO lesson (CPU-only smoke missed the bf16/CUDA path),
# we validate the vLLM TIR path with a 5-problem dry run on a real GPU
# BEFORE committing to the full 18-cell ablation. ~5 min wall time on L40s.
#
# Smoke-only: TIR mode (not tir_rag), no judge, BM25 retriever unused.
# Confirms:
#   * vLLM loads Qwen2.5-Math-1.5B-Instruct in bf16 on L40s
#   * TIRRunner emits at least one tool call → sandbox → continuation cycle
#   * \boxed answer is parseable from the trace
#
# Submit:
#   sbatch slurm/eval_tir_vllm_smoke.sbatch
#
# Pass criterion (eyeball the log):
#   - >=1 problem with n_tool_calls > 0
#   - >=1 problem with finished=True (saw a \boxed answer)
#   - no Python tracebacks
#
#SBATCH --job-name=tir-vllm-smoke
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --time=0:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] tir-vllm-smoke starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs results/tir/vllm_smoke

LOG="logs/tir_vllm_smoke_${SLURM_JOB_ID}.log"
echo "[$(date)] log -> $LOG"

# 5 problems, TIR mode, vLLM on real model. Use eval.jsonl as the source
# (smoke just needs problems to feed; doesn't matter which split).
$PY -u scripts/eval_tir.py \
    --mode tir \
    --eval-jsonl data/splits/eval.jsonl \
    --n 5 \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --backend vllm \
    --max-new-tokens 2048 \
    --max-tool-calls 4 \
    --out results/tir/vllm_smoke \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

# Pass-criteria check: parse the summary.json and assert at least one
# tool call AND at least one finished trace. Smoke must be loud on
# failure so we don't silently advance to the 18-cell ablation.
if [ "$rc" -eq 0 ]; then
    SUMMARY=results/tir/vllm_smoke/summary.json
    if [ ! -f "$SUMMARY" ]; then
        echo "[$(date)] FAIL: $SUMMARY missing"
        exit 4
    fi
    $PY - <<EOF
import json, sys
s = json.load(open("$SUMMARY"))
print("[smoke-summary]", json.dumps(s, ensure_ascii=False))
ok = s.get("n_tool_calls_total", 0) > 0 and s.get("n_scored", 0) > 0
sys.exit(0 if ok else 5)
EOF
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[$(date)] FAIL: vLLM smoke pass-criteria not met (rc=$rc)"
    else
        # Mark vllm-smoke pass per commit so the ablation runner can refuse
        # to launch if it hasn't been validated.
        COMMIT="$(git rev-parse --short HEAD)"
        SENTINEL="logs/tir_vllm_smoke_passed_${COMMIT}.txt"
        echo "vllm smoke passed at $(date) job=$SLURM_JOB_ID" > "$SENTINEL"
        echo "[$(date)] wrote sentinel: $SENTINEL"
    fi
fi

echo "[$(date)] tir-vllm-smoke rc=$rc"
exit $rc
