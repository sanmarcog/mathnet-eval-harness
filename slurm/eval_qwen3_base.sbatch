#!/bin/bash
# Run base Qwen3-1.7B on the full 500-problem MathNet eval with the model's
# math-recommended setup: thinking mode ON, \boxed{} prompt, sampling
# (temp=0.6, top_p=0.95, top_k=20, min_p=0) with a fixed seed for
# reproducibility. Thinking-mode generations are much longer, so wall is
# allocated accordingly.
#
#SBATCH --job-name=qwen3-base-eval
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
echo "[$(date)] qwen3-base-eval starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"

cd "$REPO"
mkdir -p logs results/full

LOG=logs/qwen3_base_${SLURM_JOB_ID}.log
echo "[$(date)] launching base Qwen3-1.7B eval; log -> $LOG"

# Run Qwen3 with thinking-on + Qwen3 math-recommended sampling via vLLM.
# vLLM's batched throughput makes thinking-mode eval tractable at our
# walltime (HF generate was ~30 tok/s and never converged within 2048
# tokens; vLLM targets 100+ tok/s with paged attention + batching).
# Thinking-on with 4096 max tokens captures enough reasoning for
# most olympiad problems; the fraction that still truncate is flagged
# in NOTES methodology.
$PY -u scripts/eval_qwen_hf.py \
    --base-model Qwen/Qwen3-1.7B \
    --split data/splits/eval.jsonl \
    --out results/full/qwen3-1.7b-base \
    --model-alias qwen3-1.7b-base \
    --backend vllm \
    --prompt-format boxed \
    --enable-thinking \
    --sampling recommended \
    --seed 0 \
    --max-new-tokens 4096 \
    --skip-existing \
    2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "[$(date)] qwen3-base-eval rc=$rc"
exit $rc
