#!/bin/bash
# Eval Run 3's merged model on the full 500-problem eval split. Same
# config as eval_qwen3_base.sbatch / eval_qwen3_run2.sbatch
# (thinking-on, 16K, vLLM) so the result is directly paired with the
# Qwen3 base 36.8% headline.
#
# Submit AFTER slurm/train_qlora_run3.sbatch produces the merged model
# at /gscratch/scrubbed/sanmarco/adapters/qwen3-mathnet-run3-merged.
#
#SBATCH --job-name=qwen3-run3-eval
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
echo "[$(date)] qwen3-run3-eval starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
MERGED=/gscratch/scrubbed/sanmarco/adapters/qwen3-mathnet-run3-merged
EVAL_OUT=$REPO/results/full/qwen3-1.7b-run3

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs results/full

LOG=logs/qwen3_run3_eval_${SLURM_JOB_ID}.log

if [ ! -d "$MERGED" ]; then
    echo "ERROR: merged Run-3 model not found at $MERGED"
    echo "Run slurm/train_qlora_run3.sbatch first."
    exit 2
fi

$PY -u scripts/eval_qwen_hf.py \
    --base-model "$MERGED" \
    --split data/splits/eval.jsonl \
    --out "$EVAL_OUT" \
    --model-alias qwen3-1.7b-run3 \
    --backend vllm \
    --prompt-format boxed \
    --enable-thinking \
    --sampling recommended \
    --seed 0 \
    --max-new-tokens 16384 \
    --skip-existing \
    2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "[$(date)] eval rc=$rc"
exit $rc
