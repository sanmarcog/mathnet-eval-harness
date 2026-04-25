#!/bin/bash
# Run D-LR: single-variable ablation isolating learning rate.
# Identical to Run B (Run 1 + completion_only_loss) in every dimension
# EXCEPT --learning-rate 5e-5 (was 2e-4). Hypothesis: 2e-4 is too
# aggressive on an instruction-tuned base and is causing catastrophic
# forgetting that shows up as -6 to -8pp regression vs base.
#
# per-device-batch / grad-accum bumped from 2/8 to 4/4 (same effective
# batch 16, better A40 utilization). This is a throughput-only change,
# the optimization math is unchanged.
#
# Pipeline:
#   1. verify response template
#   2. train  -> adapters/qwen25-mathnet-run-d-lr/
#   3. merge  -> adapters/qwen25-mathnet-run-d-lr-merged/
#   4. eval   -> results/full/qwen25-1.5b-run-d-lr-n150/  (vLLM, n=150)
#
#SBATCH --job-name=qlora-run-d-lr
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=a40:1
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] qlora-run-d-lr starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
ADAPTERS_ROOT=/gscratch/scrubbed/sanmarco/adapters
OUT=$ADAPTERS_ROOT/qwen25-mathnet-run-d-lr
MERGED=${OUT}-merged
EVAL_OUT=$REPO/results/full/qwen25-1.5b-run-d-lr-n150

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs "$ADAPTERS_ROOT" "$EVAL_OUT"

LOG=logs/qlora_run_d_lr_${SLURM_JOB_ID}.log

# === Stage 0: verify response template alignment ===
echo "[$(date)] stage 0: verify response template alignment"
$PY -u scripts/verify_response_template.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --train-jsonl data/splits/train.jsonl \
    --n 16 \
    2>&1 | tee -a "$LOG"
rc_verify=${PIPESTATUS[0]}
echo "[$(date)] verify rc=$rc_verify"
[ "$rc_verify" -eq 0 ] || { echo "ABORTING: response template misaligned"; exit $rc_verify; }

# === Stage 1: train (Run B config + lr=5e-5 + batch=4/accum=4) ===
echo "[$(date)] stage 1: train, lr=5e-5"
$PY -u scripts/train_qlora.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --out-dir "$OUT" \
    --train-jsonl data/splits/train.jsonl \
    --eval-jsonl  data/splits/eval.jsonl \
    --mid-eval-log-path "$OUT/mid_eval.jsonl" \
    --completion-only-loss \
    --learning-rate 5e-5 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 4 \
    2>&1 | tee -a "$LOG"
rc_train=${PIPESTATUS[0]}
echo "[$(date)] train rc=$rc_train"
[ "$rc_train" -eq 0 ] || exit $rc_train

# === Stage 2: merge ===
echo "[$(date)] stage 2: merge LoRA into bf16 base -> $MERGED"
$PY -u scripts/merge_adapter.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-dir "$OUT" \
    --out-dir "$MERGED" \
    2>&1 | tee -a "$LOG"
rc_merge=${PIPESTATUS[0]}
echo "[$(date)] merge rc=$rc_merge"
[ "$rc_merge" -eq 0 ] || exit $rc_merge

# === Stage 3: eval merged on same n=150 IDs as Run 1 / Run B ===
echo "[$(date)] stage 3: eval merged via vLLM, n=150"
$PY -u scripts/eval_qwen_hf.py \
    --base-model "$MERGED" \
    --split data/splits/eval.jsonl \
    --out "$EVAL_OUT" \
    --model-alias qwen25-1.5b-run-d-lr-n150 \
    --backend vllm \
    --precision bf16 \
    --prompt-format boxed \
    --sampling greedy \
    --seed 0 \
    --max-new-tokens 4096 \
    --n 150 \
    2>&1 | tee -a "$LOG"
rc_eval=${PIPESTATUS[0]}
echo "[$(date)] eval rc=$rc_eval"

ls -la "$OUT" "$MERGED" "$EVAL_OUT" || true
exit $rc_eval
