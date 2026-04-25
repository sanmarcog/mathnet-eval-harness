#!/bin/bash
# Run B: single-variable ablation isolating completion_only_loss.
# Identical to Run 1 in every other dimension (Qwen2.5-1.5B-Instruct,
# English-only train, LoRA r=16 / alpha=32, max_seq_length=2048,
# per-device-batch=2 grad-accum=8, 2 epochs, lr=2e-4, seed=0). The only
# change vs Run 1 is --completion-only-loss, which masks system+user
# tokens so the loss only updates against the assistant completion.
#
# Hypothesis: Run 1's regression vs base (10% vs 14% paired n=50) is
# explained by training on the full prompt instead of just the answer.
# If B clears base meaningfully, completion-only-loss is the load-bearing
# fix and the kitchen-sink Run 2 is justified.
#
# Pipeline:
#   1. train  -> adapters/qwen25-mathnet-run-b/
#   2. merge  -> adapters/qwen25-mathnet-run-b-merged/  (bf16 standalone)
#   3. eval   -> results/full/qwen25-1.5b-run-b-n150/   (vLLM, n=150)
#
#SBATCH --job-name=qlora-run-b
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=a40:1
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] qlora-run-b starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
ADAPTERS_ROOT=/gscratch/scrubbed/sanmarco/adapters
OUT=$ADAPTERS_ROOT/qwen25-mathnet-run-b
MERGED=${OUT}-merged
EVAL_OUT=$REPO/results/full/qwen25-1.5b-run-b-n150

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs "$ADAPTERS_ROOT" "$EVAL_OUT"

LOG=logs/qlora_run_b_${SLURM_JOB_ID}.log

# === Stage 0: verify response template aligns with this model + dataset ===
# Pre-flight check: completion_only_loss silently does nothing if the
# template is not a contiguous token subsequence.
echo "[$(date)] stage 0: verify response template alignment"
$PY -u scripts/verify_response_template.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --train-jsonl data/splits/train.jsonl \
    --n 16 \
    2>&1 | tee -a "$LOG"
rc_verify=${PIPESTATUS[0]}
echo "[$(date)] verify rc=$rc_verify"
[ "$rc_verify" -eq 0 ] || { echo "ABORTING: response template misaligned"; exit $rc_verify; }

# === Stage 1: train ===
echo "[$(date)] stage 1: train (Run 1 config + completion-only-loss)"
$PY -u scripts/train_qlora.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --out-dir "$OUT" \
    --train-jsonl data/splits/train.jsonl \
    --eval-jsonl  data/splits/eval.jsonl \
    --mid-eval-log-path "$OUT/mid_eval.jsonl" \
    --completion-only-loss \
    2>&1 | tee "$LOG"
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

# === Stage 3: eval merged model on same n=150 IDs as Run 1 ===
echo "[$(date)] stage 3: eval merged model via vLLM, n=150 (paired with Run 1)"
$PY -u scripts/eval_qwen_hf.py \
    --base-model "$MERGED" \
    --split data/splits/eval.jsonl \
    --out "$EVAL_OUT" \
    --model-alias qwen25-1.5b-run-b-n150 \
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
