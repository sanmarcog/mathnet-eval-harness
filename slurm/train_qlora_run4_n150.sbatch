#!/bin/bash
# Run 4 with n=150 target: extend the partial pilot's 65 self-distill rows
# to ~150 correct examples by attempting another ~365 source problems.
#
# Reasoning: at 65 rows × eff batch 4 × 2 epochs = 32 opt steps the
# expected per-step effect (~0.12 pp/step from Run 2/3 empirics) puts
# us at ~3-4 pp absolute movement — borderline for our n=500 eval to
# distinguish from base. At 150 rows × 2 epochs = 75 steps, expected
# ~8-10 pp movement, which our eval can detect at >95% power.
#
# Costs ~2.5 h additional base inference (~365 attempts at ~25 s each
# via vLLM thinking-on 16K) on top of the lite training time. Fits
# Monday afternoon ship window.
#
# Pipeline:
#   1. extend self-distill data to n_attempt=430 (skip-existing reuses 65)
#   2. verify response template alignment
#   3. train (LR 1e-5, eff batch 4, 2 epochs, max_seq 8192)
#   4. merge -> bf16 standalone
# (eval is in slurm/eval_qwen3_run4.sbatch — chain via afterok)
#
#SBATCH --job-name=qlora-run4-n150
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --time=5:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] qlora-run4-n150 starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
ADAPTERS_ROOT=/gscratch/scrubbed/sanmarco/adapters
OUT=$ADAPTERS_ROOT/qwen3-mathnet-run4
MERGED=${OUT}-merged
DISTILL_DATA=$REPO/data/splits/train_self_distill.jsonl

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs "$ADAPTERS_ROOT"

LOG=logs/qlora_run4_n150_${SLURM_JOB_ID}.log
echo "[$(date)] qlora-run4-n150 log -> $LOG"

# === Stage 1: extend self-distill data ===
# Target: 430 source rows attempted -> ~150 correct kept (37% base rate).
# --skip-existing reuses the pilot's 65 already-correct rows; remaining
# 365 attempts are a mix of retries (~110 failed before) + new (~255 fresh).
echo "[$(date)] stage 1: self-distill data extension, n_attempt=430 (skip-existing)"
$PY -u scripts/build_self_distill_data.py \
    --source-jsonl data/splits/train_multilingual_filtered.jsonl \
    --base-model Qwen/Qwen3-1.7B \
    --out-jsonl "$DISTILL_DATA" \
    --n-attempt 430 \
    --max-new-tokens 16384 \
    --seed 0 \
    --skip-existing \
    2>&1 | tee -a "$LOG"
rc_gen=${PIPESTATUS[0]}
echo "[$(date)] data gen rc=$rc_gen"
[ "$rc_gen" -eq 0 ] || exit $rc_gen

n_kept=$(wc -l < "$DISTILL_DATA")
echo "[$(date)] $n_kept correct examples in $DISTILL_DATA"
if [ "$n_kept" -lt 100 ]; then
    echo "ERROR: too few self-distill rows ($n_kept < 100); training would be undersized."
    exit 3
fi

# === Stage 2: verify response template alignment ===
echo "[$(date)] stage 2: verify response template"
$PY -u scripts/verify_response_template.py \
    --base-model Qwen/Qwen3-1.7B \
    --train-jsonl "$DISTILL_DATA" \
    --n 8 \
    2>&1 | tee -a "$LOG"
rc_v=${PIPESTATUS[0]}
echo "[$(date)] verify rc=$rc_v"
[ "$rc_v" -eq 0 ] || exit $rc_v

# === Stage 3: train (LR 1e-5, eff batch 4, 2 epochs ~ 75 opt steps on 150 rows) ===
echo "[$(date)] stage 3: train"
$PY -u scripts/train_qlora.py \
    --base-model Qwen/Qwen3-1.7B \
    --out-dir "$OUT" \
    --train-jsonl "$DISTILL_DATA" \
    --eval-jsonl  data/splits/eval.jsonl \
    --mid-eval-log-path "$OUT/mid_eval.jsonl" \
    --lora-r 64 \
    --lora-alpha 128 \
    --max-seq-length 8192 \
    --num-train-epochs 2 \
    --learning-rate 1e-5 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --completion-only-loss \
    --enable-thinking \
    --mid-eval-n 0 \
    2>&1 | tee -a "$LOG"
rc_train=${PIPESTATUS[0]}
echo "[$(date)] train rc=$rc_train"
[ "$rc_train" -eq 0 ] || exit $rc_train

# === Stage 4: merge ===
echo "[$(date)] stage 4: merge"
$PY -u scripts/merge_adapter.py \
    --base-model Qwen/Qwen3-1.7B \
    --adapter-dir "$OUT" \
    --out-dir "$MERGED" \
    2>&1 | tee -a "$LOG"
rc_m=${PIPESTATUS[0]}
ls -la "$MERGED" || true

echo "[$(date)] Run 4 (n150) train+merge complete. Eval will fire if chained."
exit $rc_m
