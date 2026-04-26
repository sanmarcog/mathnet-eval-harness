#!/bin/bash
# Run 3: Recipe-match QLoRA fine-tune of Qwen3-1.7B with boxed-augmented
# training data.
#
# Run 2 caught a real bug: the original train_multilingual_filtered.jsonl
# had \boxed{} in only ~1.5% of rows, so 3 epochs at 2e-5 catastrophically
# unlearned the boxing convention. -33.8 pp paired delta on the eval
# (3.0% vs base 36.8%, p < 0.0001). Run 3 fixes that by appending
#   "Therefore, the final answer is $\boxed{<final_answer>}$"
# to each row's solution before training.
#
# Hyperparameters are unchanged from Run 2 (recipe-match Alibaba's
# Qwen2.5-Math 1.5B SFT: LR 2e-5, 3 epochs, eff batch 128, seq 4096).
# We're not retuning hyperparameters — we're fixing the data.
#
# Pipeline:
#   0a. verify boxed augmentation (round-trip extract_answer on N=16 rows)
#   0b. verify response template alignment
#   1.  train  -> adapters/qwen3-mathnet-run3/
#   2.  merge  -> adapters/qwen3-mathnet-run3-merged/
# (eval is in slurm/eval_qwen3_run3.sbatch — submit after this completes)
#
# PREREQUISITE: scripts/augment_train_with_boxed.py has been run to
# produce data/splits/train_multilingual_filtered_boxed.jsonl
#
#SBATCH --job-name=qlora-qwen3-run3
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
echo "[$(date)] qlora-qwen3-run3 starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
ADAPTERS_ROOT=/gscratch/scrubbed/sanmarco/adapters
OUT=$ADAPTERS_ROOT/qwen3-mathnet-run3
MERGED=${OUT}-merged
TRAIN_FILE=data/splits/train_multilingual_filtered_boxed.jsonl

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs "$ADAPTERS_ROOT"

LOG=logs/qlora_run3_${SLURM_JOB_ID}.log
echo "[$(date)] launching Run-3 (boxed augmentation); log -> $LOG"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: $TRAIN_FILE does not exist."
    echo "Run scripts/augment_train_with_boxed.py first."
    exit 2
fi

# === Stage 0a: verify boxed augmentation round-trips through chat template ===
echo "[$(date)] stage 0a: verify boxed augmentation"
$PY -u scripts/verify_boxed_augmentation.py \
    --train-jsonl "$TRAIN_FILE" \
    --base-model Qwen/Qwen3-1.7B \
    --n 16 \
    2>&1 | tee -a "$LOG"
rc_box=${PIPESTATUS[0]}
echo "[$(date)] verify_boxed rc=$rc_box"
[ "$rc_box" -eq 0 ] || { echo "ABORTING: boxed augmentation failed verification"; exit $rc_box; }

# === Stage 0b: verify response template alignment ===
echo "[$(date)] stage 0b: verify response template alignment"
$PY -u scripts/verify_response_template.py \
    --base-model Qwen/Qwen3-1.7B \
    --train-jsonl "$TRAIN_FILE" \
    --n 16 \
    2>&1 | tee -a "$LOG"
rc_tmpl=${PIPESTATUS[0]}
echo "[$(date)] verify_template rc=$rc_tmpl"
[ "$rc_tmpl" -eq 0 ] || { echo "ABORTING: response template misaligned"; exit $rc_tmpl; }

# === Stage 1: train (recipe-match: LR 2e-5, 3 epochs, eff batch 128) ===
echo "[$(date)] stage 1: train, lr=2e-5, 3 epochs, effective batch 128"
$PY -u scripts/train_qlora.py \
    --base-model Qwen/Qwen3-1.7B \
    --out-dir "$OUT" \
    --train-jsonl "$TRAIN_FILE" \
    --eval-jsonl  data/splits/eval.jsonl \
    --mid-eval-log-path "$OUT/mid_eval.jsonl" \
    --lora-r 64 \
    --lora-alpha 128 \
    --max-seq-length 4096 \
    --num-train-epochs 3 \
    --learning-rate 2e-5 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 32 \
    --completion-only-loss \
    --mid-eval-n 0 \
    2>&1 | tee -a "$LOG"
rc_train=${PIPESTATUS[0]}
echo "[$(date)] train rc=$rc_train"
[ "$rc_train" -eq 0 ] || exit $rc_train
ls -la "$OUT" || true

# === Stage 2: merge LoRA into bf16 base ===
echo "[$(date)] stage 2: merge LoRA adapter into base -> $MERGED"
$PY -u scripts/merge_adapter.py \
    --base-model Qwen/Qwen3-1.7B \
    --adapter-dir "$OUT" \
    --out-dir "$MERGED" \
    2>&1 | tee -a "$LOG"
rc_merge=${PIPESTATUS[0]}
echo "[$(date)] merge rc=$rc_merge"
[ "$rc_merge" -eq 0 ] || exit $rc_merge
ls -la "$MERGED" || true

echo "[$(date)] train+merge complete. Submit slurm/eval_qwen3_run3.sbatch next."
exit 0
