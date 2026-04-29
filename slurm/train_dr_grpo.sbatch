#!/bin/bash
# Dr. GRPO full training: ~200 steps on Qwen3-1.7B with bias-corrected
# GRPO (Liu et al., 2503.20783). Direct test of the Run-4 hypothesis
# that the SFT failure mode (length-amplification) is fixed by the
# bias-corrected RL variant.
#
# Submit AFTER calibrate_dr_grpo.sbatch confirms 48GB fit:
#   sbatch slurm/train_dr_grpo.sbatch
#
# Chain eval automatically:
#   tid=$(sbatch --parsable slurm/train_dr_grpo.sbatch)
#   sbatch --dependency=afterok:$tid slurm/eval_dr_grpo.sbatch
#
# Wall-time budget: 36h. Requeue-able. If preempted, GRPOTrainer
# resumes from the last checkpoint at save-steps boundary.
#
#SBATCH --job-name=drgrpo-train
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] drgrpo-train starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
ADAPTERS_ROOT=/gscratch/scrubbed/sanmarco/adapters
OUT=$ADAPTERS_ROOT/qwen3-mathnet-drgrpo
MERGED=${OUT}-merged

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

# Distributed env vars required even for single-GPU when vLLM colocate
# is enabled in TRL's GRPOTrainer (it instantiates vLLM's uniproc executor
# which reads RANK / WORLD_SIZE from env).
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

cd "$REPO"
mkdir -p logs "$ADAPTERS_ROOT"

LOG=logs/dr_grpo_train_${SLURM_JOB_ID}.log
echo "[$(date)] log -> $LOG"

# === Stage 1: Dr. GRPO training ===
# 200 steps × group size 4 × eff batch 4 = 3200 rollouts seen.
# At 3K-token avg completion, that's ~10M tokens of generation.
echo "[$(date)] stage 1: Dr. GRPO training"
$PY -u scripts/train_dr_grpo.py \
    --base-model Qwen/Qwen3-1.7B \
    --out-dir "$OUT" \
    --max-steps 200 \
    --num-generations 4 \
    --max-completion-length 3000 \
    --learning-rate 1e-6 \
    --beta 0.0 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --lora-r 64 \
    --lora-alpha 128 \
    --logging-steps 1 \
    --save-steps 25 \
    --seed 0 \
    2>&1 | tee -a "$LOG"
rc_train=${PIPESTATUS[0]}
echo "[$(date)] train rc=$rc_train"
[ "$rc_train" -eq 0 ] || exit $rc_train

# === Stage 2: merge LoRA into bf16 standalone for fast vLLM eval ===
echo "[$(date)] stage 2: merge"
$PY -u scripts/merge_adapter.py \
    --base-model Qwen/Qwen3-1.7B \
    --adapter-dir "$OUT" \
    --out-dir "$MERGED" \
    2>&1 | tee -a "$LOG"
rc_m=${PIPESTATUS[0]}
ls -la "$MERGED" || true

echo "[$(date)] Dr. GRPO train+merge complete. Eval will fire if chained."
exit $rc_m
