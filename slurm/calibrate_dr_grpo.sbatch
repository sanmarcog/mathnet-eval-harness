#!/bin/bash
# Dr. GRPO 48GB calibration: 5 steps on a small subset, log throughput,
# confirm OOM-free. Run BEFORE the full training sbatch.
#
# Goal: measure tokens/sec and per-step wall time so we can extrapolate
# the full-run wall time and lock in num-generations / max-completion-
# length / batch settings before committing 36 hours of compute.
#
# Outputs:
#   - logs/dr_grpo_calib_<jobid>.log (full stdout)
#   - /tmp/dr_grpo_calib (throwaway adapter)
#   - nvidia-smi snapshots before/during/after
#
# If this OOMs at 48GB, drop --num-generations to 2 (last-resort) or
# request a 2x48GB or single 80GB salloc instead.
#
#SBATCH --job-name=drgrpo-calib
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail
echo "[$(date)] drgrpo-calib starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python

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
# Pick a port derived from the job ID to avoid collisions with other
# jobs on shared nodes (29500 is the common default and frequently busy).
export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))

# Reduce CUDA memory fragmentation (PyTorch suggests this when OOM hits)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p logs

LOG=logs/dr_grpo_calib_${SLURM_JOB_ID}.log
echo "[$(date)] log -> $LOG"

# 5 steps × group size 4 × ~50 prompts = quick sanity sweep
$PY -u scripts/train_dr_grpo.py \
    --base-model Qwen/Qwen3-1.7B \
    --out-dir /tmp/dr_grpo_calib \
    --max-rows 50 \
    --max-steps 5 \
    --num-generations 2 \
    --max-completion-length 12000 \
    --gradient-accumulation-steps 4 \
    --logging-steps 1 \
    --save-steps 100 \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[$(date)] calib rc=$rc"

# Throughput summary at the end so we can grep one line.
echo
echo "=== POST-CALIB GPU STATE ==="
nvidia-smi | head -25 || true
echo
echo "=== TIMING (grep TRL log lines for timing/throughput) ==="
grep -E "step|loss|reward|tokens" "$LOG" | tail -40 || true

echo
if [ "$rc" -eq 0 ]; then
    echo "[$(date)] CALIB OK — extrapolate from per-step timing in the log,"
    echo "         then submit slurm/train_dr_grpo.sbatch."
else
    echo "[$(date)] CALIB FAILED rc=$rc — check log for OOM or config error."
    echo "         If OOM: try --num-generations 2 or move to 2x48GB or 1x80GB."
fi
exit $rc
