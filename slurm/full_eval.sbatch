#!/bin/bash
# Full frontier-model eval on Hyak Klone.
# CPU-only (the workload is API HTTP calls, not compute).
# Launches all 5 model evals as parallel background processes on the same
# compute node, then waits for them all. The run_eval.py harness writes
# each problem's JSON before the next call starts, and the inference cache
# makes restart-after-preemption nearly free.
#
# Submit:
#     sbatch slurm/full_eval.sbatch
# Monitor:
#     tail -f logs/gemini-3-pro.log logs/sonnet-4-6.log ...
#     squeue -u $USER
#
#SBATCH --job-name=mathnet-eval
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail

echo "[$(date)] sbatch job starting on $(hostname)"
echo "[$(date)] SLURM_JOB_ID=$SLURM_JOB_ID SLURM_NODELIST=$SLURM_NODELIST"

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"

cd "$REPO"
mkdir -p logs results/full

echo "[$(date)] running preflight ..."
if ! $PY scripts/preflight.py; then
    echo "[$(date)] preflight FAILED -- aborting before any paid API call" >&2
    exit 1
fi

echo "[$(date)] preflight OK -- launching 5 parallel eval processes"

# Each process is I/O-bound (API calls), so 5-on-4-CPUs is fine. Logs are
# per-model so we can tail them independently.
$PY -u scripts/run_eval.py --model sonnet-4-6   --split eval --n 500 --out results/full/sonnet-4-6   > logs/sonnet-4-6.log   2>&1 &
PID_SONNET=$!
$PY -u scripts/run_eval.py --model gpt-5.4      --split eval --n 500 --out results/full/gpt-5.4      > logs/gpt-5.4.log      2>&1 &
PID_GPT=$!
$PY -u scripts/run_eval.py --model gpt-5.4-mini --split eval --n 500 --out results/full/gpt-5.4-mini > logs/gpt-5.4-mini.log 2>&1 &
PID_GPT_MINI=$!
# Gemini capped at 300 (not 500) to fit budget under the $60 ceiling --
# Gemini thoughts tokens are the dominant cost driver and the 10-problem
# calibration showed mean 4,307 thoughts/problem despite a thinking_budget
# of 4,096. See results/smoke/gemini-3-pro-cap4096/ and NOTES for details.
$PY -u scripts/run_eval.py --model gemini-3-pro --split eval --n 300 --out results/full/gemini-3-pro > logs/gemini-3-pro.log 2>&1 &
PID_GEMINI=$!
$PY -u scripts/run_eval.py --model opus-4-7     --split eval --n 100 --out results/full/opus-4-7     > logs/opus-4-7.log     2>&1 &
PID_OPUS=$!

echo "[$(date)] PIDs: sonnet=$PID_SONNET gpt=$PID_GPT gpt-mini=$PID_GPT_MINI gemini=$PID_GEMINI opus=$PID_OPUS"

# Wait for each individually so we get per-process exit codes.
rc=0
for name_pid in "sonnet:$PID_SONNET" "gpt-5.4:$PID_GPT" "gpt-5.4-mini:$PID_GPT_MINI" "gemini:$PID_GEMINI" "opus:$PID_OPUS"; do
    name="${name_pid%:*}"
    pid="${name_pid#*:}"
    wait "$pid"
    ec=$?
    if [ "$ec" -ne 0 ]; then
        echo "[$(date)] $name exited $ec"
        rc=1
    else
        echo "[$(date)] $name done"
    fi
done

echo "[$(date)] sbatch job ending (overall rc=$rc)"
exit $rc
