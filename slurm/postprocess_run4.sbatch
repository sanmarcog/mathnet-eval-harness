#!/bin/bash
# Post-process Run 4: chain via afterok on the eval job. Fires the
# moment eval completes successfully and runs:
#   1. grade_results.py with judge (the actual LLM-API spend)
#   2. analyze_finetune_vs_base.py (paired vs base)
#   3. compare_configs.py (Run 2/3/4 vs base in one table)
#   4. make_figures.py (regenerates finetune_progression.png with Run 4)
# All four are CPU + API only — no GPU needed.
#
# Submit:
#     sbatch --dependency=afterok:34895213 slurm/postprocess_run4.sbatch
#
#SBATCH --job-name=run4-postprocess
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
cd "$REPO"

# Load ANTHROPIC_API_KEY for the judge step
[ -f .env ] && set -a && source .env && set +a
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1

LOG=logs/run4_postprocess_${SLURM_JOB_ID}.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] postprocess starting on $(hostname)"
echo "[$(date)] result dir contains $(ls results/full/qwen3-1.7b-run4/*.json 2>/dev/null | wc -l) files"

echo
echo '=== stage 1: grade Run 4 with judge ==='
$PY scripts/grade_results.py --dir results/full/qwen3-1.7b-run4 --use-judge
rc1=$?
echo "[$(date)] grade rc=$rc1"

echo
echo '=== stage 2: paired analysis vs Qwen3 base ==='
$PY scripts/analyze_finetune_vs_base.py \
    --finetune-dir results/full/qwen3-1.7b-run4 \
    --base-dir     results/full/qwen3-1.7b-base \
    --out          docs/run4_analysis.md
rc2=$?
echo "[$(date)] analyze rc=$rc2"

echo
echo '=== stage 3: multi-config comparison (base + Run 2/3/4) ==='
$PY scripts/compare_configs.py \
    --baseline results/full/qwen3-1.7b-base \
    --configs  results/full/qwen3-1.7b-run4 \
               results/full/qwen3-1.7b-run3 \
               results/full/qwen3-1.7b-run2 \
    --out      docs/run4_full_comparison.md
rc3=$?
echo "[$(date)] compare rc=$rc3"

echo
echo '=== stage 4: regenerate figures ==='
$PY scripts/make_figures.py
rc4=$?
echo "[$(date)] figures rc=$rc4"

echo
echo "[$(date)] postprocess complete (grade=$rc1 analyze=$rc2 compare=$rc3 figures=$rc4)"
echo "[$(date)] outputs ready:"
echo "  - docs/run4_analysis.md (paired Run 4 vs base)"
echo "  - docs/run4_full_comparison.md (4-way table)"
echo "  - results/figures/finetune_progression.png (updated)"
echo "  - results/full/qwen3-1.7b-run4/*.graded.json + summary.json"
exit 0
