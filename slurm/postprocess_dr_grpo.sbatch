#!/bin/bash
# Post-process Dr. GRPO eval. Chained via afterok on eval. Stages:
#   1. grade with judge (LLM-API spend; minimal since most resolve cheap)
#   2. paired analysis vs Qwen3-1.7B base
#   3. paired analysis vs Run 4 (does Dr. GRPO recover the SFT regression?)
#   4. multi-config table: base + Run 2/3/4 + Dr. GRPO
#   5. regenerate diagnostic figures (now with Dr. GRPO included)
# All CPU + API only — no GPU needed.
#
# Submit:
#   sbatch --dependency=afterok:<eval_jobid> slurm/postprocess_dr_grpo.sbatch
#
#SBATCH --job-name=drgrpo-postprocess
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

[ -f .env ] && set -a && source .env && set +a
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1

LOG=logs/drgrpo_postprocess_${SLURM_JOB_ID}.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] postprocess starting on $(hostname)"
echo "[$(date)] result dir contains $(ls results/full/qwen3-1.7b-drgrpo/*.json 2>/dev/null | wc -l) files"

echo
echo '=== stage 1: grade Dr. GRPO with judge ==='
$PY scripts/grade_results.py --dir results/full/qwen3-1.7b-drgrpo --use-judge
rc1=$?
echo "[$(date)] grade rc=$rc1"

echo
echo '=== stage 2: paired analysis vs Qwen3 base ==='
$PY scripts/analyze_finetune_vs_base.py \
    --finetune-dir results/full/qwen3-1.7b-drgrpo \
    --base-dir     results/full/qwen3-1.7b-base \
    --out          docs/drgrpo_vs_base_analysis.md
rc2=$?
echo "[$(date)] vs-base rc=$rc2"

echo
echo '=== stage 3: paired analysis vs Run 4 ==='
$PY scripts/analyze_finetune_vs_base.py \
    --finetune-dir results/full/qwen3-1.7b-drgrpo \
    --base-dir     results/full/qwen3-1.7b-run4 \
    --out          docs/drgrpo_vs_run4_analysis.md
rc3=$?
echo "[$(date)] vs-run4 rc=$rc3"

echo
echo '=== stage 4: multi-config comparison (base + Run 2/3/4 + drgrpo) ==='
$PY scripts/compare_configs.py \
    --baseline results/full/qwen3-1.7b-base \
    --configs  results/full/qwen3-1.7b-drgrpo \
               results/full/qwen3-1.7b-run4 \
               results/full/qwen3-1.7b-run3 \
               results/full/qwen3-1.7b-run2 \
    --out      docs/drgrpo_full_comparison.md
rc4=$?
echo "[$(date)] compare rc=$rc4"

echo
echo '=== stage 5: regenerate diagnostic figures ==='
$PY scripts/make_diagnostic_figures.py
rc5=$?
echo "[$(date)] diag figures rc=$rc5"

echo
echo "[$(date)] postprocess complete (grade=$rc1 vs-base=$rc2 vs-run4=$rc3 compare=$rc4 figures=$rc5)"
echo "[$(date)] outputs ready:"
echo "  - docs/drgrpo_vs_base_analysis.md (paired vs base)"
echo "  - docs/drgrpo_vs_run4_analysis.md (paired vs Run 4)"
echo "  - docs/drgrpo_full_comparison.md (5-way table)"
echo "  - results/figures/miss_mode_decomposition.png (now includes Dr. GRPO bar)"
echo "  - results/full/qwen3-1.7b-drgrpo/*.graded.json + summary.json"
exit 0
