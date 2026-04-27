#!/bin/bash
# Pilot for Run 4 (self-distillation). Goal: get a statistically meaningful
# directional signal in ~3h before committing to a full overnight run.
#
# Pre-registered decision rule (paired vs Qwen3 base 36.8% on same 100 IDs):
#   pilot accuracy >= 30%   ->  proceed to full Run 4
#   pilot accuracy 15-29%   ->  ambiguous, discuss
#   pilot accuracy <= 14%   ->  stop, write up negative finding
#
# Pipeline:
#   0. verify response template alignment on the (still-to-be-built) self-distill data
#      -- skipped at pilot time; data verification happens after gen
#   1. base inference on 200 train problems (~50min via vLLM, thinking-on, 16K)
#   2. filter for base-correct -> ~75 expected at base's 37% rate
#   3. train (LoRA r=64 alpha=128, LR 1e-5, eff batch 4, 2 epochs, max_seq 8192,
#      enable_thinking=True, completion_only_loss; ~38 opt steps)
#   4. merge -> bf16 standalone
#   5. eval n=100 (first 100 IDs of eval split) via vLLM, thinking-on, 16K
#
#SBATCH --job-name=pilot-run4
#SBATCH --account=demo
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -uo pipefail
echo "[$(date)] pilot-run4 starting on $(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi | head -25 || true

REPO=/gscratch/scrubbed/sanmarco/mathnet-eval-harness
PY=/gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python
ADAPTERS_ROOT=/gscratch/scrubbed/sanmarco/adapters
OUT=$ADAPTERS_ROOT/qwen3-mathnet-pilot4
MERGED=${OUT}-merged
EVAL_OUT=$REPO/results/full/qwen3-1.7b-pilot4-n100

# Self-distill data path: shared across pilot and full Run 4 via --skip-existing.
DISTILL_DATA=$REPO/data/splits/train_self_distill.jsonl

export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
export PYTHONPATH="$REPO/src"
export DO_NOT_TRACK=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
mkdir -p logs "$ADAPTERS_ROOT" "$EVAL_OUT"

LOG=logs/pilot_run4_${SLURM_JOB_ID}.log
echo "[$(date)] pilot-run4 log -> $LOG"

# === Stage 1: build self-distill data on first 200 train problems ===
echo "[$(date)] stage 1: self-distill data gen, n_attempt=200"
$PY -u scripts/build_self_distill_data.py \
    --source-jsonl data/splits/train_multilingual_filtered.jsonl \
    --base-model Qwen/Qwen3-1.7B \
    --out-jsonl "$DISTILL_DATA" \
    --n-attempt 200 \
    --max-new-tokens 16384 \
    --seed 0 \
    --skip-existing \
    2>&1 | tee -a "$LOG"
rc_gen=${PIPESTATUS[0]}
echo "[$(date)] data gen rc=$rc_gen"
[ "$rc_gen" -eq 0 ] || exit $rc_gen

n_kept=$(wc -l < "$DISTILL_DATA")
echo "[$(date)] $n_kept correct examples in $DISTILL_DATA"
if [ "$n_kept" -lt 30 ]; then
    echo "ERROR: too few self-distill rows ($n_kept < 30); training would be noise."
    exit 3
fi

# === Stage 2: verify response template alignment on the self-distill data ===
echo "[$(date)] stage 2: verify response template"
$PY -u scripts/verify_response_template.py \
    --base-model Qwen/Qwen3-1.7B \
    --train-jsonl "$DISTILL_DATA" \
    --n 8 \
    2>&1 | tee -a "$LOG"
rc_v=${PIPESTATUS[0]}
echo "[$(date)] verify rc=$rc_v"
[ "$rc_v" -eq 0 ] || exit $rc_v

# === Stage 3: train (small data; eff batch 4, 2 epochs) ===
echo "[$(date)] stage 3: train (LR=1e-5, eff batch 4, 2 epochs, max_seq 8192)"
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
echo "[$(date)] merge rc=$rc_m"
[ "$rc_m" -eq 0 ] || exit $rc_m

# === Stage 5: eval first n=100 IDs of eval split ===
echo "[$(date)] stage 5: eval merged, n=100, paired vs base on same 100 IDs"
$PY -u scripts/eval_qwen_hf.py \
    --base-model "$MERGED" \
    --split data/splits/eval.jsonl \
    --out "$EVAL_OUT" \
    --model-alias qwen3-1.7b-pilot4-n100 \
    --backend vllm \
    --prompt-format boxed \
    --enable-thinking \
    --sampling recommended \
    --seed 0 \
    --max-new-tokens 16384 \
    --n 100 \
    --skip-existing \
    2>&1 | tee -a "$LOG"
rc_eval=${PIPESTATUS[0]}
echo "[$(date)] eval rc=$rc_eval"

ls -la "$EVAL_OUT" || true
echo "[$(date)] PILOT DONE."
echo "[$(date)] Next: PYTHONPATH=src $PY scripts/grade_results.py --dir $EVAL_OUT --use-judge"
echo "[$(date)] Then: PYTHONPATH=src $PY scripts/analyze_finetune_vs_base.py --finetune-dir $EVAL_OUT --base-dir results/full/qwen3-1.7b-base --out docs/pilot4_analysis.md"
exit $rc_eval
