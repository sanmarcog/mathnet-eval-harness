"""Build the Week-1 eval + train splits from MathNet and save them as JSONL.

Usage (Hyak):
    export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
    /gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python -u scripts/build_splits.py \
        --eval-size 500 --train-size 5000 --out data/splits

Outputs:
    data/splits/eval.jsonl   ({eval_size} problems)
    data/splits/train.jsonl  ({train_size} problems, disjoint from eval)
    data/splits/stats.json   summary of the funnel + per-strata counts
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mathnet_eval.data import (
    EVAL_COLUMNS,
    TRAIN_COLUMNS,
    apply_week1_filters,
    load_mathnet,
    stratified_split,
    to_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-size", type=int, default=500)
    p.add_argument("--train-size", type=int, default=5000)
    p.add_argument("--out", type=Path, default=Path("data/splits"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--strata-col", default="competition",
        help="Column to stratify on. 'competition' is usually richer than 'country'.",
    )
    p.add_argument(
        "--multilingual", action="store_true",
        help="Drop the English-only filter in the TRAIN build (eval always stays English-only so it remains comparable to Day-2 frontier numbers). Produces train_multilingual.jsonl.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Load with a narrow column set; skip images entirely (they're unused
    # downstream once we've filtered text-only) by loading all columns EXCEPT
    # images' struct bytes. pyarrow doesn't support "all except" directly, so
    # we enumerate the lightweight columns we actually need.
    print(">>> loading MathNet (data/all/ partition only) ...")
    cols = [
        "id", "country", "competition", "language",
        "problem_markdown", "solutions_markdown", "topics_flat",
        "problem_type", "final_answer", "images",
    ]
    df = load_mathnet(columns=cols)
    print(f"    loaded {len(df)} rows, {df.shape[1]} cols")

    print("\n>>> ENGLISH-ONLY funnel (for eval split)")
    en_filtered = apply_week1_filters(df, verbose=True, english_only=True)
    if len(en_filtered) < args.eval_size + 1:
        raise SystemExit(
            f"After filtering only {len(en_filtered)} rows remain, but eval_size={args.eval_size}. "
            "Consider widening the scope."
        )

    print(f"\n>>> stratified split by '{args.strata_col}' (seed={args.seed})")
    eval_df, en_train_df = stratified_split(
        en_filtered,
        eval_size=args.eval_size,
        train_size=args.train_size,
        strata_col=args.strata_col,
        seed=args.seed,
    )
    print(f"    eval:  {len(eval_df)} rows across {eval_df[args.strata_col].nunique()} strata")
    print(f"    train (English-only): {len(en_train_df)} rows across {en_train_df[args.strata_col].nunique()} strata")

    args.out.mkdir(parents=True, exist_ok=True)
    to_jsonl(eval_df, args.out / "eval.jsonl", columns=EVAL_COLUMNS)
    to_jsonl(en_train_df, args.out / "train.jsonl", columns=TRAIN_COLUMNS)

    if args.multilingual:
        print("\n>>> MULTILINGUAL funnel (for training only)")
        ml_filtered = apply_week1_filters(df, verbose=True, english_only=False)
        # Drop the eval IDs so train + eval stay disjoint.
        eval_ids = set(eval_df["id"])
        ml_train_all = ml_filtered[~ml_filtered["id"].isin(eval_ids)].reset_index(drop=True)
        # Stratify the multilingual training pool the same way, but cap at train_size.
        _, ml_train_df = stratified_split(
            ml_train_all,
            eval_size=1,  # tiny eval carveout; we just want train_df back from the helper
            train_size=args.train_size,
            strata_col=args.strata_col,
            seed=args.seed,
        )
        print(f"    train (multilingual): {len(ml_train_df)} rows across {ml_train_df[args.strata_col].nunique()} strata")
        to_jsonl(ml_train_df, args.out / "train_multilingual.jsonl", columns=TRAIN_COLUMNS)

    stats = {
        "seed": args.seed,
        "strata_col": args.strata_col,
        "filter_funnel": {
            "total_loaded": len(df),
            "after_all_filters": len(filtered),
        },
        "eval_size": len(eval_df),
        "train_size": len(train_df),
        "eval_per_strata": dict(Counter(eval_df[args.strata_col]).most_common()),
        "train_per_strata": dict(Counter(train_df[args.strata_col]).most_common()),
        "problem_type_counts": {
            "eval":  dict(Counter(eval_df["problem_type"]).most_common()),
            "train": dict(Counter(train_df["problem_type"]).most_common()),
        },
        "columns_saved": EVAL_COLUMNS,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"\n>>> wrote {args.out}/eval.jsonl, train.jsonl, stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
