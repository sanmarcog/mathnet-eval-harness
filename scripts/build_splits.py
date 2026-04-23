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

    print("\n>>> filter funnel")
    filtered = apply_week1_filters(df, verbose=True)
    if len(filtered) < args.eval_size + 1:
        raise SystemExit(
            f"After filtering only {len(filtered)} rows remain, but eval_size={args.eval_size}. "
            "Consider widening the scope (e.g. include bilingual rows)."
        )

    print(f"\n>>> stratified split by '{args.strata_col}' (seed={args.seed})")
    eval_df, train_df = stratified_split(
        filtered,
        eval_size=args.eval_size,
        train_size=args.train_size,
        strata_col=args.strata_col,
        seed=args.seed,
    )
    print(f"    eval:  {len(eval_df)} rows across {eval_df[args.strata_col].nunique()} strata")
    print(f"    train: {len(train_df)} rows across {train_df[args.strata_col].nunique()} strata")

    args.out.mkdir(parents=True, exist_ok=True)
    to_jsonl(eval_df, args.out / "eval.jsonl")
    to_jsonl(train_df, args.out / "train.jsonl")

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
