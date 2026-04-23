"""Inspect the ShadenA/MathNet dataset — schema, size, strata distribution.

Run once when onboarding so we know the field layout before writing
splits / prompts / graders. Prints to stdout, no files written.

Usage (Hyak):
    module load conda && conda activate qlora
    HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache python scripts/inspect_mathnet.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

# Default HF cache to scratch so we don't fill the home-dir quota.
os.environ.setdefault("HF_HOME", "/gscratch/scrubbed/sanmarco/hf_cache")

from datasets import load_dataset  # noqa: E402


DATASET_ID = "ShadenA/MathNet"


def fmt(v, maxlen: int = 160) -> str:
    s = repr(v)
    return s if len(s) <= maxlen else s[:maxlen] + f"...<+{len(s) - maxlen} chars>"


def main() -> int:
    print(f"Loading {DATASET_ID} (config='all', split='train') ...", flush=True)
    ds = load_dataset(DATASET_ID, name="all", split="train")

    n = len(ds)
    print(f"\nTotal rows: {n:,}")
    print(f"Features: {ds.features}")
    print(f"Column names: {ds.column_names}")

    print("\n--- First 2 example rows (truncated) ---")
    for i in range(min(2, n)):
        ex = ds[i]
        print(f"\nRow {i}:")
        for k, v in ex.items():
            print(f"  {k}: {fmt(v)}")

    # Distribution across common stratification axes, if those columns exist.
    for col in ("competition", "country", "language", "year", "difficulty", "level"):
        if col in ds.column_names:
            c = Counter(ds[col])
            print(f"\n--- {col} distribution ({len(c)} unique) ---")
            for k, v in c.most_common(25):
                print(f"  {k}: {v}")
            if len(c) > 25:
                print(f"  ... (+{len(c) - 25} more)")

    # Image / modality signal: report which columns carry non-trivial binary content.
    print("\n--- Modality hints (first 100 rows) ---")
    sample = ds.select(range(min(100, n)))
    for col in ds.column_names:
        types = Counter(type(x).__name__ for x in sample[col])
        nulls = sum(1 for x in sample[col] if x is None)
        print(f"  {col}: types={dict(types)} nulls={nulls}/{len(sample)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
