"""Build the dev N=100 split for the Week-2 TIR-RAG retrieval ablation.

Stratified-by-top-topic from `data/splits/train.jsonl` (the Week-1 English
train set, 3596 rows; what the pre-reg calls `train_english.jsonl`),
proportional to the eval-500 top-topic distribution. A problem may carry
multiple top-level topic prefixes, so each problem is assigned to the
*single rarest* top-level topic it carries — that gives the small
under-represented topics (e.g. Geometry at ~11%) a fair shot at hitting
their target count without being eaten by Algebra+X dual-tagged rows.

Excludes:
  * any IDs that appear in the exemplar bank, if --bank is passed.
    (Run this BEFORE building the bank — then have the bank-builder
    exclude these dev IDs in turn — or pass an existing bank to skip
    contaminated IDs.)

Usage:
    python scripts/build_dev_split.py \\
        --train data/splits/train.jsonl \\
        --eval data/splits/eval.jsonl \\
        --n 100 --seed 0 \\
        --out data/splits/dev_100.jsonl

Outputs:
    data/splits/dev_100.jsonl       — 100 rows, same schema as train.jsonl
    data/splits/dev_100.stats.json  — funnel + topic distribution
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


TOP_LEVEL_ORDER = ["Algebra", "Number Theory", "Discrete Mathematics", "Geometry"]


def top_level_topics(topics_flat) -> set[str]:
    """Return the set of top-level prefixes ('Algebra', 'Geometry', ...)
    a problem is tagged with. May be empty for un-tagged rows."""
    out: set[str] = set()
    if not topics_flat:
        return out
    for t in topics_flat:
        if not t:
            continue
        head = t.split(">")[0].strip()
        if head:
            out.add(head)
    return out


def assign_rarest_top(topics: set[str], freq: dict[str, int]) -> str | None:
    """When a problem carries multiple top-level topics, assign it to
    whichever one is rarest in the corpus. Avoids Algebra eating
    everything (since most multi-topic problems carry Algebra)."""
    if not topics:
        return None
    return min(topics, key=lambda t: freq.get(t, 0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default="data/splits/train.jsonl")
    p.add_argument("--eval", dest="eval_path", default="data/splits/eval.jsonl",
                   help="Used only to compute target proportions.")
    p.add_argument("--bank", default=None,
                   help="Optional exemplar bank JSONL. IDs in the bank are excluded.")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/splits/dev_100.jsonl")
    return p.parse_args()


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()

    train = load_jsonl(args.train)
    eval_rows = load_jsonl(args.eval_path)

    # ---- Exclusion sets ----
    exclude_ids: set[str] = set()
    eval_ids = {r["id"] for r in eval_rows}
    overlap_eval = sum(1 for r in train if r["id"] in eval_ids)
    if overlap_eval:
        # train.jsonl was built disjoint from eval.jsonl, so this should be 0.
        # Print + drop just in case.
        train = [r for r in train if r["id"] not in eval_ids]
    exclude_ids |= eval_ids

    bank_ids: set[str] = set()
    if args.bank and Path(args.bank).exists():
        for row in load_jsonl(args.bank):
            bank_ids.add(row["id"])
        train = [r for r in train if r["id"] not in bank_ids]
    exclude_ids |= bank_ids

    print(f"[funnel] loaded train={len(train)+len(bank_ids)+overlap_eval} rows")
    print(f"[funnel] dropped {overlap_eval} eval-overlap (sanity)")
    print(f"[funnel] dropped {len(bank_ids)} bank-id rows (if any)")
    print(f"[funnel] candidate pool: {len(train)} rows")

    # ---- Compute target counts from eval-500 top-topic distribution ----
    eval_topic_counts: Counter[str] = Counter()
    for r in eval_rows:
        for t in top_level_topics(r.get("topics_flat")):
            eval_topic_counts[t] += 1
    # Restrict to the four canonical roots to keep targets stable.
    canonical = {t: eval_topic_counts.get(t, 0) for t in TOP_LEVEL_ORDER}
    canonical_total = sum(canonical.values())
    targets: dict[str, int] = {}
    for t in TOP_LEVEL_ORDER:
        targets[t] = round(canonical[t] / canonical_total * args.n)
    # Fix off-by-one rounding so sum(targets) == n.
    drift = args.n - sum(targets.values())
    if drift != 0:
        # Apply drift to the largest bucket.
        largest = max(targets, key=lambda k: targets[k])
        targets[largest] += drift
    print(f"[targets] {targets}  (sum={sum(targets.values())}, requested n={args.n})")

    # ---- Bin the candidate pool by rarest top-topic ----
    train_freq: Counter[str] = Counter()
    for r in train:
        for t in top_level_topics(r.get("topics_flat")):
            train_freq[t] += 1

    bins: dict[str, list[dict]] = {t: [] for t in TOP_LEVEL_ORDER}
    untagged = 0
    for r in train:
        topics = top_level_topics(r.get("topics_flat")) & set(TOP_LEVEL_ORDER)
        if not topics:
            untagged += 1
            continue
        bucket = assign_rarest_top(topics, dict(train_freq))
        bins[bucket].append(r)
    print(f"[binning] untagged-or-other-topic rows excluded: {untagged}")
    for t in TOP_LEVEL_ORDER:
        print(f"[binning] {t:25s} pool={len(bins[t])}")

    # ---- Sample ----
    rng = random.Random(args.seed)
    selected: list[dict] = []
    for t in TOP_LEVEL_ORDER:
        pool = list(bins[t])
        rng.shuffle(pool)
        k = min(targets[t], len(pool))
        if k < targets[t]:
            print(f"[warn] {t}: requested {targets[t]} but pool has {len(pool)}")
        selected.extend(pool[:k])

    # If we're short due to a small bin, top up from the largest remaining
    # untouched pool, deterministically.
    short = args.n - len(selected)
    if short > 0:
        leftover: list[dict] = []
        chosen_ids = {r["id"] for r in selected}
        for t in TOP_LEVEL_ORDER:
            for r in bins[t]:
                if r["id"] not in chosen_ids:
                    leftover.append(r)
        rng.shuffle(leftover)
        selected.extend(leftover[:short])

    # Stable order for downstream reproducibility.
    selected.sort(key=lambda r: r["id"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[wrote] {out_path}  rows={len(selected)}")

    # ---- Stats / parity ----
    dev_topic_counts: Counter[str] = Counter()
    for r in selected:
        for t in top_level_topics(r.get("topics_flat")):
            dev_topic_counts[t] += 1
    print()
    print(f"[parity] eval-500 vs dev-{len(selected)} top-level topic % (multi-tagged):")
    print(f"  {'Topic':25s}  {'eval-500':>10s}  {'dev':>10s}")
    for t in TOP_LEVEL_ORDER:
        ev = eval_topic_counts.get(t, 0) / max(1, len(eval_rows)) * 100
        dv = dev_topic_counts.get(t, 0) / max(1, len(selected)) * 100
        print(f"  {t:25s}  {ev:>9.1f}%  {dv:>9.1f}%")

    stats = {
        "seed": args.seed,
        "n": len(selected),
        "candidate_pool_size": len(train),
        "bank_ids_excluded": len(bank_ids),
        "eval_overlap_dropped": overlap_eval,
        "targets": targets,
        "bin_pool_sizes": {t: len(bins[t]) for t in TOP_LEVEL_ORDER},
        "dev_topic_counts_multitag": dict(dev_topic_counts.most_common()),
        "eval_topic_counts_multitag": dict(eval_topic_counts.most_common()),
    }
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[wrote] {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
