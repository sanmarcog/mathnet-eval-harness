"""Paired McNemar exact two-sided test between two model result dirs.

Use to defend (or refute) a parity claim with proper paired statistics.
The 36.8% / 36.7% Qwen3-1.7B vs Mini "parity" finding needs this number
because it sits next to claims that DO report McNemar (e.g. Run 4 vs base).

    python scripts/compute_parity_mcnemar.py \\
        --a results/full/qwen3-1.7b-base \\
        --b results/full/gpt-5.4-mini

Prints accuracies on the paired intersection plus the McNemar exact
two-sided p-value. No file output — just console.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def load_correct(d: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for f in d.glob("*.graded.json"):
        if f.name.startswith("summary"):
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        pid = data.get("id")
        if pid is None:
            continue
        out[pid] = (data.get("grade") or {}).get("method", "miss") != "miss"
    return out


def mcnemar_two_sided_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    one_sided = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, one_sided * 2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, type=Path)
    p.add_argument("--b", required=True, type=Path)
    args = p.parse_args()

    a = load_correct(args.a)
    b = load_correct(args.b)
    shared = sorted(set(a) & set(b))
    if not shared:
        print(f"ERROR: no shared IDs between {args.a} and {args.b}")
        return 1

    n = len(shared)
    a_correct = sum(1 for i in shared if a[i])
    b_correct = sum(1 for i in shared if b[i])
    both    = sum(1 for i in shared if a[i] and b[i])
    a_only  = sum(1 for i in shared if a[i] and not b[i])
    b_only  = sum(1 for i in shared if not a[i] and b[i])
    neither = sum(1 for i in shared if not a[i] and not b[i])
    p_val = mcnemar_two_sided_p(a_only, b_only)

    print(f"  paired n           = {n}")
    print(f"  {args.a.name:<25s} = {a_correct}/{n}  ({a_correct/n*100:.1f}%)")
    print(f"  {args.b.name:<25s} = {b_correct}/{n}  ({b_correct/n*100:.1f}%)")
    print()
    print(f"  transition matrix:")
    print(f"    both correct          {both}")
    print(f"    {args.a.name} only    {a_only}  (improvements over {args.b.name})")
    print(f"    {args.b.name} only    {b_only}  (regressions vs {args.b.name})")
    print(f"    neither               {neither}")
    print()
    print(f"  McNemar exact two-sided p = {p_val:.4f}")
    print(f"  (discordant pairs {a_only + b_only}; smaller side {min(a_only, b_only)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
