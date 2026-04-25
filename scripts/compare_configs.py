"""Compare N graded result directories on a single eval split. Produces
a markdown table for the writeup with per-config accuracy, grader-path
distribution, and optional pairwise McNemar tests against a baseline.

Designed to drop into the writeup as the "we ran 4 configs" table the
manager wants — Qwen3 base single-sample, Qwen3 base BoN, Run 2 single,
Run 2 BoN — and any other axes we add later (budget forcing, etc.).

    python scripts/compare_configs.py \\
        --baseline results/full/qwen3-1.7b-base \\
        --configs results/full/qwen3-1.7b-base-bon8 \\
                  results/full/qwen3-1.7b-run2 \\
                  results/full/qwen3-1.7b-run2-bon8 \\
        --out docs/test_time_scaling_table.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path


METHODS = ["exact", "normalized", "symbolic", "judge", "miss"]


def load_graded(d: Path) -> dict:
    out = {}
    for f in d.glob("*.graded.json"):
        if f.name.startswith("summary"):
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        pid = data.get("id")
        if pid is None:
            continue
        out[pid] = {
            "correct": data.get("grade", {}).get("method") != "miss",
            "method":  data.get("grade", {}).get("method", "miss"),
        }
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
    p.add_argument("--baseline", required=True, type=Path,
                   help="Reference result dir; pairwise tests run against this.")
    p.add_argument("--configs", nargs="+", required=True, type=Path,
                   help="One or more result dirs to compare against the baseline.")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    all_dirs = [args.baseline] + list(args.configs)
    loaded = {d.name: load_graded(d) for d in all_dirs}

    # Intersect IDs across all loaded dirs so the comparison is paired.
    id_sets = [set(g.keys()) for g in loaded.values()]
    if not id_sets:
        print("ERROR: no result dirs loaded")
        return 1
    shared = sorted(set.intersection(*id_sets))
    if not shared:
        print("ERROR: no shared problem IDs across all dirs")
        return 1
    print(f">>> shared IDs across {len(loaded)} dirs: {len(shared)}")

    rows = []
    base_name = args.baseline.name
    base_g = loaded[base_name]
    base_correct = sum(1 for i in shared if base_g[i]["correct"])

    for d in all_dirs:
        g = loaded[d.name]
        n_correct = sum(1 for i in shared if g[i]["correct"])
        method_counts = {m: 0 for m in METHODS}
        for i in shared:
            method_counts[g[i]["method"]] += 1
        delta = n_correct - base_correct

        if d.name == base_name:
            p_val = None
        else:
            base_only = sum(1 for i in shared
                            if base_g[i]["correct"] and not g[i]["correct"])
            cfg_only = sum(1 for i in shared
                           if not base_g[i]["correct"] and g[i]["correct"])
            p_val = mcnemar_two_sided_p(base_only, cfg_only)

        rows.append({
            "name": d.name,
            "correct": n_correct,
            "accuracy": n_correct / len(shared) * 100,
            "delta_pp": delta / len(shared) * 100,
            "method_counts": method_counts,
            "p_val": p_val,
        })

    lines: list[str] = []
    lines.append("# Test-time-scaling table (paired)\n")
    lines.append(f"All comparisons are paired on the {len(shared)} problem IDs "
                 f"present in every result directory.\n")
    lines.append(f"Baseline: `{base_name}`\n")
    lines.append("## Headline\n")
    lines.append("| Config | n correct | Accuracy | Δ vs baseline | McNemar p |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        delta_str = (f"{r['delta_pp']:+.1f} pp" if r["name"] != base_name else "—")
        p_str = (f"{r['p_val']:.4f}" if r["p_val"] is not None else "—")
        lines.append(f"| {r['name']} | {r['correct']}/{len(shared)} | "
                     f"**{r['accuracy']:.1f}%** | {delta_str} | {p_str} |")
    lines.append("")

    lines.append("## Grader-path distribution (counts)\n")
    header = "| Config | " + " | ".join(METHODS) + " |"
    sep = "|---|" + "---|" * len(METHODS)
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = " | ".join(str(r["method_counts"][m]) for m in METHODS)
        lines.append(f"| {r['name']} | {cells} |")
    lines.append("")

    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    print()
    print(f"  {'config':<45}  {'acc':>6}  {'Δ pp':>6}  {'p':>6}")
    for r in rows:
        delta_str = (f"{r['delta_pp']:+.1f}" if r["name"] != base_name else "—")
        p_str = (f"{r['p_val']:.3f}" if r["p_val"] is not None else "—")
        print(f"  {r['name']:<45}  {r['accuracy']:5.1f}%  {delta_str:>6}  {p_str:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
