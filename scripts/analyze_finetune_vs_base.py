"""Paired analysis of a fine-tuned model's results vs the base model on
the same problem IDs. Produces a markdown report covering:

  - paired accuracy + transition matrix
  - McNemar exact two-sided p
  - per-competition delta (top regressions and improvements)
  - grader-path comparison (exact / normalized / symbolic / judge / miss)
  - miss-rate decomposition: saturation-driven vs wrong-but-committed

Designed to be runnable the moment Run 2 results land — no scrambling.

    python scripts/analyze_finetune_vs_base.py \\
        --finetune-dir results/full/qwen3-1.7b-run2 \\
        --base-dir     results/full/qwen3-1.7b-base \\
        --out          docs/run2_analysis.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path


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
            "competition": data.get("competition", "?"),
            "output_tokens": data.get("usage", {}).get("output_tokens", 0),
            "has_boxed": "\\boxed{" in (data.get("response_text") or ""),
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
    p.add_argument("--finetune-dir", required=True, type=Path)
    p.add_argument("--base-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="Markdown report path")
    p.add_argument("--saturation-threshold", type=int, default=16384,
                   help="output_tokens >= this is treated as ceiling-hit")
    args = p.parse_args()

    ft = load_graded(args.finetune_dir)
    base = load_graded(args.base_dir)
    shared = sorted(set(ft) & set(base))
    if not shared:
        print(f"ERROR: no shared IDs between {args.finetune_dir} and {args.base_dir}")
        return 1

    n = len(shared)
    ft_correct = sum(1 for i in shared if ft[i]["correct"])
    base_correct = sum(1 for i in shared if base[i]["correct"])

    both = sum(1 for i in shared if ft[i]["correct"] and base[i]["correct"])
    ft_only = sum(1 for i in shared if ft[i]["correct"] and not base[i]["correct"])
    base_only = sum(1 for i in shared if not ft[i]["correct"] and base[i]["correct"])
    neither = sum(1 for i in shared if not ft[i]["correct"] and not base[i]["correct"])
    p_val = mcnemar_two_sided_p(base_only, ft_only)

    methods = ["exact", "normalized", "symbolic", "judge", "miss"]
    base_methods = {m: 0 for m in methods}
    ft_methods = {m: 0 for m in methods}
    for i in shared:
        base_methods[base[i]["method"]] += 1
        ft_methods[ft[i]["method"]] += 1

    # miss decomposition (ft side)
    saturated = sum(1 for i in shared
                    if not ft[i]["correct"] and ft[i]["output_tokens"] >= args.saturation_threshold)
    saturated_no_boxed = sum(1 for i in shared
                             if not ft[i]["correct"]
                             and ft[i]["output_tokens"] >= args.saturation_threshold
                             and not ft[i]["has_boxed"])
    wrong_committed = sum(1 for i in shared
                          if not ft[i]["correct"]
                          and ft[i]["output_tokens"] < args.saturation_threshold)
    base_saturated = sum(1 for i in shared
                         if not base[i]["correct"]
                         and base[i]["output_tokens"] >= args.saturation_threshold)
    base_sat_no_boxed = sum(1 for i in shared
                            if not base[i]["correct"]
                            and base[i]["output_tokens"] >= args.saturation_threshold
                            and not base[i]["has_boxed"])

    # per-competition delta (ft - base correct count)
    by_comp: dict[str, dict] = defaultdict(lambda: {"n": 0, "ft": 0, "base": 0})
    for i in shared:
        c = ft[i]["competition"] or "?"
        by_comp[c]["n"] += 1
        by_comp[c]["ft"] += int(ft[i]["correct"])
        by_comp[c]["base"] += int(base[i]["correct"])
    comp_rows = sorted(
        ((c, v["n"], v["base"], v["ft"], v["ft"] - v["base"]) for c, v in by_comp.items()),
        key=lambda r: (r[4], -r[1])
    )
    bottom_5 = [r for r in comp_rows if r[1] >= 2][:5]
    top_5 = [r for r in comp_rows if r[1] >= 2][-5:][::-1]

    lines: list[str] = []
    lines.append(f"# Run 2 vs base — paired analysis (n={n})\n")
    lines.append(f"Compared: `{args.finetune_dir.name}` vs `{args.base_dir.name}`\n")

    lines.append("## Headline\n")
    lines.append("| | Correct | Accuracy |")
    lines.append("|---|---|---|")
    lines.append(f"| {args.base_dir.name} | {base_correct}/{n} | **{base_correct/n*100:.1f}%** |")
    lines.append(f"| {args.finetune_dir.name} | {ft_correct}/{n} | **{ft_correct/n*100:.1f}%** |")
    lines.append(f"| **Delta** | | **{(ft_correct-base_correct)/n*100:+.1f} pp** |")
    lines.append("")
    lines.append(f"McNemar exact two-sided **p = {p_val:.4f}** "
                 f"(discordant {base_only + ft_only}; smaller side {min(base_only, ft_only)})\n")

    lines.append("## Transition matrix\n")
    lines.append("|   | base ✓ | base ✗ |")
    lines.append("|---|---|---|")
    lines.append(f"| **ft ✓** | {both} | {ft_only}  *(improved)* |")
    lines.append(f"| **ft ✗** | {base_only}  *(regressed)* | {neither} |")
    lines.append("")

    lines.append("## Grader-path counts\n")
    lines.append("| Path | base | ft | delta |")
    lines.append("|---|---|---|---|")
    for m in methods:
        lines.append(f"| {m} | {base_methods[m]} | {ft_methods[m]} | "
                     f"{ft_methods[m] - base_methods[m]:+d} |")
    lines.append("")

    lines.append("## Miss decomposition (ft model)\n")
    n_miss = n - ft_correct
    lines.append(f"- Total `miss`: **{n_miss}** ({n_miss/n*100:.1f}%)")
    lines.append(f"- ...of which **saturated at ≥{args.saturation_threshold} output tokens**: "
                 f"{saturated} ({saturated/max(1,n_miss)*100:.0f}% of misses)")
    lines.append(f"- ...of which **saturated AND no `\\boxed{{}}`** (convergence failure): "
                 f"{saturated_no_boxed} ({saturated_no_boxed/max(1,n_miss)*100:.0f}% of misses)")
    lines.append(f"- ...of which **wrong-but-committed** (output < ceiling): "
                 f"{wrong_committed} ({wrong_committed/max(1,n_miss)*100:.0f}% of misses)")
    lines.append("")
    lines.append(f"For comparison, base had {n - base_correct} misses, "
                 f"{base_saturated} saturated ({base_saturated/max(1,n-base_correct)*100:.0f}%), "
                 f"{base_sat_no_boxed} saturated-and-no-boxed.")
    lines.append("")

    lines.append("## Where Run 2 helped most (top 5 competitions, n≥2)\n")
    lines.append("| Competition | n | base | ft | delta |")
    lines.append("|---|---|---|---|---|")
    for c, nn, bb, ff, d in top_5:
        lines.append(f"| {c[:60]} | {nn} | {bb} | {ff} | {d:+d} |")
    lines.append("")
    lines.append("## Where Run 2 regressed most (bottom 5 competitions, n≥2)\n")
    lines.append("| Competition | n | base | ft | delta |")
    lines.append("|---|---|---|---|---|")
    for c, nn, bb, ff, d in bottom_5:
        lines.append(f"| {c[:60]} | {nn} | {bb} | {ff} | {d:+d} |")
    lines.append("")

    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    print(f"\nQuick read:")
    print(f"  base    {base_correct}/{n} = {base_correct/n*100:.1f}%")
    print(f"  ft      {ft_correct}/{n} = {ft_correct/n*100:.1f}%")
    print(f"  delta   {(ft_correct-base_correct)/n*100:+.1f} pp")
    print(f"  p       {p_val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
