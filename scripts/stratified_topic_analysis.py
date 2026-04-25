"""Stratified accuracy breakdown by topic prefix from MathNet's
`topics_flat` field. A problem can be tagged with multiple topics; this
script counts each problem ONCE per distinct top-level prefix it carries
(so a Algebra+Number Theory problem contributes to both rows). Top-level
prefixes are the four MathNet roots: Algebra, Geometry, Number Theory,
Discrete Mathematics.

    python scripts/stratified_topic_analysis.py \\
        --dir results/full/qwen3-1.7b-base \\
        --out docs/qwen3_base_topic_breakdown.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


TOP_LEVEL = ["Algebra", "Number Theory", "Discrete Mathematics", "Geometry"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--min-n", type=int, default=5,
                   help="Hide subtopic rows with fewer problems than this")
    args = p.parse_args()

    files = list(args.dir.glob("*.graded.json"))
    if not files:
        print(f"ERROR: no .graded.json files in {args.dir}")
        return 1

    by_top = defaultdict(lambda: {"n": 0, "correct": 0})
    by_sub = defaultdict(lambda: defaultdict(lambda: {"n": 0, "correct": 0}))

    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        correct = d.get("grade", {}).get("method") != "miss"
        topics = d.get("topics_flat") or []
        seen_top = set()
        seen_sub = set()
        for t in topics:
            parts = [s.strip() for s in t.split(" > ")]
            if not parts:
                continue
            top = parts[0]
            seen_top.add(top)
            if len(parts) >= 2:
                sub = parts[1]
                seen_sub.add((top, sub))
        for top in seen_top:
            by_top[top]["n"] += 1
            by_top[top]["correct"] += int(correct)
        for top, sub in seen_sub:
            by_sub[top][sub]["n"] += 1
            by_sub[top][sub]["correct"] += int(correct)

    lines: list[str] = []
    lines.append(f"# Topic-stratified accuracy — `{args.dir.name}`\n")
    lines.append(f"Sample size: {len(files)} problems "
                 f"(each may be tagged with multiple top-level prefixes).\n")

    lines.append("## Top-level breakdown\n")
    lines.append("| Topic | n | correct | accuracy |")
    lines.append("|---|---|---|---|")
    rows = sorted(((t, by_top[t]) for t in TOP_LEVEL if t in by_top),
                  key=lambda r: -r[1]["n"])
    for top, v in rows:
        lines.append(f"| {top} | {v['n']} | {v['correct']} | "
                     f"**{v['correct']/v['n']*100:.1f}%** |")
    lines.append("")

    for top in TOP_LEVEL:
        if top not in by_sub:
            continue
        lines.append(f"## {top} — subtopic breakdown\n")
        lines.append("| Subtopic | n | correct | accuracy |")
        lines.append("|---|---|---|---|")
        sub_rows = sorted(by_sub[top].items(), key=lambda x: -x[1]["n"])
        for sub, v in sub_rows:
            if v["n"] < args.min_n:
                continue
            lines.append(f"| {sub} | {v['n']} | {v['correct']} | "
                         f"**{v['correct']/v['n']*100:.1f}%** |")
        lines.append("")

    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
