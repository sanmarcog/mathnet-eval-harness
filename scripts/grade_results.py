"""Grade a directory of per-problem model responses against their gold answers.

Usage:
    python scripts/grade_results.py --dir results/smoke/sonnet-4-6
    python scripts/grade_results.py --dir results/smoke/sonnet-4-6 --use-judge

Reads every `{id}.json` in `--dir`, calls `grade()` on the `response_text`
against `gold_final_answer`, and writes:
    - `{id}.graded.json` enriched with a `grade` field
    - `grading_summary.json` with overall accuracy + per-method counts

Exit code 0 always (the grader never errors — a miss is a valid result).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from mathnet_eval.grading import grade


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, type=Path, help="directory of {id}.json response files")
    p.add_argument("--use-judge", action="store_true", help="enable LLM-as-judge fallback layer")
    p.add_argument("--judge-model", default="sonnet-4-6")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    response_files = sorted(p for p in args.dir.glob("*.json")
                            if not p.name.endswith(".graded.json")
                            and p.name not in {"summary.json", "grading_summary.json"})
    print(f"grading {len(response_files)} responses in {args.dir}")

    method_counts: Counter[str] = Counter()
    n_correct = 0
    per_competition: dict[str, dict[str, int]] = {}

    for f in response_files:
        rec = json.loads(f.read_text())
        gold = rec.get("gold_final_answer")
        if not isinstance(gold, str) or not gold.strip():
            method_counts["no-gold"] += 1
            continue

        g = grade(
            rec["response_text"],
            gold,
            problem=rec.get("prompt"),
            use_judge=args.use_judge,
            judge_model=args.judge_model,
        )
        rec["grade"] = asdict(g)
        (args.dir / f"{f.stem}.graded.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))

        method_counts[g.method] += 1
        if g.correct:
            n_correct += 1

        comp = rec.get("competition") or "unknown"
        d = per_competition.setdefault(comp, {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += int(g.correct)

        status = "✓" if g.correct else "✗"
        pred = (g.predicted or "")[:80]
        gold_preview = gold[:60]
        print(f"  {status} id={rec['id']} method={g.method:>10s}  pred={pred!r}  gold={gold_preview!r}")

    n = sum(method_counts.values()) or 1
    summary = {
        "dir": str(args.dir),
        "n_responses": len(response_files),
        "n_scored": sum(v for k, v in method_counts.items() if k != "no-gold"),
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        "method_counts": dict(method_counts),
        "per_competition": per_competition,
        "used_judge": args.use_judge,
    }
    (args.dir / "grading_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\naccuracy: {n_correct}/{n} = {100*n_correct/n:.1f}%")
    print("by method:", dict(method_counts))
    print(f"summary -> {args.dir}/grading_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
