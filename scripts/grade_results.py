"""Grade a directory of per-problem model responses against their gold answers.

Usage:
    python scripts/grade_results.py --dir results/smoke/sonnet-4-6
    python scripts/grade_results.py --dir results/smoke/sonnet-4-6 --use-judge

Reads every `{id}.json` in `--dir`, calls `grade()` on `response_text`
against `gold_final_answer`, and writes:
    - `{id}.graded.json` enriched with a `grade` field
    - `summary.json` consolidating eval-run fields (from run_eval.py) with
      grading results, cost estimate, and per-competition accuracy.

Cost is computed from `provider_model_id` and the eval-side token totals;
grading/judge API calls are *not* included (they currently go through the
inference harness without flowing back to this summary — minor follow-up).

Exit code 0 always (the grader never errors — a miss is a valid result).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from mathnet_eval.grading import grade


# USD per 1M tokens. Verified April 2026.
PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-7":   {"input": 5.00, "output": 25.00},
    "gpt-5.4":           {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini":      {"input": 0.75, "output": 4.50},
    "gemini-3-pro":      {"input": 2.00, "output": 12.00},
    "gemini-3-pro-preview":   {"input": 2.00, "output": 12.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    # Local HF models run on our GPU -- no per-token API cost. Kept here
    # so estimate_cost_usd returns 0.0 rather than None for these aliases.
    "qwen-2.5-1.5b-instruct": {"input": 0.0, "output": 0.0},
    "qwen-mathnet-run1":      {"input": 0.0, "output": 0.0},
    "qwen3-1.7b-base":        {"input": 0.0, "output": 0.0},
    "qwen3-mathnet-run2":     {"input": 0.0, "output": 0.0},
}


def estimate_cost_usd(
    provider_model_id: str | None,
    in_tokens: int,
    out_tokens: int,
    thinking_tokens: int = 0,
) -> float | None:
    """Cost from token totals. `thinking_tokens` (Gemini `thoughts_tokens` /
    OpenAI `reasoning_tokens` / Anthropic `thinking_tokens`) is added to the
    output side because every provider we use bills hidden reasoning at the
    output rate. Omitting it used to under-count Gemini by ~6x."""
    p = PRICING_USD_PER_MTOK.get(provider_model_id or "")
    if not p:
        return None
    billed_output = out_tokens + (thinking_tokens or 0)
    return round(
        in_tokens * p["input"] / 1_000_000 + billed_output * p["output"] / 1_000_000,
        4,
    )


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

    # Merge with the eval-run summary written by run_eval.py, if present.
    run_summary_path = args.dir / "summary.json"
    base: dict = {}
    if run_summary_path.exists():
        try:
            base = json.loads(run_summary_path.read_text())
        except Exception:
            base = {}

    provider_model_id = None
    sample = next(iter(response_files), None)
    if sample is not None:
        try:
            provider_model_id = json.loads(sample.read_text()).get("model")
        except Exception:
            pass

    in_tok = base.get("total_input_tokens", 0)
    out_tok = base.get("total_output_tokens", 0)
    # Sum hidden-reasoning tokens across all responses -- run_eval.py does not
    # aggregate these into its summary, so we do it here.
    th_tok = 0
    for f in response_files:
        try:
            u = (json.loads(f.read_text()).get("usage", {}) or {})
        except Exception:
            continue
        th_tok += int(u.get("thoughts_tokens") or u.get("reasoning_tokens") or u.get("thinking_tokens") or 0)
    eval_cost = estimate_cost_usd(provider_model_id, in_tok, out_tok, th_tok)

    n = sum(v for k, v in method_counts.items() if k != "no-gold") or 1

    # Aggregate per-problem thinking tokens (covers Gemini thoughts_tokens,
    # OpenAI reasoning_tokens, Anthropic thinking_tokens). Consistency
    # matters for Week-4 cross-model analysis — see NOTES methodology.
    import statistics
    thinking_per_problem: list[int] = []
    for f in response_files:
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        u = r.get("usage", {}) or {}
        t = u.get("thoughts_tokens") or u.get("reasoning_tokens") or u.get("thinking_tokens") or 0
        thinking_per_problem.append(int(t))

    thinking_stats: dict = {"n": len(thinking_per_problem)}
    if thinking_per_problem and max(thinking_per_problem) > 0:
        tvs = thinking_per_problem
        thinking_stats.update({
            "total": sum(tvs),
            "min": min(tvs),
            "median": int(statistics.median(tvs)),
            "mean": int(statistics.mean(tvs)),
            "max": max(tvs),
            "p95": int(statistics.quantiles(tvs, n=20)[18]) if len(tvs) >= 20 else None,
        })
    else:
        thinking_stats["note"] = "no thinking/reasoning tokens reported by this backend"

    summary = {
        **base,
        "provider_model_id": provider_model_id,
        "n_responses": len(response_files),
        "n_scored": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n,
        "method_counts": dict(method_counts),
        "per_competition": per_competition,
        "used_judge": args.use_judge,
        "estimated_eval_cost_usd": eval_cost,
        "cost_notes": "excludes LLM-as-judge API spend (not yet tracked)",
        "thinking_tokens_stats": thinking_stats,
    }
    run_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\naccuracy: {n_correct}/{n} = {100*n_correct/n:.1f}%")
    print("by method:", dict(method_counts))
    if eval_cost is not None:
        print(f"eval-side cost: ${eval_cost:.4f} (model={provider_model_id})")
    print(f"summary -> {run_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
