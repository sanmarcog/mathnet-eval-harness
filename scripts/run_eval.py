"""Run a model over a split and save raw responses to disk.

Usage:
    python scripts/run_eval.py --model sonnet-4-6 --split eval --n 3 \
        --out results/smoke/sonnet-4-6

Each problem's response is written as a separate JSON file so a crash
mid-run doesn't lose anything. A `summary.json` aggregates at the end.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mathnet_eval.data import SYSTEM_PROMPT, format_prompt
from mathnet_eval.inference import generate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model alias from mathnet_eval.inference.MODELS")
    p.add_argument(
        "--split", required=True, type=Path,
        help="path to a JSONL file (e.g. data/splits/eval.jsonl) OR a split name ('eval'/'train') rooted at data/splits/",
    )
    p.add_argument("--n", type=int, default=None, help="limit to first N problems; default = all")
    p.add_argument("--out", required=True, type=Path, help="output dir for per-problem JSONs")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--no-cache", action="store_true",
        help="force re-call provider even if cached response exists",
    )
    return p.parse_args()


def resolve_split(path: Path) -> Path:
    if path.exists():
        return path
    # Allow bare "eval" / "train".
    fallback = Path("data/splits") / f"{path}.jsonl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"split not found: {path} (also tried {fallback})")


def main() -> int:
    args = parse_args()
    split_path = resolve_split(args.split)
    args.out.mkdir(parents=True, exist_ok=True)

    problems = [json.loads(line) for line in split_path.read_text().splitlines()]
    if args.n is not None:
        problems = problems[: args.n]
    print(f"running {args.model} on {len(problems)} problems from {split_path}")

    t_start = time.perf_counter()
    total_in_tokens = total_out_tokens = 0
    cached_hits = 0
    errors: list[dict] = []

    for i, p in enumerate(problems):
        pid = p["id"]
        out_path = args.out / f"{pid}.json"
        user_prompt = format_prompt(p)
        try:
            resp = generate(
                user_prompt,
                args.model,
                use_cache=not args.no_cache,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                system=SYSTEM_PROMPT,
            )
        except Exception as e:
            errors.append({"id": pid, "error": repr(e)})
            print(f"  [{i+1}/{len(problems)}] id={pid}  ERROR: {e!r}")
            continue

        record = {
            "id": pid,
            "country": p.get("country"),
            "competition": p.get("competition"),
            "gold_final_answer": p.get("final_answer"),
            "topics_flat": p.get("topics_flat"),
            "prompt": user_prompt,
            "model": resp.model,
            "response_text": resp.text,
            "usage": resp.usage,
            "latency_s": resp.latency_s,
            "cached": resp.cached,
        }
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))

        total_in_tokens += resp.usage.get("input_tokens", 0)
        total_out_tokens += resp.usage.get("output_tokens", 0)
        cached_hits += int(resp.cached)
        tag = "cache" if resp.cached else f"{resp.latency_s:.1f}s"
        print(
            f"  [{i+1}/{len(problems)}] id={pid} {args.model} ({tag}) "
            f"in={resp.usage.get('input_tokens')} out={resp.usage.get('output_tokens')}"
        )

    elapsed = time.perf_counter() - t_start
    summary = {
        "model": args.model,
        "split": str(split_path),
        "n_problems": len(problems),
        "n_errors": len(errors),
        "cached_hits": cached_hits,
        "total_input_tokens": total_in_tokens,
        "total_output_tokens": total_out_tokens,
        "elapsed_s": elapsed,
        "errors": errors,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"\ndone in {elapsed:.1f}s: {len(problems) - len(errors)} ok, {len(errors)} errors, "
        f"{cached_hits} cached. tokens in={total_in_tokens} out={total_out_tokens}. "
        f"summary -> {args.out}/summary.json"
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
