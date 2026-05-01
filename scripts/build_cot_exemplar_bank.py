"""Build the CoT exemplar bank used by ``--mode tir_rag --exemplar-type cot``
in the 18-cell retrieval ablation.

Mirrors `build_tir_exemplar_bank.py` but in CoT mode: model generates a
single greedy response (no Python sandbox), we extract the `\\boxed`
answer, filter for correctness, and save the natural-language reasoning
(everything before the boxed answer) as the exemplar's `reasoning` field.

Why a separate bank: the project's headline secondary question is "do
tool-using rollouts as exemplars beat CoT-only rollouts as exemplars at
1.5B?" — a question about the *content* of retrieved exemplars, not the
formatting. Re-formatting TIR exemplars as CoT (the dev-convenience
fallback) tests a related but different question. The dedicated CoT
bank lets the ablation answer the headline question cleanly.

Filter: keep rows where (a) `extract_answer(rollout)` matches gold under
cheap-grader, AND (b) some `reasoning` text precedes the boxed answer
(rows that just emit `\\boxed{X}` with no reasoning are dropped — they
provide no exemplar value).

Usage:
    python scripts/build_cot_exemplar_bank.py --smoke           # fixture copy
    python scripts/build_cot_exemplar_bank.py --smoke-real      # 8-row CPU sanity
    sbatch slurm/build_cot_exemplar_bank.sbatch                  # full bank, vLLM
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess as _sp
import sys
import time
from pathlib import Path

# scripts/ + src/ on sys.path so we can import eval_tir's backend
# factories. Cannot use `from scripts.eval_tir` because the qlora env
# has an unrelated `scripts` package shadowing our directory at
# site-packages (regular packages outrank namespace packages).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

from mathnet_eval.grading import grade  # noqa: E402
from mathnet_eval.tir import extract_boxed_answer  # noqa: E402
from mathnet_eval.tir_prompts import COT_SYSTEM, format_cot_user  # noqa: E402


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true",
                   help="Copy the 3-row fixture bank to --out and exit. No model load.")
    p.add_argument("--smoke-real", action="store_true",
                   help="Run the real CoT bank-builder loop on CPU with Qwen2.5-0.5B "
                        "for N rows. Writes tests/.smoke_real_cot_passed_<commit>.")
    p.add_argument("--n", type=int, default=None, help="rows to process (defaults: 8 for smoke-real, all for full)")
    p.add_argument("--train-jsonl", default="data/splits/train.jsonl")
    p.add_argument("--exclude-jsonl", default="data/splits/dev_100.jsonl",
                   help="JSONL of rows to exclude from bank construction (default: dev_100). "
                        "Prevents trivial-retrieval contamination of the 18-cell ablation.")
    p.add_argument("--out", default="results/tir/exemplar_bank_cot.jsonl")
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    return p.parse_args()


def fixture_smoke(args) -> int:
    """Copy the committed smoke fixture (which carries both TIR + CoT
    fields) to --out so the ablation runner has a CoT bank to point at."""
    src = Path("tests/tir_smoke_exemplar_bank.jsonl")
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    n = sum(1 for _ in open(dst))
    print(f"[fixture-smoke] wrote {n} rows to {dst}", flush=True)
    return 0


def write_smoke_real_sentinel() -> None:
    try:
        commit = _sp.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "nogit"
    Path("tests").mkdir(exist_ok=True)
    sentinel = Path(f"tests/.smoke_real_cot_passed_{commit}")
    sentinel.write_text(f"smoke-real-cot passed at {time.time()}\n")
    print(f"[smoke-real-cot] wrote sentinel: {sentinel}", flush=True)


def split_reasoning_and_boxed(text: str) -> tuple[str, str | None]:
    """Return (reasoning, boxed_answer). Reasoning is everything before
    the LAST `\\boxed{...}` occurrence, with that final-answer line
    stripped. If no `\\boxed`, returns (text, None)."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return text.strip(), None
    last = matches[-1]
    reasoning = text[: last.start()].rstrip()
    # Strip trailing "Final answer:" / "Therefore," scaffolding lines.
    reasoning = re.sub(r"\n\s*(?:therefore|thus|so|hence|final\s+answer)[^.\n]*[.:]?\s*$",
                       "", reasoning, flags=re.IGNORECASE).rstrip()
    return reasoning, last.group(1).strip()


def real_bank_build(args) -> int:
    if args.backend == "hf":
        from eval_tir import make_hf_backend  # type: ignore
        generate_fn, tokenizer = make_hf_backend(args.model, device=args.device)
    elif args.backend == "vllm":
        from eval_tir import make_vllm_backend  # type: ignore
        generate_fn, tokenizer = make_vllm_backend(args.model)
    else:
        raise ValueError(args.backend)

    rows: list[dict] = []
    with open(args.train_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    excluded_ids: set[str] = set()
    excl_path = Path(args.exclude_jsonl)
    if excl_path.exists():
        with open(excl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    excluded_ids.add(json.loads(line)["id"])
    n_before = len(rows)
    rows = [r for r in rows if r["id"] not in excluded_ids]
    print(f"[data] excluded {n_before - len(rows)} dev IDs from {excl_path}", flush=True)

    cap = args.n if args.n is not None else (8 if args.smoke_real else None)
    if cap:
        rows = rows[:cap]
    print(f"[data] loaded {len(rows)} train rows", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_kept = 0
    n_no_answer = 0
    n_no_reasoning = 0
    n_wrong = 0
    t_start = time.time()
    with open(out_path, "w") as out_f:
        for i, row in enumerate(rows):
            problem = row["problem_markdown"].strip()
            gold = (row.get("final_answer") or "").strip()
            messages = [
                {"role": "system", "content": COT_SYSTEM},
                {"role": "user", "content": format_cot_user(problem)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            response = generate_fn(prompt, args.max_new_tokens, ["<|im_end|>"])

            reasoning, boxed = split_reasoning_and_boxed(response)
            if boxed is None:
                n_no_answer += 1
                continue
            if not reasoning.strip():
                n_no_reasoning += 1
                continue
            g = grade(response, gold, problem=problem, use_judge=False)
            if not g.correct:
                n_wrong += 1
                continue

            exemplar = {
                "id": row["id"],
                "problem": problem,
                "reasoning": reasoning,
                "final_answer": boxed,
                "topics_flat": row.get("topics_flat"),
                "competition": row.get("competition"),
            }
            out_f.write(json.dumps(exemplar, ensure_ascii=False) + "\n")
            n_kept += 1
            print(f"[{i+1}/{len(rows)}] id={row['id']} kept (reasoning_chars={len(reasoning)})", flush=True)

    elapsed = time.time() - t_start
    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "mode": "cot",
        "n_rows_in": len(rows),
        "n_kept": n_kept,
        "n_no_answer": n_no_answer,
        "n_no_reasoning": n_no_reasoning,
        "n_wrong_answer": n_wrong,
        "yield_rate": n_kept / len(rows) if rows else 0.0,
        "elapsed_s": round(elapsed, 2),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[summary] {summary}  -> {out_path}", flush=True)

    if args.smoke_real:
        # Same rationale as build_tir_exemplar_bank.py: --smoke-real
        # validates the pipeline; n_kept may be 0 with 0.5B on 8 olympiad
        # problems. Sentinel writes regardless so bank-build sbatch can
        # launch on a fresh commit.
        write_smoke_real_sentinel()
        return 0

    return 0 if n_kept > 0 else 1


def main() -> int:
    args = parse_args()
    if args.smoke:
        return fixture_smoke(args)
    return real_bank_build(args)


if __name__ == "__main__":
    sys.exit(main())
