"""Build the TIR exemplar bank used by ``--mode tir_rag``.

For each row in `train_english.jsonl`:
  1. Run Qwen2.5-Math-1.5B-Instruct in TIR mode (single greedy rollout).
  2. Extract: the last ```python block (the "code" exemplars retrieve),
     its corresponding ```output block (the executed result), and the
     final \\boxed answer.
  3. Keep only rows where (a) the model called Python at least once AND
     (b) the boxed answer matches gold under cheap-grader.
  4. Write filtered rows to ``results/tir/exemplar_bank.jsonl`` (or the
     `--out` path).

The smoke variant skips the model entirely and writes a 3-row hand-graded
bank — the same fixture committed at ``tests/tir_smoke_exemplar_bank.jsonl``.
That keeps smoke usable without GPU and exercises the JSONL-write path.

Usage:
    python scripts/build_tir_exemplar_bank.py --smoke           # 3 fixture rows
    python scripts/build_tir_exemplar_bank.py --n 8 --smoke-real
        # 8-row real-model rollout, CPU + Qwen2.5-0.5B for sanity
    sbatch slurm/build_tir_exemplar_bank.sbatch                  # full bank, vLLM
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Put scripts/ + src/ on sys.path so we can import the backend factories
# from eval_tir.py and the library modules. Cannot use `from scripts.X`
# because the qlora env has an unrelated `scripts` package shadowing
# our directory at site-packages (regular packages outrank namespace
# packages regardless of sys.path order). Mirrors the ablation runner.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

from mathnet_eval.grading import grade  # noqa: E402
from mathnet_eval.tir import (  # noqa: E402
    PythonSandbox,
    TIRRunner,
    extract_boxed_answer,
    extract_last_python_block,
)
from mathnet_eval.tir_prompts import TIR_SYSTEM, format_tir_user  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true",
                   help="Copy the 3-row fixture bank to --out and exit. No model load.")
    p.add_argument("--smoke-real", action="store_true",
                   help="Run the real bank-builder loop on CPU with Qwen2.5-0.5B "
                        "for N rows. Slower than --smoke but exercises the full path.")
    p.add_argument("--n", type=int, default=None, help="rows to process (defaults: 8 for smoke-real, all for full)")
    p.add_argument("--train-jsonl", default="data/splits/train.jsonl")
    p.add_argument("--exclude-jsonl", default="data/splits/dev_100.jsonl",
                   help="JSONL of rows to exclude from bank construction (default: dev_100). "
                        "Prevents trivial-retrieval contamination of the 18-cell ablation.")
    p.add_argument("--out", default="results/tir/exemplar_bank.jsonl")
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-tool-calls", type=int, default=4)
    return p.parse_args()


def fixture_smoke(args) -> int:
    """Copy the committed smoke fixture to --out so the rest of the
    pipeline (eval_tir.py --mode tir_rag --smoke) has a bank to point at."""
    src = Path("tests/tir_smoke_exemplar_bank.jsonl")
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    n = sum(1 for _ in open(dst))
    print(f"[fixture-smoke] wrote {n} rows to {dst}", flush=True)
    return 0


def write_smoke_real_sentinel() -> None:
    """Write tests/.smoke_real_passed_<commit> so the bank-build sbatch
    can refuse to launch unless --smoke-real has succeeded on this commit.
    Per ruling 3 of the deviation review: the bank-build sbatch is the
    only sbatch that requires the real-rollout smoke; the eval sbatch
    requires only the cheaper sandbox-sentinel + smoke pass."""
    import subprocess as _sp

    try:
        commit = _sp.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "nogit"
    Path("tests").mkdir(exist_ok=True)
    sentinel = Path(f"tests/.smoke_real_tir_passed_{commit}")
    sentinel.write_text(f"smoke-real-tir passed at {time.time()}\n")
    print(f"[smoke-real-tir] wrote sentinel: {sentinel}", flush=True)


def real_bank_build(args) -> int:
    """Run the bank-building loop with a real model. Used by both
    --smoke-real (CPU, tiny model, N=8) and the production sbatch
    (GPU, vLLM, full train set)."""
    # Backend
    if args.backend == "hf":
        from eval_tir import make_hf_backend  # type: ignore
        generate_fn, tokenizer = make_hf_backend(args.model, device=args.device)
    elif args.backend == "vllm":
        from eval_tir import make_vllm_backend  # type: ignore
        generate_fn, tokenizer = make_vllm_backend(args.model)
    else:
        raise ValueError(args.backend)

    # Train rows
    rows: list[dict] = []
    with open(args.train_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Exclude dev-split IDs so the 18-cell retrieval ablation can't trivially
    # retrieve its own queries. Skipped if the file doesn't exist (e.g.
    # --smoke-real before dev split is built).
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

    sandbox = PythonSandbox()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_kept = 0
    n_no_python = 0
    n_wrong = 0
    t_start = time.time()
    with open(out_path, "w") as out_f:
        for i, row in enumerate(rows):
            problem = row["problem_markdown"].strip()
            gold = (row.get("final_answer") or "").strip()
            messages = [
                {"role": "system", "content": TIR_SYSTEM},
                {"role": "user", "content": format_tir_user(problem)},
            ]
            prompt_prefix = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            runner = TIRRunner(
                sandbox=sandbox,
                generate_fn=generate_fn,
                max_tool_calls=args.max_tool_calls,
                max_new_tokens_per_step=min(1024, args.max_new_tokens),
                max_new_tokens_total=args.max_new_tokens,
            )
            trace = runner.run(prompt_prefix)

            code = extract_last_python_block(trace.full_text)
            answer = extract_boxed_answer(trace.full_text) or ""
            g = grade(trace.full_text, gold, problem=problem, use_judge=False)

            if trace.n_tool_calls == 0 or code is None:
                n_no_python += 1
                continue
            if not g.correct:
                n_wrong += 1
                continue

            # Pull the matched ```output``` for the kept code block.
            tail = trace.full_text.rsplit("```python", 1)[-1]
            output_text = ""
            if "```output" in tail:
                output_chunk = tail.split("```output", 1)[1]
                # Trim closing fence
                output_text = output_chunk.split("```", 1)[0].strip()

            exemplar = {
                "id": row["id"],
                "problem": problem,
                "code": code,
                "output": output_text,
                "final_answer": answer,
                "topics_flat": row.get("topics_flat"),
                "competition": row.get("competition"),
            }
            out_f.write(json.dumps(exemplar, ensure_ascii=False) + "\n")
            n_kept += 1
            print(f"[{i+1}/{len(rows)}] id={row['id']} kept (n_tool_calls={trace.n_tool_calls})", flush=True)

    elapsed = time.time() - t_start
    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "n_rows_in": len(rows),
        "n_kept": n_kept,
        "n_no_python": n_no_python,
        "n_wrong_answer": n_wrong,
        "yield_rate": n_kept / len(rows) if rows else 0.0,
        "elapsed_s": round(elapsed, 2),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[summary] {summary}  -> {out_path}", flush=True)

    if args.smoke_real:
        # Path-validation only: write sentinel as long as the loop completed
        # without exception. n_kept may be 0 with the 0.5B stand-in model on
        # N=8 olympiad problems — that is fine, the goal is to catch
        # import/sandbox/chat-template/JSONL-write bugs, not produce a
        # usable bank. Production full runs are downstream of the sentinel
        # and never write it.
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
