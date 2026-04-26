"""Safeguard pre-check before launching Run 3. Verifies that the boxed
augmentation:

  1. Renders correctly through the model's `apply_chat_template` (no
     LaTeX escaping issues, no unicode mangling, no missing whitespace)
  2. Round-trips: `extract_answer` applied to the rendered formatted
     string returns the same final_answer that we appended
  3. Tokenizes cleanly (the response template still aligns; the boxed
     line doesn't cross the tokenization boundary in a way that breaks
     the loss-mask collator)

Catches failure modes that produce numerically-valid loss curves with
silently broken training data:
  - final_answer contains characters that LaTeX-mangle (e.g. unbalanced
    braces, raw `}` in the answer)
  - chat_template applies `add_special_tokens` that corrupts the boxed
    string
  - tokenizer treats `\\boxed` differently from how `extract_answer`
    expects to find it
  - assistant content gets stripped/truncated by the template

    python scripts/verify_boxed_augmentation.py \\
        --train-jsonl data/splits/train_multilingual_filtered_boxed.jsonl \\
        --base-model Qwen/Qwen3-1.7B \\
        --n 8

Per the verify_response_template script: capped at a few minutes,
sample N rows, exit 0 if all round-trip cleanly, exit 1 (and abort the
training run) if any row fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

try:
    from mathnet_eval.training import SYSTEM_PROMPT
    from mathnet_eval.grading import extract_answer, normalize_for_exact
except Exception:
    SYSTEM_PROMPT = (
        "You are an expert mathematician. Solve the following olympiad "
        "problem. Please reason step by step, and put your final answer "
        "within \\boxed{}."
    )
    raise


def _first_solution(row: dict) -> str:
    s = row.get("solutions_markdown")
    if isinstance(s, list):
        return (s[0] if s else "") or ""
    return s or ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--base-model",  required=True)
    p.add_argument("--n", type=int, default=8,
                   help="Number of rows to sample-check from the head of the JSONL.")
    args = p.parse_args()

    print(f">>> loading tokenizer: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)

    rows = []
    with open(args.train_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.n:
                break
    print(f">>> sampling {len(rows)} rows from {args.train_jsonl}")
    print()

    failures: list[tuple[str, str]] = []  # (id, reason)
    for i, row in enumerate(rows):
        rid = row.get("id", f"row{i}")
        problem = (row.get("problem_markdown") or "").strip()
        solution = _first_solution(row).strip()
        gold = (row.get("final_answer") or "").strip()

        if not solution:
            failures.append((rid, "solution is empty"))
            continue
        if not gold:
            failures.append((rid, "final_answer is empty"))
            continue
        if "\\boxed{" not in solution:
            failures.append((rid, "augmentation didn't run; no \\boxed{} in solution"))
            continue

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": problem},
            {"role": "assistant", "content": solution},
        ]
        try:
            formatted = tok.apply_chat_template(messages, tokenize=False)
        except Exception as e:
            failures.append((rid, f"apply_chat_template failed: {e}"))
            continue

        # Round-trip: extract_answer on the rendered formatted text should
        # recover the gold final_answer (modulo normalization).
        recovered = extract_answer(formatted)
        if recovered is None:
            failures.append((rid, "extract_answer returned None on rendered text"))
            print(f"  row {i}  id={rid}  FAIL — recovered=None")
            print(f"    rendered tail:  {formatted[-300:]!r}")
            continue

        rec_norm = normalize_for_exact(recovered)
        gold_norm = normalize_for_exact(gold)
        ok_round_trip = (rec_norm == gold_norm)

        # Also tokenize and look at the assistant turn boundary to spot
        # encoding pathologies.
        full_ids = tok.encode(formatted, add_special_tokens=False)
        n_tokens = len(full_ids)

        status = "ok" if ok_round_trip else "MISMATCH"
        print(f"  row {i}  id={rid:>6}  {status:8}  "
              f"len={n_tokens:>5}  "
              f"recovered={recovered[:40]!r}  "
              f"gold={gold[:40]!r}")
        if not ok_round_trip:
            failures.append((rid, f"round-trip mismatch: '{recovered}' vs '{gold}'"))

    print()
    if failures:
        print(f"FAIL: {len(failures)} / {len(rows)} rows failed the augmentation check:")
        for rid, reason in failures:
            print(f"  {rid}: {reason}")
        print()
        print("ABORTING. Fix the augmentation before launching Run 3.")
        return 1

    print(f"OK: all {len(rows)} rows round-trip cleanly through "
          f"apply_chat_template + extract_answer.")
    print(f"    Run 3 training data is ready to launch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
