"""Pre-flight check for QLoRA fine-tunes that use TRL's
`completion_only_loss=True` (or the legacy `DataCollatorForCompletionOnlyLM`).
Both rely on a *substring match* of `response_template` token-ids inside
the tokenized chat-template output. If the substring is not found, the
collator silently fails to mask anything and training proceeds against the
full sequence — there is no exception, the loss just silently includes
system + user tokens.

Run this before every fine-tune on a new (model, dataset, response_template)
triple. Exits 0 if the template aligns on every sampled row, 1 otherwise.

    python scripts/verify_response_template.py \\
        --base-model Qwen/Qwen2.5-1.5B-Instruct \\
        --train-jsonl data/splits/train.jsonl \\
        --response-template '<|im_start|>assistant\\n' \\
        --n 8

This was the first concrete diagnostic that ruled out a silent-fail mode for
Run B (single-variable ablation isolating completion_only_loss). Keep it in
the launch checklist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer

DEFAULT_TEMPLATE = "<|im_start|>assistant\n"

# Match the SYSTEM_PROMPT used by the training loop so this script is a
# faithful preview of what SFTTrainer will actually tokenize.
try:
    from mathnet_eval.training import SYSTEM_PROMPT
except Exception:
    SYSTEM_PROMPT = "You are a careful, rigorous mathematics olympiad solver. Reason step by step and place the final answer in \\boxed{}."


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    L = len(needle)
    needle_t = tuple(needle)
    for i in range(len(haystack) - L + 1):
        if tuple(haystack[i:i + L]) == needle_t:
            return i
    return -1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--response-template", default=DEFAULT_TEMPLATE)
    p.add_argument("--n", type=int, default=8,
                   help="Number of rows to sample-check from the head of the JSONL")
    args = p.parse_args()

    print(f">>> loading tokenizer: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)

    template_ids = tok.encode(args.response_template, add_special_tokens=False)
    template_strs = [tok.convert_ids_to_tokens(i) for i in template_ids]
    print(f">>> response template: {args.response_template!r}")
    print(f"    token ids   : {template_ids}")
    print(f"    token strs  : {template_strs}")
    print(f"    length      : {len(template_ids)} tokens")
    print()

    rows = []
    with open(args.train_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.n:
                break

    print(f">>> sampling {len(rows)} rows from {args.train_jsonl}")
    print()

    failures: list[tuple[int, str]] = []
    masked_fracs: list[float] = []

    for i, row in enumerate(rows):
        problem = (row.get("problem_markdown") or "").strip()
        sols = row.get("solutions_markdown")
        if isinstance(sols, list):
            solution = sols[0] if sols else ""
        else:
            solution = sols or ""
        solution = solution.strip()
        final = (row.get("final_answer") or "").strip()
        if not solution:
            solution = f"\\boxed{{{final}}}"

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": problem},
            {"role": "assistant", "content": solution},
        ]
        formatted = tok.apply_chat_template(messages, tokenize=False)
        full_ids = tok.encode(formatted, add_special_tokens=False)

        idx = find_subsequence(full_ids, template_ids)
        if idx < 0:
            failures.append((i, row.get("id", "?")))
            print(f"  row {i}  id={row.get('id','?')}  TEMPLATE NOT FOUND  (len={len(full_ids)})")
            continue

        completion_start = idx + len(template_ids)
        masked_frac = completion_start / len(full_ids)
        masked_fracs.append(masked_frac)
        n_completion = len(full_ids) - completion_start
        print(f"  row {i}  id={row.get('id','?'):>6}  template@{idx:>4}  "
              f"masked={masked_frac*100:5.1f}%  completion={n_completion:>5} tok")

    print()
    if failures:
        print(f"FAIL: {len(failures)}/{len(rows)} rows did not contain the response template.")
        print("      completion_only_loss would silently fail on these rows.")
        print("      check (a) the response_template string is exact,")
        print("            (b) the chat template adds it at assistant-turn start,")
        print("            (c) tokenization is not splitting it differently.")
        return 1

    avg = sum(masked_fracs) / len(masked_fracs)
    print(f"OK: response template found in all {len(rows)} sampled rows.")
    print(f"    mean masked fraction (system + user + template): {avg*100:.1f}%")
    print(f"    completion_only_loss is wired correctly for this (model, dataset, template).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
