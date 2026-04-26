"""Augment a training JSONL by appending an explicit boxed-answer line to
each row's solution. This is the Run 3 fix for the Run 2 regression:
Run 2 trained on raw MathNet `solutions_markdown` (only ~1.5% of which
contained \\boxed{}), which catastrophically unlearned the boxing
convention from the base model. -33.8 pp paired delta on the eval, p < 0.0001.

Augmentation per row:
  if the existing solution already contains \\boxed{} -> leave it
  else -> append "\\n\\nTherefore, the final answer is $\\boxed{<final_answer>}$"

Skips rows whose final_answer is missing (those provide no signal). The
output JSONL is otherwise identical to the input (same row order, same
fields, same id). The 'solutions_markdown' field is the only thing that
changes.

    python scripts/augment_train_with_boxed.py \\
        --in  data/splits/train_multilingual_filtered.jsonl \\
        --out data/splits/train_multilingual_filtered_boxed.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BOXED_SUFFIX_TEMPLATE = "\n\nTherefore, the final answer is $\\boxed{{{ans}}}$"


def get_first_solution(row: dict) -> str:
    s = row.get("solutions_markdown")
    if isinstance(s, list):
        return (s[0] if s else "") or ""
    return s or ""


def augment_row(row: dict) -> tuple[dict, str]:
    """Returns (new_row, status) where status is one of:
        'appended'  — added the boxed line
        'kept'      — already had \\boxed{}, left untouched
        'skipped'   — no final_answer or no solution; cannot augment
    """
    sol = get_first_solution(row).strip()
    final = (row.get("final_answer") or "").strip()
    if not sol or not final:
        return row, "skipped"
    if "\\boxed{" in sol:
        return row, "kept"

    suffix = BOXED_SUFFIX_TEMPLATE.format(ans=final)
    new_sol = sol + suffix

    new_row = dict(row)
    if isinstance(row.get("solutions_markdown"), list):
        new_row["solutions_markdown"] = [new_sol] + (row["solutions_markdown"][1:] if row["solutions_markdown"] else [])
    else:
        new_row["solutions_markdown"] = new_sol
    return new_row, "appended"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="in_path",  required=True, type=Path)
    p.add_argument("--out", dest="out_path", required=True, type=Path)
    args = p.parse_args()

    rows = [json.loads(l) for l in args.in_path.open()]
    print(f">>> read {len(rows)} rows from {args.in_path}")

    counts = {"appended": 0, "kept": 0, "skipped": 0}
    out_rows: list[dict] = []
    for r in rows:
        new_r, status = augment_row(r)
        counts[status] += 1
        if status != "skipped":
            out_rows.append(new_r)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f">>> appended:  {counts['appended']}")
    print(f">>> already had \\boxed{{}}: {counts['kept']}")
    print(f">>> skipped (no answer):    {counts['skipped']}")
    print(f">>> kept (in output):       {len(out_rows)} rows -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
