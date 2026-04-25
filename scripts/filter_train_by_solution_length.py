"""Filter a train JSONL by minimum solution length (in tokens) so that
near-zero-signal rows (empty `solutions_markdown` falling back to bare
`\\boxed{X}`) don't dominate the gradient. The verify_response_template
diagnostic surfaced these rows on the multilingual augmentation: ~14% of
14,585 multilingual rows had no solution text at all, and another ~6% had
near-empty solutions (<100 tokens of derivation).

    python scripts/filter_train_by_solution_length.py \\
        --in  data/splits/train_multilingual.jsonl \\
        --out data/splits/train_multilingual_filtered.jsonl \\
        --tokenizer Qwen/Qwen3-1.7B \\
        --min-tokens 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def get_solution(row: dict) -> str:
    s = row.get("solutions_markdown")
    if isinstance(s, list):
        s = s[0] if s else ""
    return (s or "").strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="in_path",  required=True, type=Path)
    p.add_argument("--out", dest="out_path", required=True, type=Path)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--min-tokens", type=int, default=100)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = [json.loads(l) for l in args.in_path.open()]
    print(f">>> read {len(rows)} rows from {args.in_path}")

    kept: list[dict] = []
    dropped_empty = 0
    dropped_short = 0
    for r in rows:
        sol = get_solution(r)
        if not sol:
            dropped_empty += 1
            continue
        n = len(tok.encode(sol, add_special_tokens=False))
        if n < args.min_tokens:
            dropped_short += 1
            continue
        kept.append(r)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f">>> dropped: empty={dropped_empty}  short(<{args.min_tokens} tok)={dropped_short}")
    print(f">>> kept: {len(kept)} ({len(kept)/len(rows)*100:.1f}%) -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
