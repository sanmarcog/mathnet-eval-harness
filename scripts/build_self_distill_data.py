"""Build a self-distilled training set: run base Qwen3-1.7B on training
problems with thinking-on, keep only those where the model produced a
correct \\boxed{} answer, save as a new train JSONL where
solutions_markdown is the model's own reasoning trace.

Used for Run 4 (self-distillation experiment). Per [Why Does
Self-Distillation Degrade Reasoning (arxiv 2603.24472)] the key thing is
to PRESERVE the long reasoning trace — don't strip thinking before saving.

    python scripts/build_self_distill_data.py \\
        --source-jsonl data/splits/train_multilingual_filtered.jsonl \\
        --base-model Qwen/Qwen3-1.7B \\
        --out-jsonl data/splits/train_self_distill.jsonl \\
        --n-attempt 200 \\
        --max-new-tokens 16384 \\
        --skip-existing

`--skip-existing` lets the pilot's 200-row generation be reused by the
full Run 4's 1350-row generation: only the additional rows are run.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

# Match eval-time prompt; same SYSTEM_PROMPT_BOXED used by eval_qwen_hf.py
SYSTEM_PROMPT_BOXED = (
    "You are an expert mathematician. Solve the following olympiad problem. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
LEADING_THINK_RE = re.compile(r"^\s*<think>\s*", re.DOTALL)
TRAILING_IM_END_RE = re.compile(r"<\|im_end\|>\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-jsonl", required=True, type=Path)
    p.add_argument("--base-model", required=True)
    p.add_argument("--out-jsonl", required=True, type=Path)
    p.add_argument("--n-attempt", type=int, default=200,
                   help="Max problems to attempt from the head of source")
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true",
                   help="If out-jsonl exists, skip already-attempted ids and append new ones")
    return p.parse_args()


def normalize_assistant_content(raw: str) -> str:
    """The vLLM output may include the <think> opener (because we set
    skip_special_tokens=False) and a trailing <|im_end|>. Strip those so
    the chat template's enable_thinking=True can add its own <think>\\n
    opener and <|im_end|> closer cleanly when this content is rendered as
    the assistant turn at training time."""
    s = LEADING_THINK_RE.sub("", raw, count=1)
    s = TRAILING_IM_END_RE.sub("", s)
    return s


def main() -> int:
    args = parse_args()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Late imports so that `--help` works without these heavy deps.
    from mathnet_eval.grading import extract_answer, normalize_for_exact, symbolic_equal

    tok = AutoTokenizer.from_pretrained(args.base_model)

    # Read source rows
    rows = [json.loads(l) for l in args.source_jsonl.open()][: args.n_attempt]
    print(f">>> source rows considered: {len(rows)}")

    # Skip-existing: read IDs already in the out file
    already_done: set[str] = set()
    existing_kept: list[dict] = []
    if args.skip_existing and args.out_jsonl.exists():
        with args.out_jsonl.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("id"):
                    already_done.add(r["id"])
                    existing_kept.append(r)
        print(f">>> skip-existing: {len(already_done)} ids already attempted, will keep them")

    pending = [r for r in rows if r.get("id") not in already_done]
    print(f">>> {len(pending)} new problems to attempt")
    if not pending:
        print(">>> nothing new to do.")
        return 0

    # Build prompts with thinking-on
    prompts = []
    for r in pending:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_BOXED},
            {"role": "user",   "content": r["problem_markdown"].strip()},
        ]
        try:
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
        except TypeError:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # vLLM, recommended sampling, thinking-on. skip_special_tokens=False so
    # we keep <think> tokens in raw output (we strip them ourselves on save).
    print(f">>> loading {args.base_model}")
    llm = LLM(model=args.base_model, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.9,
              max_model_len=max(args.max_new_tokens + 1024, 8192))
    sp = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        max_tokens=args.max_new_tokens, seed=args.seed,
        skip_special_tokens=False,
    )

    print(f">>> running base on {len(prompts)} prompts (thinking-on, vLLM batched)")
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    wall = time.perf_counter() - t0
    print(f">>> base inference done in {wall/60:.1f} min "
          f"({wall/max(len(prompts),1):.1f}s/problem avg)")

    # Filter for correct
    new_kept: list[dict] = []
    n_attempt = 0
    n_extracted = 0
    n_correct = 0
    for r, out in zip(pending, outputs):
        n_attempt += 1
        raw_text = out.outputs[0].text
        # strip thinking for answer extraction
        post_think = THINK_RE.sub("", raw_text).strip()
        pred = extract_answer(post_think) or extract_answer(raw_text)
        gold = (r.get("final_answer") or "").strip()
        if pred is None or not gold:
            continue
        n_extracted += 1
        is_correct = (
            pred.strip() == gold
            or normalize_for_exact(pred) == normalize_for_exact(gold)
            or symbolic_equal(pred, gold)
        )
        if not is_correct:
            continue
        n_correct += 1

        # Save assistant content with leading <think> stripped (chat template
        # will re-add it when training renders this row with
        # add_generation_prompt=False, enable_thinking=True).
        assistant_content = normalize_assistant_content(raw_text)
        new_row = dict(r)
        new_row["solutions_markdown"] = [assistant_content]
        new_kept.append(new_row)

    # Write all kept rows (existing + new) to out
    all_kept = existing_kept + new_kept
    with args.out_jsonl.open("w") as f:
        for r in all_kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f">>> attempted: {n_attempt}, extracted any answer: {n_extracted}, "
          f"correct: {n_correct} ({n_correct/max(n_attempt,1)*100:.1f}%)")
    print(f">>> total rows in {args.out_jsonl}: {len(all_kept)} "
          f"({len(existing_kept)} prior + {n_correct} new)")

    # Quick sanity dump on first new kept row
    if new_kept:
        sample = new_kept[0]
        sol = sample["solutions_markdown"][0]
        print()
        print(f">>> sanity-check on first new kept row (id={sample.get('id')}):")
        print(f"    gold: {sample.get('final_answer','')[:80]!r}")
        print(f"    assistant content head (300 chars): {sol[:300]!r}")
        print(f"    contains </think>: {'</think>' in sol}")
        print(f"    contains \\boxed{{}}: {chr(92)+'boxed{' in sol}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
