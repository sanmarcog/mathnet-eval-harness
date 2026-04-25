"""Best-of-N inference: sample N completions per problem, extract a final
answer from each, and pick the majority-voted answer as the model's
output. Free pp at inference time, no retraining needed.

Loosely follows [Wang et al. 2022, Self-Consistency]
(https://arxiv.org/abs/2203.11171). Reads the same problem JSONL as
`eval_qwen_hf.py` and writes the same per-problem JSON shape so
`grade_results.py` consumes it without changes — the `response_text`
field is the winning completion's text, and a sibling `bon` block
records the N samples and the vote tally.

    python scripts/eval_best_of_n.py \\
        --base-model /path/to/merged_run2 \\
        --split data/splits/eval.jsonl \\
        --out results/full/qwen3-1.7b-run2-bon8 \\
        --n-samples 8 \\
        --prompt-format boxed --enable-thinking \\
        --max-new-tokens 16384

Notes:
- Sampling defaults to Qwen3 math-recommended (temp=0.6, top_p=0.95,
  top_k=20, min_p=0). Greedy bon is degenerate; use temperature.
- Vote tally is on the *normalized* extracted answer. Ties broken by
  vote count then by first-seen order; we record the tie in the json.
- Seeds: each of the N completions for a single prompt uses different
  internal seeds (vLLM handles this); the run-level seed parameter is
  for reproducibility of the prompt order.
- If a sample fails to produce an extractable answer, it doesn't vote.
  If ALL N fail, the record's `response_text` is the longest sample's
  text (so the grader can still flag it as miss).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from mathnet_eval.grading import extract_answer, normalize_for_exact

DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
SYSTEM_PROMPT_BOXED = (
    "You are an expert mathematician. Solve the following olympiad problem. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)
SYSTEM_PROMPT_FINAL_ANSWER = (
    "You are an expert mathematician. Solve the following olympiad problem. "
    "Show your work briefly, then put the final answer on its own line in the form: "
    "Final answer: <answer>."
)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                   help="HF model id or local path to a merged checkpoint.")
    p.add_argument("--n-samples", type=int, default=8,
                   help="Number of completions to draw per problem.")
    p.add_argument("--n", type=int, default=None,
                   help="Limit to first N problems from the split (debugging).")
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--model-alias", default=None)
    p.add_argument("--prompt-format", choices=("final-answer", "boxed"), default="boxed")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _vote(samples: list[str]) -> tuple[str, int, dict[str, int]]:
    """Majority-vote on the normalized extracted answer across N samples.
    Returns (winning_normalized_answer, vote_count, full_tally).
    Empty string if no sample produced an extractable answer."""
    norm_to_count: Counter[str] = Counter()
    for text in samples:
        ans = extract_answer(text)
        if ans is None:
            continue
        norm = normalize_for_exact(ans)
        if not norm:
            continue
        norm_to_count[norm] += 1
    if not norm_to_count:
        return "", 0, {}
    winner, count = norm_to_count.most_common(1)[0]
    return winner, count, dict(norm_to_count)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    system_prompt = (SYSTEM_PROMPT_BOXED if args.prompt_format == "boxed"
                     else SYSTEM_PROMPT_FINAL_ANSWER)
    model_id = args.model_alias or args.base_model.rstrip("/").split("/")[-1].lower()

    sp = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=0.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    print(f">>> [vLLM BoN] loading {args.base_model}  n_samples={args.n_samples}  "
          f"temp={args.temperature}  thinking={args.enable_thinking}  seed={args.seed}")
    llm = LLM(model=args.base_model, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.9,
              max_model_len=max(args.max_new_tokens + 1024, 8192))

    problems = [json.loads(l) for l in args.split.read_text().splitlines() if l.strip()]
    if args.n is not None:
        problems = problems[: args.n]

    to_run = []
    for p in problems:
        out_path = args.out / f"{p['id']}.json"
        if args.skip_existing and out_path.exists():
            try:
                prev = json.loads(out_path.read_text())
                if prev.get("response_text") is not None:
                    continue
            except Exception:
                pass
        to_run.append(p)
    print(f">>> {len(to_run)} problems to generate (skipped {len(problems) - len(to_run)})")

    prompts = []
    for p in to_run:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": p["problem_markdown"].strip()},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    wall = time.perf_counter() - t0
    print(f">>> vLLM BoN done in {wall/60:.1f}min for {len(prompts)} prompts × N={args.n_samples} "
          f"= {len(prompts)*args.n_samples} completions ({wall/max(len(prompts),1):.1f}s/problem avg)")

    total_in = total_out = 0
    for p, out in zip(to_run, outputs):
        pid = p["id"]
        sample_texts = []
        sample_raw_texts = []
        sample_tokens = []
        for s in out.outputs:
            sample_raw_texts.append(s.text)
            sample_texts.append(_strip_thinking(s.text))
            sample_tokens.append(len(s.token_ids))
            total_out += len(s.token_ids)
        input_tokens = len(out.prompt_token_ids)
        total_in += input_tokens

        winner_norm, winner_count, tally = _vote(sample_texts)

        # pick the response_text: a sample whose normalized answer matches the
        # winner. If no sample produced any answer, fall back to the longest.
        if winner_norm:
            chosen_idx = next(
                (i for i, t in enumerate(sample_texts)
                 if (a := extract_answer(t)) is not None
                 and normalize_for_exact(a) == winner_norm),
                0,
            )
        else:
            chosen_idx = max(range(len(sample_texts)),
                             key=lambda i: sample_tokens[i])

        record = {
            "id": pid,
            "country": p.get("country"),
            "competition": p.get("competition"),
            "gold_final_answer": p.get("final_answer"),
            "topics_flat": p.get("topics_flat"),
            "prompt": p["problem_markdown"].strip(),
            "model": model_id,
            "response_text": sample_texts[chosen_idx],
            "raw_response_text": (sample_raw_texts[chosen_idx]
                                  if sample_raw_texts[chosen_idx] != sample_texts[chosen_idx]
                                  else None),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": sample_tokens[chosen_idx],
                "bon_total_output_tokens": sum(sample_tokens),
            },
            "latency_s": None,
            "cached": False,
            "bon": {
                "n_samples": args.n_samples,
                "winner_normalized": winner_norm,
                "winner_votes": winner_count,
                "tally": tally,
                "all_samples": sample_texts,
            },
        }
        (args.out / f"{pid}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))

    summary = {
        "model": model_id,
        "base_model": args.base_model,
        "adapter": None,
        "prompt_format": args.prompt_format,
        "sampling": "bon",
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
        "backend": "vllm-bon",
        "split": str(args.split),
        "n_problems": len(problems),
        "n_scored_fresh": len(to_run),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "elapsed_s": wall,
        "errors": [],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f">>> done: {len(to_run)} problems × {args.n_samples} samples. "
          f"in={total_in} out={total_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
