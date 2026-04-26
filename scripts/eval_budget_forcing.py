"""Budget-forcing eval, adapted from s1 (simplescaling/s1). When the model
emits its thinking-end token (</think> for Qwen3), we don't accept the
stop — we append the literal string 'Wait' to the prompt and re-call
generate. This forces the model to keep reasoning, surfacing more of its
latent capability before committing to a final answer.

Per the s1 paper, this is the technique that lets a small SFT'd model
trade test-time compute for accuracy along a Pareto frontier.

    python scripts/eval_budget_forcing.py \\
        --base-model /path/to/merged_run2 \\
        --split data/splits/eval.jsonl \\
        --out results/full/qwen3-1.7b-run2-bf \\
        --num-ignore 1 \\
        --max-tokens-thinking 16000 \\
        --max-tokens-answer 1024

The 3-phase algorithm (s1 recipe):
  1. Initial thinking with `stop_token_ids=[</think>]` and budget
     `max_tokens_thinking`. Stop at </think> or hit budget.
  2. Wait-loop (NUM_IGNORE iterations): append "Wait" to the accumulated
     output, re-generate with the remaining thinking budget, repeat.
  3. Final-answer phase with no stop tokens; let the model commit.

Per-problem JSON has the same shape as eval_qwen_hf so grade_results.py
consumes it. Sibling 'bf' block records the per-phase token counts.

Pre-check protocol (do this manually first, on a handful of problems):
verify Qwen3 actually picks up the 'Wait' cue and continues reasoning
rather than echoing a final answer or repeating itself. If the model
doesn't pick it up, BF won't work cleanly without fine-tuning data that
exhibits the pattern.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

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
WAIT_STRING = "\nWait"
THINKING_END_LITERAL = "</think>"
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--num-ignore", type=int, default=1,
                   help="Number of times to append 'Wait' and re-generate when "
                        "the model tries to end thinking.")
    p.add_argument("--max-tokens-thinking", type=int, default=16000,
                   help="Total budget across the thinking + Wait passes.")
    p.add_argument("--max-tokens-answer", type=int, default=1024,
                   help="Max tokens for the final-answer phase after thinking.")
    p.add_argument("--n", type=int, default=None,
                   help="Limit to first N problems (debugging).")
    p.add_argument("--model-alias", default=None)
    p.add_argument("--prompt-format", choices=("final-answer", "boxed"), default="boxed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    system_prompt = (SYSTEM_PROMPT_BOXED if args.prompt_format == "boxed"
                     else SYSTEM_PROMPT_FINAL_ANSWER)
    model_id = args.model_alias or args.base_model.rstrip("/").split("/")[-1].lower()

    # Find the </think> token id. Qwen3 emits this as a single special token.
    think_end_ids = tokenizer.encode(THINKING_END_LITERAL, add_special_tokens=False)
    print(f">>> </think> tokenizes to ids: {think_end_ids}")
    if len(think_end_ids) != 1:
        print(f"WARNING: </think> is not a single token in this tokenizer. "
              f"stop_token_ids may not catch it cleanly; budget-forcing may misbehave.")

    print(f">>> [vLLM BF] loading {args.base_model}  num_ignore={args.num_ignore}  "
          f"max_thinking={args.max_tokens_thinking}  max_answer={args.max_tokens_answer}")
    # max_model_len = thinking + Wait extensions + answer + prompt headroom
    max_model_len = args.max_tokens_thinking + args.max_tokens_answer + 1024
    llm = LLM(model=args.base_model, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.9, max_model_len=max_model_len)

    # Build the initial prompts (with thinking enabled).
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

    initial_prompts = []
    for p in to_run:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": p["problem_markdown"].strip()},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        initial_prompts.append(prompt)

    # === Phase 1: initial thinking, stop at </think> ===
    t0 = time.perf_counter()
    sp_think = SamplingParams(
        max_tokens=args.max_tokens_thinking,
        temperature=0.0,
        stop_token_ids=think_end_ids,
        skip_special_tokens=False,
        min_tokens=0,
        seed=args.seed,
    )
    print(f">>> phase 1: initial thinking on {len(initial_prompts)} prompts")
    phase1_outputs = llm.generate(initial_prompts, sp_think)

    # Track per-problem state through the Wait loop.
    state = []
    for p, prompt, out in zip(to_run, initial_prompts, phase1_outputs):
        text = out.outputs[0].text
        n_tok = len(out.outputs[0].token_ids)
        finish_reason = out.outputs[0].finish_reason
        state.append({
            "p": p,
            "prompt": prompt,
            "thinking_text": text,
            "phase1_tokens": n_tok,
            "wait_tokens": [],          # tokens added per Wait pass
            "finish_reasons": [finish_reason],
        })

    # === Phase 2: Wait loop ===
    for it in range(args.num_ignore):
        # Build new prompts only for problems that hit </think> (i.e., have
        # remaining thinking budget). If the model maxed out without </think>,
        # appending Wait may still help but that's optional — we do it anyway
        # for parity with s1.
        new_prompts = []
        active_idx = []
        for i, s in enumerate(state):
            tokens_used = s["phase1_tokens"] + sum(s["wait_tokens"])
            remaining = args.max_tokens_thinking - tokens_used
            if remaining <= 100:
                continue
            # The accumulated output so far + "Wait" + (let it continue thinking)
            extended = s["prompt"] + s["thinking_text"] + WAIT_STRING
            for w in s["wait_tokens"]:
                # already accumulated wait extensions
                pass
            new_prompts.append(extended)
            active_idx.append(i)
        if not new_prompts:
            break
        print(f">>> phase 2 wait pass {it+1}/{args.num_ignore}: {len(new_prompts)} active prompts")
        # Use the remaining smallest-budget conservatively; vLLM bounds per-seq
        sp_wait = SamplingParams(
            max_tokens=args.max_tokens_thinking,
            temperature=0.0,
            stop_token_ids=think_end_ids,
            skip_special_tokens=False,
            min_tokens=0,
            seed=args.seed,
        )
        outs = llm.generate(new_prompts, sp_wait)
        for j, o in zip(active_idx, outs):
            wt = o.outputs[0].text
            n_tok = len(o.outputs[0].token_ids)
            state[j]["thinking_text"] = state[j]["thinking_text"] + WAIT_STRING + wt
            state[j]["wait_tokens"].append(n_tok)
            state[j]["finish_reasons"].append(o.outputs[0].finish_reason)

    # === Phase 3: final-answer commit ===
    # Append </think> to close the thinking block (so the chat template's
    # answer phase begins), then let the model generate freely.
    final_prompts = []
    for s in state:
        # If thinking_text already ended at </think> (vLLM stopped there), the
        # tokenizer represents that visually as the literal string. We append
        # </think>\n\n to make the answer phase clean.
        closing = "" if s["thinking_text"].rstrip().endswith(THINKING_END_LITERAL) else THINKING_END_LITERAL
        final_prompts.append(s["prompt"] + s["thinking_text"] + closing + "\n\n")

    sp_answer = SamplingParams(
        max_tokens=args.max_tokens_answer,
        temperature=0.0,
        skip_special_tokens=False,
        min_tokens=0,
        seed=args.seed,
    )
    print(f">>> phase 3: final answer commit on {len(final_prompts)} prompts")
    phase3_outputs = llm.generate(final_prompts, sp_answer)
    for s, out in zip(state, phase3_outputs):
        s["answer_text"] = out.outputs[0].text
        s["answer_tokens"] = len(out.outputs[0].token_ids)
        s["finish_reasons"].append(out.outputs[0].finish_reason)

    wall = time.perf_counter() - t0
    print(f">>> BF run done in {wall/60:.1f}min for {len(state)} problems "
          f"({wall/max(len(state),1):.1f}s/problem avg)")

    # === Write per-problem JSONs ===
    total_in = total_out = 0
    for s in state:
        p = s["p"]
        # Reconstruct the full response_text the way grade_results.py expects:
        # the model's full output — both the (extended) thinking and the answer.
        thinking_block = "<think>\n" + s["thinking_text"]
        if not s["thinking_text"].rstrip().endswith(THINKING_END_LITERAL):
            thinking_block += THINKING_END_LITERAL
        full_raw = thinking_block + "\n\n" + s["answer_text"]
        full_visible = _strip_thinking(full_raw)
        out_tokens = s["phase1_tokens"] + sum(s["wait_tokens"]) + s["answer_tokens"]
        total_out += out_tokens
        # input tokens approximate (prompt token count after chat template)
        total_in += len(tokenizer.encode(s["prompt"], add_special_tokens=False))

        record = {
            "id": p["id"],
            "country": p.get("country"),
            "competition": p.get("competition"),
            "gold_final_answer": p.get("final_answer"),
            "topics_flat": p.get("topics_flat"),
            "prompt": p["problem_markdown"].strip(),
            "model": model_id,
            "response_text": full_visible,
            "raw_response_text": full_raw if full_raw != full_visible else None,
            "usage": {
                "input_tokens": len(tokenizer.encode(s["prompt"], add_special_tokens=False)),
                "output_tokens": out_tokens,
            },
            "latency_s": None,
            "cached": False,
            "bf": {
                "num_ignore": args.num_ignore,
                "phase1_tokens": s["phase1_tokens"],
                "wait_tokens": s["wait_tokens"],
                "answer_tokens": s["answer_tokens"],
                "finish_reasons": s["finish_reasons"],
            },
        }
        (args.out / f"{p['id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2))

    summary = {
        "model": model_id,
        "base_model": args.base_model,
        "adapter": None,
        "prompt_format": args.prompt_format,
        "sampling": "budget-forcing",
        "num_ignore": args.num_ignore,
        "max_tokens_thinking": args.max_tokens_thinking,
        "max_tokens_answer": args.max_tokens_answer,
        "enable_thinking": True,
        "seed": args.seed,
        "backend": "vllm-bf",
        "split": str(args.split),
        "n_problems": len(problems),
        "n_scored_fresh": len(state),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "elapsed_s": wall,
        "errors": [],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f">>> done. in={total_in} out={total_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
