"""Run a local Qwen-2.5-1.5B model (base or with LoRA adapter) across an
eval split and save per-problem JSONs in the same format as the frontier
API runs, so `grade_results.py` can process them uniformly.

Usage (inside GPU sbatch):
    # Base model
    python scripts/eval_qwen_hf.py --split data/splits/eval.jsonl \
        --out results/full/qwen-base

    # With adapter
    python scripts/eval_qwen_hf.py --split data/splits/eval.jsonl \
        --out results/full/qwen-mathnet-run1 \
        --adapter /gscratch/scrubbed/sanmarco/adapters/qwen-mathnet-run1
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT_FINAL_ANSWER = (
    "You are an expert mathematician. Solve the following olympiad problem. "
    "Show your work briefly, then put the final answer on its own line in the form: "
    "Final answer: <answer>."
)
SYSTEM_PROMPT_BOXED = (
    "You are an expert mathematician. Solve the following olympiad problem. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                   help="HF model id. Default: Qwen/Qwen2.5-1.5B-Instruct. Set to Qwen/Qwen3-1.7B for Run-2.")
    p.add_argument("--adapter", type=Path, default=None,
                   help="Path to a PEFT adapter dir. Omit for base model.")
    p.add_argument("--n", type=int, default=None, help="Limit to first N problems.")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--model-alias", default=None,
                   help="Label for the `model` field in each JSON.")
    p.add_argument("--prompt-format", choices=("final-answer", "boxed"), default="final-answer",
                   help="'final-answer' matches our Qwen 2.5 / training convention; "
                        "'boxed' matches Qwen3 math-recommended prompting.")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Enable Qwen3 thinking mode in chat template (ignored by Qwen2.5).")
    p.add_argument("--sampling", choices=("greedy", "recommended"), default="greedy",
                   help="'greedy' = deterministic; 'recommended' = Qwen3 math-recommended "
                        "(temp=0.6, top_p=0.95, top_k=20, min_p=0).")
    p.add_argument("--seed", type=int, default=0, help="Manual seed for torch / sampling reproducibility.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip problems that already have a {id}.json in --out dir (enables cheap restart).")
    return p.parse_args()


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from a Qwen3 thinking-mode response.
    Safe no-op on Qwen 2.5 output (no such blocks)."""
    return _THINK_RE.sub("", text).strip()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(args.seed)

    print(f">>> loading tokenizer + 4-bit base: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )

    if args.adapter:
        from peft import PeftModel
        print(f">>> loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, str(args.adapter))
        model_id = args.model_alias or args.adapter.name
    else:
        model_id = args.model_alias or args.base_model.split("/")[-1].lower()

    model.eval()

    system_prompt = SYSTEM_PROMPT_BOXED if args.prompt_format == "boxed" else SYSTEM_PROMPT_FINAL_ANSWER

    # Sampling config.
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "pad_token_id": tokenizer.eos_token_id}
    if args.sampling == "recommended":
        # Qwen3 math-recommended settings (temp 0.6, top_p 0.95, top_k 20, min_p 0).
        gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
    else:
        gen_kwargs.update(do_sample=False)

    print(f">>> prompt-format={args.prompt_format}  sampling={args.sampling}  enable_thinking={args.enable_thinking}  seed={args.seed}")

    problems = [json.loads(l) for l in args.split.read_text().splitlines() if l.strip()]
    if args.n is not None:
        problems = problems[: args.n]
    print(f">>> running {model_id} on {len(problems)} problems")

    total_in = total_out = 0
    n_errors = n_skipped = 0
    errors: list[dict] = []
    t_start = time.perf_counter()

    for i, p in enumerate(problems):
        pid = p["id"]
        out_path = args.out / f"{pid}.json"

        if args.skip_existing and out_path.exists():
            try:
                prev = json.loads(out_path.read_text())
                if prev.get("response_text") is not None:
                    total_in += prev.get("usage", {}).get("input_tokens", 0)
                    total_out += prev.get("usage", {}).get("output_tokens", 0)
                    n_skipped += 1
                    if (i + 1) % 50 == 0:
                        print(f"  [{i+1}/{len(problems)}] id={pid} (skipped; existing)")
                    continue
            except Exception:
                pass  # broken prior file -> re-run it

        user_prompt = p["problem_markdown"].strip()

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            latency = time.perf_counter() - t0

            input_tokens = int(inputs["input_ids"].shape[1])
            output_tokens = int(out.shape[1] - input_tokens)
            raw_text = tokenizer.decode(out[0][input_tokens:], skip_special_tokens=True)
            # Strip <think>...</think> before storing so the grader extracts from the final answer, not the thinking trace.
            text = _strip_thinking(raw_text)
        except Exception as e:
            n_errors += 1
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
            "model": model_id,
            "response_text": text,
            "raw_response_text": raw_text if raw_text != text else None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "latency_s": latency,
            "cached": False,
        }
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        total_in += input_tokens; total_out += output_tokens

        if (i + 1) == 1 or (i + 1) % 20 == 0 or (i + 1) == len(problems):
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(problems)}] id={pid}  lat={latency:.1f}s  in={input_tokens} out={output_tokens}  (total {elapsed/60:.1f}min)")

    elapsed = time.perf_counter() - t_start
    summary = {
        "model": model_id,
        "base_model": args.base_model,
        "adapter": str(args.adapter) if args.adapter else None,
        "prompt_format": args.prompt_format,
        "sampling": args.sampling,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
        "split": str(args.split),
        "n_problems": len(problems),
        "n_errors": n_errors,
        "n_skipped_existing": n_skipped,
        "cached_hits": 0,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "elapsed_s": elapsed,
        "errors": errors,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\ndone in {elapsed/60:.1f}min: {len(problems) - n_errors - n_skipped} ok, {n_skipped} skipped, {n_errors} errors.  tokens in={total_in} out={total_out}  summary -> {args.out}/summary.json")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
