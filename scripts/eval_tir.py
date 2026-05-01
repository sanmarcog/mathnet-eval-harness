"""Week-2 TIR + RAG-of-TIR eval driver.

One script, three modes (`cot`, `tir`, `tir_rag`), one ``--smoke`` flag
that exercises the full pipeline on CPU with a tiny model. The smoke is
load-bearing: per the Dr. GRPO lesson, dep-hell + queue-cycle costs make
fail-on-login-node cheaper than fail-on-A40-after-30min-queue-wait.

Modes:
    cot      — single-shot generation, extract \\boxed answer
    tir      — TIR loop with Python sandbox, up to 4 tool calls
    tir_rag  — TIR loop + retrieved exemplars prepended to user message

Smoke vs production:
    --smoke   N=4, CPU, Qwen2.5-0.5B-Instruct, no judge, fixture bank,
              max 1 tool call per problem, exits non-zero on any
              uncaught exception, writes a sentinel file on success
              (logs/tir_smoke_passed_<commit>.txt).
    (default) N=500, GPU/vLLM, Qwen2.5-Math-1.5B-Instruct, judge enabled.

Usage:
    # smoke (run on Mac / login node):
    python scripts/eval_tir.py --mode tir_rag --smoke

    # production (sbatch):
    python scripts/eval_tir.py --mode tir_rag \\
        --eval-jsonl data/splits/eval.jsonl \\
        --bank results/tir/exemplar_bank.jsonl \\
        --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --backend vllm \\
        --use-judge \\
        --out results/tir/tir_rag/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Make `src/mathnet_eval/...` importable when this script is invoked
# directly (matches the convention of other scripts/ entrypoints).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathnet_eval.grading import grade  # noqa: E402
from mathnet_eval.retrieval import build_retriever, load_bank  # noqa: E402
from mathnet_eval.tir import (  # noqa: E402
    PythonSandbox,
    TIRRunner,
    extract_boxed_answer,
)
from mathnet_eval.tir_prompts import (  # noqa: E402
    COT_SYSTEM,
    TIR_RAG_SYSTEM,
    TIR_SYSTEM,
    format_cot_user,
    format_tir_rag_user,
    format_tir_user,
)


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

def with_smoke_tool_canary(real_generate_fn):
    """Wrap a generate_fn so the FIRST generation per problem returns a
    canned `````python ... ````` block. Smoke-only.

    Why: smoke runs against a tiny stand-in model (Qwen2.5-0.5B-Instruct on
    CPU) that is too small to reliably follow the TIR prompt convention.
    Without this, smoke never emits a tool call, so the sandbox + tool-
    output-feedback path stays untested. The canary forces the path to
    fire on every problem; the real model (Qwen2.5-Math-1.5B-Instruct in
    production) doesn't need this and runs without it.

    The canary fires once per *generation budget reset*, which the runner
    tracks via the prompt_so_far it passes in. We detect "first call for
    this problem" by checking whether prompt_so_far ends with the assistant
    turn opener (no generated text yet)."""
    def wrapped(prompt: str, max_new_tokens: int, stop_strings: list[str]) -> str:
        # Heuristic: if the prompt ends with the assistant-turn opener and
        # no generated content yet, this is the first call. We give it a
        # canned tool call so the sandbox path is exercised.
        if prompt.rstrip().endswith("<|im_start|>assistant"):
            return "I'll compute the answer using Python.\n```python\nprint(2)\n```\n"
        return real_generate_fn(prompt, max_new_tokens, stop_strings)

    return wrapped


def make_hf_backend(model_name: str, device: str = "cpu"):
    """Return (generate_fn, tokenizer). Lazy-imports transformers/torch so
    `--mode cot` smoke can be edited without paying the import cost up
    front (HF imports are slow)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch  # type: ignore

    print(f"[backend=hf] loading {model_name} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    ).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def generate_fn(prompt: str, max_new_tokens: int, stop_strings: list[str]) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        for s in stop_strings:
            if s in new_text:
                cut = new_text.index(s) + len(s)
                new_text = new_text[:cut]
                break
        return new_text

    return generate_fn, tokenizer


def make_vllm_backend(model_name: str, max_model_len: int = 8192):
    """vLLM backend for production. One-prompt-at-a-time semantics so the
    TIR loop can iterate per-problem; vLLM's scheduling overhead on a
    warm engine is small enough that this is acceptable for the n=500
    eval. Batched per-step generation is a follow-up optimization."""
    from transformers import AutoTokenizer  # type: ignore
    from vllm import LLM, SamplingParams  # type: ignore

    print(f"[backend=vllm] loading {model_name}  max_model_len={max_model_len}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
    )

    def generate_fn(prompt: str, max_new_tokens: int, stop_strings: list[str]) -> str:
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            seed=0,
            stop=stop_strings or None,
            include_stop_str_in_output=True,
        )
        out = llm.generate([prompt], sampling_params=sp, use_tqdm=False)
        return out[0].outputs[0].text

    return generate_fn, tokenizer


# ---------------------------------------------------------------------------
# Per-problem evaluation
# ---------------------------------------------------------------------------

def build_messages(
    mode: str,
    problem: str,
    exemplars: list[dict] | None,
    exemplar_type: str = "tir",
) -> list[dict]:
    if mode == "cot":
        return [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": format_cot_user(problem)},
        ]
    if mode == "tir":
        return [
            {"role": "system", "content": TIR_SYSTEM},
            {"role": "user", "content": format_tir_user(problem)},
        ]
    if mode == "tir_rag":
        assert exemplars is not None
        return [
            {"role": "system", "content": TIR_RAG_SYSTEM},
            {"role": "user", "content": format_tir_rag_user(problem, exemplars, exemplar_type=exemplar_type)},
        ]
    raise ValueError(f"unknown mode: {mode}")


def eval_one(
    *,
    row: dict,
    mode: str,
    tokenizer,
    generate_fn,
    sandbox: PythonSandbox | None,
    retriever,
    k: int,
    max_new_tokens_total: int,
    max_tool_calls: int,
    use_judge: bool,
    exemplar_type: str = "tir",
) -> dict:
    problem = row["problem_markdown"].strip()
    gold = (row.get("final_answer") or "").strip()

    exemplars: list[dict] | None = None
    retrieved_ids: list[str] = []
    if mode == "tir_rag":
        if hasattr(retriever, "retrieve") and retriever.__class__.__name__ == "TopicRetriever":
            exemplars = retriever.retrieve(problem, k=k, query_topics_flat=row.get("topics_flat"))
        else:
            exemplars = retriever.retrieve(problem, k=k)
        retrieved_ids = [ex["id"] for ex in exemplars]

    messages = build_messages(mode, problem, exemplars, exemplar_type=exemplar_type)
    prompt_prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    t0 = time.time()
    if mode == "cot":
        new_text = generate_fn(prompt_prefix, max_new_tokens_total, ["<|im_end|>"])
        full_text = new_text
        n_tool_calls = 0
        saturated = False
        finished = "\\boxed{" in new_text
        final_answer = extract_boxed_answer(new_text)
    else:
        runner = TIRRunner(
            sandbox=sandbox,
            generate_fn=generate_fn,
            max_tool_calls=max_tool_calls,
            max_new_tokens_per_step=min(1024, max_new_tokens_total),
            max_new_tokens_total=max_new_tokens_total,
        )
        trace = runner.run(prompt_prefix)
        full_text = trace.full_text
        n_tool_calls = trace.n_tool_calls
        saturated = trace.saturated
        finished = trace.finished
        final_answer = trace.final_answer
    elapsed = time.time() - t0

    g = grade(full_text, gold, problem=problem, use_judge=use_judge)

    return {
        "id": row["id"],
        "country": row.get("country"),
        "competition": row.get("competition"),
        "topics_flat": row.get("topics_flat"),
        "gold_final_answer": gold,
        "mode": mode,
        "response_text": full_text,
        "n_tool_calls": n_tool_calls,
        "saturated": saturated,
        "finished": finished,
        "extracted_final_answer": final_answer,
        "retrieved_exemplar_ids": retrieved_ids,
        "elapsed_s": round(elapsed, 3),
        "grade": {
            "correct": g.correct,
            "method": g.method,
            "predicted": g.predicted,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["cot", "tir", "tir_rag"], required=True)
    p.add_argument("--smoke", action="store_true",
                   help="CPU-only path with a tiny model, fixture bank, N=4, "
                        "no judge. Exits non-zero on any exception.")
    p.add_argument("--n", type=int, default=None, help="problems to score; default 4 in smoke, all in production")
    p.add_argument("--eval-jsonl", default="data/splits/eval.jsonl")
    p.add_argument("--bank", default=None, help="JSONL exemplar bank (TIR-RAG mode only)")
    p.add_argument("--model", default=None)
    p.add_argument("--backend", choices=["hf", "vllm"], default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--retrieval-policy", choices=["bm25", "dense", "topic"], default="bm25")
    p.add_argument("--k", type=int, default=3, help="exemplars to retrieve per problem")
    p.add_argument("--max-new-tokens", type=int, default=4096, help="total budget per problem")
    p.add_argument("--max-tool-calls", type=int, default=4)
    p.add_argument("--use-judge", action="store_true")
    p.add_argument("--out", default=None)
    return p.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    """Override args with smoke-friendly defaults — kept in one place so
    smoke vs production divergence is auditable."""
    if not args.smoke:
        return
    if args.n is None:
        args.n = 4
    if args.model is None:
        args.model = "Qwen/Qwen2.5-0.5B-Instruct"
    if args.backend is None:
        args.backend = "hf"
    if args.bank is None:
        args.bank = "tests/tir_smoke_exemplar_bank.jsonl"
    if args.out is None:
        args.out = "results/tir/smoke"
    # Tighten the loop so smoke finishes in a few minutes on a Mac CPU.
    args.max_new_tokens = min(args.max_new_tokens, 256)
    args.max_tool_calls = min(args.max_tool_calls, 1)
    args.use_judge = False


def apply_production_defaults(args: argparse.Namespace) -> None:
    if args.smoke:
        return
    if args.n is None:
        args.n = -1  # all
    if args.model is None:
        args.model = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    if args.backend is None:
        args.backend = "vllm"
    if args.bank is None and args.mode == "tir_rag":
        args.bank = "results/tir/exemplar_bank.jsonl"
    if args.out is None:
        args.out = f"results/tir/{args.mode}"


def write_smoke_sentinel() -> None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "nogit"
    Path("logs").mkdir(exist_ok=True)
    sentinel = Path(f"logs/tir_smoke_passed_{commit}.txt")
    sentinel.write_text(f"smoke passed at {time.time()}\n")
    print(f"[smoke] wrote sentinel: {sentinel}", flush=True)


def main() -> int:
    args = parse_args()
    apply_smoke_defaults(args)
    apply_production_defaults(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] mode={args.mode}  smoke={args.smoke}  n={args.n}  "
          f"model={args.model}  backend={args.backend}  out={out_dir}", flush=True)

    # ---- Load eval rows ----
    rows: list[dict] = []
    with open(args.eval_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.n is not None and args.n > 0:
        rows = rows[: args.n]
    print(f"[data] loaded {len(rows)} eval rows from {args.eval_jsonl}", flush=True)

    # ---- Backend ----
    if args.backend == "hf":
        generate_fn, tokenizer = make_hf_backend(args.model, device=args.device)
    elif args.backend == "vllm":
        generate_fn, tokenizer = make_vllm_backend(args.model)
    else:
        raise ValueError(args.backend)

    # In smoke mode, force the first generation to emit a tool call so the
    # sandbox path is exercised on every problem regardless of model size.
    # See docstring on with_smoke_tool_canary for the rationale.
    if args.smoke and args.mode in ("tir", "tir_rag"):
        generate_fn = with_smoke_tool_canary(generate_fn)

    # ---- Sandbox ----
    sandbox = PythonSandbox() if args.mode != "cot" else None

    # ---- Retriever ----
    retriever = None
    if args.mode == "tir_rag":
        bank = load_bank(args.bank)
        retriever = build_retriever(args.retrieval_policy, bank)
        print(f"[retrieval] policy={args.retrieval_policy}  bank_size={len(bank)}", flush=True)

    # ---- Loop ----
    out_summary = {
        "mode": args.mode,
        "smoke": args.smoke,
        "model": args.model,
        "backend": args.backend,
        "n_problems": len(rows),
        "retrieval_policy": args.retrieval_policy if args.mode == "tir_rag" else None,
        "k": args.k if args.mode == "tir_rag" else None,
        "use_judge": args.use_judge,
        "n_correct": 0,
        "n_scored": 0,
        "method_counts": {},
        "n_saturated": 0,
        "n_tool_calls_total": 0,
        "elapsed_s": 0.0,
    }

    t_start = time.time()
    for i, row in enumerate(rows):
        try:
            result = eval_one(
                row=row,
                mode=args.mode,
                tokenizer=tokenizer,
                generate_fn=generate_fn,
                sandbox=sandbox,
                retriever=retriever,
                k=args.k,
                max_new_tokens_total=args.max_new_tokens,
                max_tool_calls=args.max_tool_calls,
                use_judge=args.use_judge,
            )
        except Exception as e:
            if args.smoke:
                # Smoke must surface failures loudly so login-node testing
                # catches them.
                raise
            print(f"[error] {row['id']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            continue

        out_path = out_dir / f"{row['id']}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

        out_summary["n_scored"] += 1
        if result["grade"]["correct"]:
            out_summary["n_correct"] += 1
        out_summary["method_counts"][result["grade"]["method"]] = (
            out_summary["method_counts"].get(result["grade"]["method"], 0) + 1
        )
        if result["saturated"]:
            out_summary["n_saturated"] += 1
        out_summary["n_tool_calls_total"] += result["n_tool_calls"]

        print(
            f"[{i+1}/{len(rows)}] id={row['id']}  "
            f"correct={result['grade']['correct']}  "
            f"method={result['grade']['method']}  "
            f"tool_calls={result['n_tool_calls']}  "
            f"saturated={result['saturated']}  "
            f"elapsed={result['elapsed_s']:.1f}s",
            flush=True,
        )

    out_summary["elapsed_s"] = round(time.time() - t_start, 2)
    out_summary["accuracy"] = (
        out_summary["n_correct"] / out_summary["n_scored"]
        if out_summary["n_scored"] else 0.0
    )
    (out_dir / "summary.json").write_text(json.dumps(out_summary, ensure_ascii=False, indent=2))
    print(f"[summary] {json.dumps(out_summary, ensure_ascii=False)}", flush=True)

    if args.smoke:
        write_smoke_sentinel()

    return 0


if __name__ == "__main__":
    sys.exit(main())
