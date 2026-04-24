"""Merge a PEFT/LoRA adapter into its base model weights and save a full
standalone checkpoint. This is the standard post-training step so the
fine-tuned model can be served by vLLM (which, in our current
`eval_qwen_hf.py` wiring, does not load adapters).

Runs entirely on CPU/bf16 -- we deliberately do NOT load the base in 4-bit
here, because `merge_and_unload()` would have to dequantize anyway and the
merged weights must be written in a real dtype.

    python scripts/merge_adapter.py \\
        --base-model Qwen/Qwen3-1.7B \\
        --adapter-dir /gscratch/scrubbed/sanmarco/adapters/qwen3-mathnet-run2 \\
        --out-dir    /gscratch/scrubbed/sanmarco/adapters/qwen3-mathnet-run2-merged
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    if not args.adapter_dir.exists():
        print(f"ERROR: adapter dir not found: {args.adapter_dir}", file=sys.stderr)
        return 1

    dtype = getattr(torch, args.dtype)
    print(f">>> loading base: {args.base_model}  dtype={args.dtype}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    print(f">>> applying adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))

    print(">>> merging LoRA weights into base")
    merged = model.merge_and_unload()

    print(f">>> saving merged model: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out_dir), safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.save_pretrained(str(args.out_dir))

    print(">>> done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
