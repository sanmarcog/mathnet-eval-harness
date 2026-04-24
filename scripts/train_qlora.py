"""Fine-tune Qwen-2.5-1.5B-Instruct on MathNet with QLoRA.

Thin CLI wrapper around `mathnet_eval.training.train_qlora`. All the logic
lives there; this script just parses args and dispatches.

Usage (inside GPU-allocated Slurm job; see slurm/train_qlora.sbatch):
    export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
    export PYTHONPATH=src
    python scripts/train_qlora.py \
        --out-dir /gscratch/scrubbed/sanmarco/adapters/qwen-mathnet-run1
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from mathnet_eval.training import TrainConfig, train_qlora


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    # Surface the commonly-tuned fields as CLI flags. Everything else uses
    # the TrainConfig default (standard-recipe-first).
    p.add_argument("--base-model")
    p.add_argument("--train-jsonl")
    p.add_argument("--eval-jsonl")
    p.add_argument("--out-dir")

    p.add_argument("--mid-eval-n", type=int)
    p.add_argument("--mid-eval-log-path")
    p.add_argument(
        "--mid-eval-fractions", type=float, nargs="+",
        help="Mid-training eval triggers at these fractions of total steps (plus every epoch end). Default: 0.25 0.5 0.75",
    )

    p.add_argument("--lora-r", type=int)
    p.add_argument("--lora-alpha", type=int)
    p.add_argument("--lora-dropout", type=float)

    p.add_argument("--num-train-epochs", type=int)
    p.add_argument("--per-device-train-batch-size", type=int)
    p.add_argument("--gradient-accumulation-steps", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--max-seq-length", type=int)
    p.add_argument("--seed", type=int)

    # Run-2 additions.
    p.add_argument("--completion-only-loss", action="store_true", default=None,
                   help="Wrap in DataCollatorForCompletionOnlyLM so loss masks system+user tokens.")
    p.add_argument("--response-template",
                   help="Exact token prefix that marks the assistant turn start. Default matches Qwen2.5 / Qwen3.")
    p.add_argument("--enable-thinking", action="store_true", default=None,
                   help="Qwen3 thinking mode in chat template. Default: OFF (matches direct-response training data).")

    args = p.parse_args()

    # Merge CLI overrides onto defaults.
    cfg_kwargs = {f.name: getattr(args, f.name.replace("-", "_"), None) for f in fields(TrainConfig)}
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if v is not None}
    return TrainConfig(**cfg_kwargs)


def main() -> int:
    cfg = parse_args()
    train_qlora(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
