"""Dr. GRPO (bias-corrected GRPO) training on Qwen3-1.7B for MathNet.

Direct test of the Run-4 hypothesis: the SFT failure mode was
length-amplification (training on long traces taught the model to
think *longer*, not better). Dr. GRPO (Liu et al., 2503.20783) is
the bias-corrected variant of GRPO that addresses exactly that
length-amplification bias.

Design:
- Base: Qwen/Qwen3-1.7B (same as Run 4 — preserves paired comparison
  vs. existing n=500 eval).
- LoRA wrapper (r=64, alpha=128) for memory at 48GB. Full-parameter
  GRPO would also fit but LoRA is more conservative on optimizer state.
- Rollout group size 4 (cut from default 8 for 48GB feasibility).
- vLLM colocate generation (rollouts on same GPU as training).
- Reward: 1.0 if extract_answer(completion) matches gold, else 0.0.
  Cheap-grader only — no LLM judge during training (judge is for
  post-training eval).
- Loss: Dr. GRPO bias-corrected loss via TRL's loss_type="dr_grpo"
  (TRL >=0.16). Falls back with explicit error if not available.

Usage (calibration, ~5 steps):
    python scripts/train_dr_grpo.py --max-steps 5 --out-dir /tmp/calib

Usage (full):
    python scripts/train_dr_grpo.py \
        --max-steps 200 \
        --out-dir $ADAPTERS_ROOT/qwen3-mathnet-drgrpo
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_FILE = REPO / "data" / "splits" / "train_english.jsonl"


# Same answer extractor used in the grader. Kept inline rather than
# importing from mathnet_eval.grading to avoid pulling sympy / heavy
# dependencies into the training process.
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_PATTERNS = [
    re.compile(r"final answer (?:is\s*)?:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"answer (?:is\s*)?:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"=\s*([^=\n]+?)\s*$"),
]


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip().rstrip(".")
    return None


def normalize(s: str) -> str:
    s = s.strip().rstrip(".").rstrip()
    s = s.replace(" ", "").replace(",", "")
    return s.lower()


def reward_fn(completions, **kwargs):
    """Reward = 1.0 if extracted answer matches gold (normalized), else 0.0.

    Cheap-grader-only — no LLM judge during training. The post-training
    n=500 eval still uses the full 4-layer grader for the paired
    comparison.

    The trainer passes per-example fields via kwargs as parallel lists.
    Gold is the `gold` column we attach when building the dataset.
    """
    golds = kwargs.get("gold", [None] * len(completions))
    rewards = []
    for completion, gold in zip(completions, golds):
        # GRPOTrainer passes completions as a list of message dicts when
        # using chat templates; flatten to text for our purposes.
        if isinstance(completion, list):
            text = "".join(m.get("content", "") for m in completion)
        else:
            text = str(completion)
        pred = extract_answer(text)
        if pred is None or gold is None:
            rewards.append(0.0)
            continue
        rewards.append(1.0 if normalize(pred) == normalize(str(gold)) else 0.0)
    return rewards


def load_prompts(jsonl_path: Path, max_rows: int | None) -> Dataset:
    rows = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            d = json.loads(line)
            problem = d.get("problem") or d.get("question") or d.get("prompt")
            gold = d.get("final_answer") or d.get("answer") or d.get("gold")
            if not problem or gold is None:
                continue
            rows.append({"prompt": problem, "gold": str(gold)})
    print(f"loaded {len(rows)} prompt/gold pairs from {jsonl_path}")
    return Dataset.from_list(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--train-jsonl", default=DEFAULT_TRAIN_FILE, type=Path)
    p.add_argument("--max-rows", type=int, default=None,
                   help="cap training set size (calibration)")
    p.add_argument("--max-steps", type=int, default=200)

    # GRPO-specific knobs
    p.add_argument("--num-generations", type=int, default=4,
                   help="rollout group size (default 4 for 48GB; vanilla 8)")
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-completion-length", type=int, default=3000,
                   help="rollout cap during training (smaller than 16K eval cap to fit memory)")
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.04,
                   help="KL coefficient against reference model")

    # LoRA
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--no-lora", action="store_true",
                   help="full-parameter GRPO instead of LoRA")

    # Eval / save / log
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # TRL import is deferred so the --help path doesn't pull in heavy deps.
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise SystemExit(
            "TRL not available. Install with: pip install 'trl>=0.16'\n"
            f"original error: {e}"
        )

    # Verify the installed TRL supports Dr. GRPO loss type.
    cfg_signature = GRPOConfig.__init__.__doc__ or ""
    has_loss_type = "loss_type" in cfg_signature or hasattr(GRPOConfig, "loss_type")
    if not has_loss_type:
        # Try instantiating with loss_type to be sure.
        try:
            _ = GRPOConfig(output_dir="/tmp/_check", loss_type="dr_grpo")
        except TypeError:
            raise SystemExit(
                "Installed TRL does not support GRPOConfig(loss_type=...).\n"
                "Upgrade: pip install -U 'trl>=0.16' (Dr. GRPO support)\n"
                "or apply a manual Dr. GRPO loss patch to GRPOTrainer."
            )

    train_ds = load_prompts(args.train_jsonl, args.max_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = None if args.no_lora else LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_cfg = GRPOConfig(
        output_dir=str(args.out_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        loss_type="dr_grpo",          # bias-corrected variant
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        report_to="none",
        use_vllm=True,                 # colocate vLLM for fast rollouts
        vllm_mode="colocate",
        remove_unused_columns=False,    # keep `gold` column for reward fn
    )

    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=train_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print(f"\n=== Dr. GRPO training: {args.base_model} → {args.out_dir} ===")
    print(f"  steps={args.max_steps}  group_size={args.num_generations}  "
          f"LR={args.learning_rate}  beta={args.beta}")
    print(f"  max_completion_length={args.max_completion_length}  "
          f"LoRA={'off' if args.no_lora else f'r={args.lora_r}'}")

    trainer.train()
    trainer.save_model(str(args.out_dir))
    print(f"\nadapter saved to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
