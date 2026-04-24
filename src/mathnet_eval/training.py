"""QLoRA fine-tuning for Qwen-2.5-1.5B-Instruct on MathNet.

Standard HF SFT recipe — 4-bit NF4 base + LoRA on all attention + FFN
projections, cosine LR schedule, TRL's SFTTrainer. Not the day to invent
a training technique; if this doesn't hit the target, THEN we tune.

Mid-training eval runs the (in-memory, actively training) model against a
50-problem held-out subset every N steps and logs accuracy. The mid-run
eval skips the LLM judge — exact + symbolic layers only, which is enough
for a trend signal and avoids Sonnet spend during training.

Entry points:
- `train_qlora(...)` — main training function
- The CLI wrapper lives at `scripts/train_qlora.py`
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the following olympiad problem. "
    "Show your work briefly, then put the final answer on its own line in the form: "
    "Final answer: <answer>."
)


@dataclass
class TrainConfig:
    """Standard HF SFT recipe. Defaults follow the vanilla example for
    Qwen-2.5-1.5B + QLoRA; tune only if the first run plateaus."""

    base_model: str = BASE_MODEL
    train_jsonl: str = "data/splits/train.jsonl"
    eval_jsonl: str = "data/splits/eval.jsonl"
    out_dir: str = "./adapters/qwen-mathnet"

    # Mid-training eval fires at each of (fraction × total_steps) plus every
    # epoch end plus training end. Quarter-fractions give us 4 data points
    # across any training run for drawing a trend; epoch-end triggers let
    # us compare "end of epoch 1" vs "end of epoch 2" for the
    # overfitting-watch pattern (is eval acc still climbing in epoch 2, or
    # has it plateaued/dropped while train loss continues to fall?).
    mid_eval_n: int = 50
    mid_eval_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    mid_eval_log_path: str = "./adapters/qwen-mathnet/mid_eval.jsonl"
    mid_eval_max_new_tokens: int = 1024

    # LoRA (vanilla).
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Optim (vanilla).
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    # Sequence / logging.
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 200

    # Reproducibility.
    seed: int = 0

    # Run-2 additions.
    # When True, wraps the dataset in DataCollatorForCompletionOnlyLM so the
    # loss is only computed on assistant tokens (not system / user). Default
    # off to preserve Run-1 behavior; Run-2 sbatch flips it via --completion-only.
    completion_only_loss: bool = False
    # Template that marks the start of the assistant turn, used by the
    # completion-only collator to find where labels start being unmasked.
    # Qwen 2.5 / Qwen 3 both use the same prefix.
    response_template: str = "<|im_start|>assistant\n"
    # Optional: disable thinking mode for Qwen3-family models. Ignored by
    # Qwen 2.5 (tokenizer has no thinking mode). For Qwen3, False = direct
    # response (matches our training data which has no <think> traces).
    enable_thinking: bool = False


def _format_messages(row: dict) -> list[dict]:
    """Row -> chat-format messages for SFT. Uses the first non-empty
    solutions_markdown entry, appended with the final_answer so the model
    learns to produce both reasoning and a clearly-marked final answer."""
    problem = row["problem_markdown"].strip()
    solutions = [s for s in (row.get("solutions_markdown") or []) if s and s.strip()]
    if not solutions:
        raise ValueError(f"row {row.get('id')} has no usable solutions_markdown")
    solution = solutions[0].strip()
    final = (row.get("final_answer") or "").strip()

    assistant = solution
    if final and f"Final answer: {final}" not in solution:
        assistant = f"{solution}\n\nFinal answer: {final}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": assistant},
    ]


def _load_and_filter_train(path: str | Path, tokenizer, enable_thinking: bool = False) -> Any:
    """Load train.jsonl, drop rows with no usable solution, format to text
    via tokenizer.apply_chat_template, return a HF Dataset with a `text`
    column (SFTTrainer expects this by default).

    `enable_thinking` is forwarded to `apply_chat_template`; Qwen 3
    respects it, Qwen 2.5's tokenizer ignores the kwarg (silently accepts).
    """
    from datasets import Dataset

    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    kept = []
    dropped = 0
    for r in rows:
        sols = r.get("solutions_markdown") or []
        if not any(s and s.strip() for s in sols):
            dropped += 1
            continue
        try:
            msgs = _format_messages(r)
        except ValueError:
            dropped += 1
            continue
        # Qwen 3 tokenizer supports enable_thinking kwarg; older ones may
        # raise TypeError. Fall back on a template call without the kwarg.
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        kept.append({"text": text, "id": r["id"]})
    print(f"    train data: kept {len(kept)}, dropped {dropped} (no/empty solutions)")
    return Dataset.from_list(kept)


def _sample_mid_eval_subset(eval_jsonl: str | Path, n: int, seed: int) -> list[dict]:
    import random
    rows = [json.loads(l) for l in Path(eval_jsonl).read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def _mid_eval_generate_and_grade(model, tokenizer, subset: list[dict], max_new_tokens: int) -> dict:
    """Run the model on each held-out problem, extract the answer, score
    with `exact` + `symbolic` layers only (no judge during training).
    Returns {n_correct, n_total, accuracy, breakdown}."""
    import torch
    from .grading import extract_answer, normalize_for_exact, symbolic_equal

    model.eval()
    n_correct = 0
    methods = {"exact": 0, "normalized": 0, "symbolic": 0, "miss": 0}

    with torch.no_grad():
        for r in subset:
            problem = r["problem_markdown"].strip()
            gold = (r.get("final_answer") or "").strip()
            if not gold:
                continue
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(text)
            if pred is None:
                methods["miss"] += 1
                continue
            if pred.strip() == gold:
                n_correct += 1; methods["exact"] += 1
            elif normalize_for_exact(pred) == normalize_for_exact(gold):
                n_correct += 1; methods["normalized"] += 1
            elif symbolic_equal(pred, gold):
                n_correct += 1; methods["symbolic"] += 1
            else:
                methods["miss"] += 1

    model.train()
    n = len(subset)
    return {
        "n_total": n,
        "n_correct": n_correct,
        "accuracy": (n_correct / n) if n else 0.0,
        "method_counts": methods,
    }


class MidTrainEvalCallback(TrainerCallback):
    """Transformers callback that runs `_mid_eval_generate_and_grade` at:
      - each `fraction × max_steps` (quarter-fractions by default: 25/50/75%),
      - every epoch boundary,
      - training end.
    Dedupes when triggers coincide (e.g. 50% and end-of-epoch-1 on a 2-epoch
    run). Appends one JSONL line per eval. Judge-free — uses only the cheap
    grader layers (exact + normalized + symbolic).

    Inherits from `TrainerCallback` so every event the Trainer dispatches
    (on_init_end, on_save, on_log, ...) has a no-op default — we only
    override the hooks we care about."""

    def __init__(self, subset, tokenizer, log_path: str, eval_fractions=(0.25, 0.5, 0.75), max_new_tokens: int = 1024):
        self.subset = subset
        self.tokenizer = tokenizer
        self.eval_fractions = tuple(eval_fractions)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self._trigger_steps: set[int] = set()
        self._last_eval_step = -1

    def on_train_begin(self, args, state, control, **kwargs):
        # `state.max_steps` is populated after dataset + training args are
        # resolved, which happens before on_train_begin. Safe to compute here.
        max_steps = getattr(state, "max_steps", 0) or 0
        self._trigger_steps = {int(max_steps * f) for f in self.eval_fractions if 0 < f < 1}
        print(f"  [mid-eval] total training steps = {max_steps}; trigger steps = {sorted(self._trigger_steps)}")

    def _maybe_run(self, state, model, reason: str):
        if state.global_step == self._last_eval_step:
            return
        self._last_eval_step = state.global_step
        t0 = time.perf_counter()
        metrics = _mid_eval_generate_and_grade(model, self.tokenizer, self.subset, self.max_new_tokens)
        elapsed = time.perf_counter() - t0
        # Best-effort grab of current train loss (if the Trainer populated it).
        loss = None
        if getattr(state, "log_history", None):
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    loss = entry["loss"]; break
        entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            "reason": reason,
            "train_loss": loss,
            "elapsed_s": round(elapsed, 1),
            **metrics,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  [mid-eval {reason} step {state.global_step} epoch {state.epoch:.2f}] "
              f"acc={metrics['accuracy']:.1%} ({metrics['n_correct']}/{metrics['n_total']})  "
              f"train_loss={loss}  breakdown={metrics['method_counts']}  {elapsed:.0f}s")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self._trigger_steps:
            self._maybe_run(state, kwargs["model"], reason=f"fraction_step")

    def on_epoch_end(self, args, state, control, **kwargs):
        self._maybe_run(state, kwargs["model"], reason="epoch_end")

    def on_train_end(self, args, state, control, **kwargs):
        self._maybe_run(state, kwargs["model"], reason="train_end")


def train_qlora(cfg: TrainConfig | None = None) -> None:
    """Standard QLoRA SFT pass. Saves adapter to `cfg.out_dir` and a
    mid-training eval log to `cfg.mid_eval_log_path`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    cfg = cfg or TrainConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out_dir) / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(f">>> loading tokenizer + 4-bit base model: {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules), bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f">>> loading + formatting train data: {cfg.train_jsonl}  (enable_thinking={cfg.enable_thinking})")
    train_ds = _load_and_filter_train(cfg.train_jsonl, tokenizer, enable_thinking=cfg.enable_thinking)

    mid_subset = _sample_mid_eval_subset(cfg.eval_jsonl, cfg.mid_eval_n, cfg.seed)
    print(f">>> mid-training eval subset: {len(mid_subset)} problems sampled (seed={cfg.seed}) from {cfg.eval_jsonl}")

    # trl 1.x renamed max_seq_length -> max_length, dropped some kwargs.
    sft_args = SFTConfig(
        output_dir=cfg.out_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=True,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        report_to="none",
        dataset_text_field="text",
        max_length=cfg.max_seq_length,
        seed=cfg.seed,
    )

    mid_cb = MidTrainEvalCallback(
        subset=mid_subset,
        tokenizer=tokenizer,
        log_path=cfg.mid_eval_log_path,
        eval_fractions=cfg.mid_eval_fractions,
        max_new_tokens=cfg.mid_eval_max_new_tokens,
    )

    if cfg.completion_only_loss:
        print(">>> enabling completion-only loss via SFTConfig.completion_only_loss=True (trl 1.x native)")
        sft_args.completion_only_loss = True

    # trl 1.x renamed `tokenizer` -> `processing_class`.
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[mid_cb],
    )

    print(">>> starting training")
    trainer.train()
    print(">>> saving adapter")
    trainer.save_model(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    print(f">>> done: adapter at {cfg.out_dir}; mid-eval log at {cfg.mid_eval_log_path}")
