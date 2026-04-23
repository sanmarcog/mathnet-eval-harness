"""QLoRA fine-tuning for Qwen-2.5-1.5B-Instruct on MathNet."""

from __future__ import annotations


def train_qlora(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    train_dataset=None,
    output_dir: str = "./adapters/qwen-mathnet",
    **kwargs,
):
    """QLoRA training loop. TODO: wire up PEFT + TRL SFTTrainer + 4-bit NF4 base."""
    raise NotImplementedError
