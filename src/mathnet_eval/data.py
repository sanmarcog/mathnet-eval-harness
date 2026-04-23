"""MathNet loading, stratified splits, and prompt formatting."""

from __future__ import annotations


def load_mathnet():
    """Load the MathNet dataset from HuggingFace. TODO: fill in dataset id + config."""
    raise NotImplementedError


def stratified_split(dataset, eval_size: int = 500, train_size: int = 10_000, seed: int = 0):
    """Build a stratified eval/train split from MathNet. TODO: decide strata (subject? difficulty? language?)."""
    raise NotImplementedError


def format_prompt(problem: dict) -> str:
    """Format a MathNet record into a chat prompt. TODO: define after inspecting schema."""
    raise NotImplementedError
