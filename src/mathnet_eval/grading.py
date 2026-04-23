"""Answer extraction and correctness grading for MathNet responses."""

from __future__ import annotations


def extract_answer(response_text: str) -> str | None:
    """Pull the final answer out of a model response. TODO: likely \\boxed{...} or similar."""
    raise NotImplementedError


def is_correct(predicted: str | None, gold: str) -> bool:
    """Compare predicted vs gold answer. TODO: numeric/expression-aware comparison."""
    raise NotImplementedError
