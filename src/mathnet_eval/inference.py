"""Unified inference client across frontier APIs and local HuggingFace models.

Frontier models share a common `generate(prompt, model) -> response` surface so the
eval harness can swap them freely. Each backend handles its own auth (via env vars
loaded from .env) and rate limiting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Response:
    """A single model response — structured so graders and analysis code don't have to
    care which backend produced it."""
    model: str
    prompt: str
    text: str
    raw: dict  # full provider payload, for debugging / re-grading
    usage: dict  # tokens in/out, cost if known
    latency_s: float


def generate(prompt: str, model: str, **kwargs) -> Response:
    """Dispatch to the right backend. TODO: implement claude/openai/gemini/local."""
    raise NotImplementedError
