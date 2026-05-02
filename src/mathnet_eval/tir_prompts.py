"""Prompts for the three Week-2 eval conditions: CoT, TIR, TIR+RAG.

Kept separate from the inference layer so they can be diffed cleanly. The
TIR system prompt teaches a single tool-call convention that we parse on
the host side: model emits a ```python``` code block, we execute, return
the result wrapped in a ```output``` block, model continues until \\boxed.

Convention chosen to match Alibaba's Qwen2.5-Math TIR convention as
documented in 2409.12122 §4.2 (model emits Python code blocks, executor
returns stdout in ``output`` block). This is the format the published
+11.1pp OlympiadBench result was measured under, so transfer is
well-defined.
"""

from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# Canonical Qwen2.5-Math-Instruct CoT system prompt, taken verbatim from the
# Qwen2.5-Math tech report (2409.12122 §4) and the model's own example_demo
# in the HF model card. Earlier versions of this file used a wordier
# "expert mathematician..." prompt that triggered greedy-decoding repetition
# loops on ~39% of MathNet rollouts (4.5% accuracy on the first 220
# problems vs Alibaba's published 38.1% on OlympiadBench). The model is
# format-sensitive: deviating from the trained system prompt breaks output
# coherence, not capability. Pre-reg deviation note in tir_rag_plan.md
# documents the swap.
COT_SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Canonical Qwen2.5-Math-Instruct TIR system prompt, taken from the
# Qwen2.5-Math tech report (2409.12122 §4.2). Same rationale as COT_SYSTEM:
# this model was trained on Alibaba's exact wording, deviating from it
# breaks output format compliance.
TIR_SYSTEM = (
    "Please integrate natural language reasoning with programs to solve "
    "the problem above, and put your final answer within \\boxed{}."
)

TIR_RAG_SYSTEM = (
    TIR_SYSTEM
    + "\n\nHere are similar worked examples to refer to. Each example "
    "shows a problem, a working Python solution, the executed output, and "
    "the final answer. Use these for inspiration; the new problem may "
    "require a different approach."
)


# ---------------------------------------------------------------------------
# User-message formatters
# ---------------------------------------------------------------------------

def format_cot_user(problem: str) -> str:
    return problem.strip()


def format_tir_user(problem: str) -> str:
    return problem.strip()


def format_tir_rag_user(
    problem: str,
    exemplars: Iterable[dict],
    exemplar_type: str = "tir",
) -> str:
    """Prepend retrieved exemplars to the problem.

    `exemplar_type` selects the formatting:
      - "tir": exemplar must carry `problem, code, output, final_answer`
        (the TIR exemplar-bank schema). Renders a worked Python solution.
      - "cot": exemplar must carry `problem, reasoning, final_answer`.
        Renders a step-by-step natural-language solution. Used for the
        18-cell retrieval ablation's CoT control arm.

    The generation-time system prompt is `TIR_RAG_SYSTEM` either way (the
    model is still told it may use Python); only the *retrieved content*
    differs across the two arms.
    """
    parts: list[str] = []
    for i, ex in enumerate(exemplars, start=1):
        parts.append(f"### Example {i}\nProblem: {ex['problem'].strip()}")
        if exemplar_type == "tir":
            parts.append("```python\n" + ex["code"].strip() + "\n```")
            parts.append("```output\n" + ex["output"].strip() + "\n```")
        elif exemplar_type == "cot":
            parts.append(ex["reasoning"].strip())
        else:
            raise ValueError(f"unknown exemplar_type: {exemplar_type}")
        parts.append(f"Final answer: \\boxed{{{ex['final_answer'].strip()}}}")
    parts.append("### Now solve this problem")
    parts.append(problem.strip())
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Stop conditions for generation
# ---------------------------------------------------------------------------

# When we see one of these in the model's emitted text, generation should
# pause for either tool execution or final-answer extraction.
TOOL_CALL_OPEN = "```python"
TOOL_CALL_CLOSE = "```"
OUTPUT_OPEN = "```output"
BOXED_PATTERN = r"\\boxed\{"
