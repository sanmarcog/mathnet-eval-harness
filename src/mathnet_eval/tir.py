"""Tool-Integrated-Reasoning runner for the Week-2 eval.

Three components:

1. ``PythonSandbox`` — subprocess-isolated Python executor with timeout
   and stdout truncation. Not smolagents (despite what an earlier draft of
   the pre-reg said): smolagents' ``LocalPythonExecutor`` is in-process,
   not subprocess-isolated, and we want true isolation. The pre-reg's
   "subprocess-isolated, not E2B" line is the load-bearing one; the
   smolagents reference is the inconsistency. Sandbox is the simplest
   thing that works: ``subprocess.run([python, "-c", code], timeout=10)``.

2. Tool-call parsing: a single regex that extracts ```python blocks out of
   model output. Convention chosen to match Alibaba's published TIR format
   (2409.12122 §4.2) — code in ```python blocks, executor returns stdout
   in ```output blocks.

3. ``TIRRunner`` — orchestrates up to N tool calls per problem, given a
   ``generate_fn`` callback that abstracts the model. Keeping the model
   behind a callback lets the same loop power the smoke run (HF
   transformers, CPU, tiny model) and the production run (vLLM, GPU,
   Qwen2.5-Math-1.5B-Instruct).
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool

    @property
    def output_for_model(self) -> str:
        """The string we feed back to the model in the ```output``` block."""
        if self.timed_out:
            return f"[error] Python execution timed out."
        if self.returncode != 0:
            err = self.stderr.strip().splitlines()
            tail = "\n".join(err[-3:]) if err else "(no stderr)"
            return f"[error] Python exited with code {self.returncode}:\n{tail}"
        return self.stdout


class PythonSandbox:
    """Run Python code in a fresh subprocess with timeout + output cap.

    `python_bin`: which interpreter to use. Default is the same one running
    this code; cluster jobs should pass the conda-env python explicitly.
    """

    def __init__(
        self,
        python_bin: str | None = None,
        per_call_timeout_s: float = 60.0,
        max_output_chars: int = 2000,
    ) -> None:
        # Note on the timeout: the pre-reg originally specified 10s. That
        # is too tight on shared filesystems — on Hyak's NFS-mounted conda
        # env, a cold-start `import sympy` alone takes ~10s, so EVERY
        # subprocess call timed out before reaching its print(). Bumped to
        # 60s as a defensible cold-start-safe ceiling. For genuinely
        # runaway code, 60s is still bounded; for typical TIR snippets
        # that just call into already-loaded sympy/numpy in the
        # subprocess-startup path, the wall-clock cost is dominated by
        # python interpreter startup + sympy import.
        self.python_bin = python_bin or sys.executable
        self.per_call_timeout_s = per_call_timeout_s
        self.max_output_chars = max_output_chars

    def run(self, code: str) -> SandboxResult:
        try:
            proc = subprocess.run(
                [self.python_bin, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.per_call_timeout_s,
                # No env/cwd inherited beyond what subprocess defaults give us.
                # Network and filesystem access are *not* sandboxed at the OS
                # level — we rely on the model not emitting destructive code,
                # plus the timeout. For a stronger guarantee swap to E2B.
            )
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                stdout=(e.stdout or "").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                stderr=(e.stderr or "").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
                returncode=-1,
                timed_out=True,
            )

        stdout = proc.stdout[: self.max_output_chars]
        if len(proc.stdout) > self.max_output_chars:
            stdout += f"\n[... output truncated at {self.max_output_chars} chars]"
        stderr = proc.stderr[: self.max_output_chars]
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode,
            timed_out=False,
        )


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------

# Match ```python ... ``` non-greedy. Intentionally not anchored to start of
# line so models that emit indented blocks still parse.
_PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_last_python_block(text: str) -> str | None:
    """Return the body of the last ```python block in `text`, or None.

    "Last" because the model may emit retrieved exemplar blocks first
    (in TIR+RAG mode); we only want to execute its own newest block.
    """
    matches = _PYTHON_BLOCK_RE.findall(text)
    return matches[-1].strip() if matches else None


def has_boxed_answer(text: str) -> bool:
    return bool(_BOXED_RE.search(text))


def extract_boxed_answer(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


# ---------------------------------------------------------------------------
# TIR loop
# ---------------------------------------------------------------------------

GenerateFn = Callable[[str, int, list[str]], str]
"""Callback signature: (prompt_so_far, max_new_tokens, stop_strings) -> generated_text.

The callback is responsible for tokenization + sampling + decoding; this
keeps the loop ignorant of HF vs vLLM vs anything else."""


@dataclass
class TIRStep:
    kind: str        # "generate" | "tool" | "final"
    text: str        # generated text for "generate", code for "tool", output for tool
    tool_result: SandboxResult | None = None


@dataclass
class TIRTrace:
    full_text: str = ""           # everything that goes into the assistant turn
    steps: list[TIRStep] = field(default_factory=list)
    n_tool_calls: int = 0
    finished: bool = False        # True if the model emitted a \boxed answer
    saturated: bool = False       # True if we hit max_new_tokens or max_calls without \boxed
    final_answer: str | None = None


class TIRRunner:
    """Drive a model through up to ``max_tool_calls`` Python rounds.

    Loop:
      1. generate next chunk
      2. if assistant has emitted a complete ```python block AND no \\boxed:
            extract code, run sandbox, append ```output``` block, go to 1
      3. else: break and extract \\boxed answer

    The loop is bounded by both ``max_tool_calls`` and the cumulative
    generation budget (``max_new_tokens_total``) — whichever is hit first.
    """

    def __init__(
        self,
        sandbox: PythonSandbox,
        generate_fn: GenerateFn,
        max_tool_calls: int = 4,
        max_new_tokens_per_step: int = 1024,
        max_new_tokens_total: int = 4096,
    ) -> None:
        self.sandbox = sandbox
        self.generate_fn = generate_fn
        self.max_tool_calls = max_tool_calls
        self.max_new_tokens_per_step = max_new_tokens_per_step
        self.max_new_tokens_total = max_new_tokens_total

    def run(self, prompt_prefix: str) -> TIRTrace:
        """Run the TIR loop. ``prompt_prefix`` is everything up through the
        opening of the assistant turn (i.e. immediately after the
        chat-template's assistant-start token). The runner appends generated
        text to it across rounds."""
        trace = TIRTrace()
        budget_left = self.max_new_tokens_total

        # Stop strings: when the model writes ``` after a python block, that
        # closes its tool call and we want to immediately execute. The
        # ``` alone is ambiguous (could be opening or closing), so we also
        # detect by looking at the parsed text after each generation.
        stop_strings = ["```\n"]

        for _ in range(self.max_tool_calls + 1):
            new_text = self.generate_fn(
                prompt_prefix + trace.full_text,
                min(self.max_new_tokens_per_step, budget_left),
                stop_strings,
            )
            trace.full_text += new_text
            trace.steps.append(TIRStep("generate", new_text))
            budget_left -= max(1, len(new_text) // 4)  # rough char->token

            # If we've already produced a final answer, stop.
            if has_boxed_answer(trace.full_text):
                trace.finished = True
                trace.final_answer = extract_boxed_answer(trace.full_text)
                break

            # Otherwise, look for an unexecuted python block at the end.
            code = extract_last_python_block(trace.full_text)
            if code is None or trace.n_tool_calls >= self.max_tool_calls:
                # No tool call to execute, OR exhausted our call budget.
                # If the generation simply ran out of tokens, mark saturated.
                if budget_left <= 0:
                    trace.saturated = True
                break

            # Has the last block already been "answered" by an output block
            # following it? (i.e. did the model emit ```python ... ``` and
            # then ```output ... ```?) — if so don't re-execute.
            tail = trace.full_text.rsplit("```python", 1)[-1]
            if "```output" in tail:
                # Already executed in a prior step somehow; bail rather than
                # loop. Should not normally happen.
                break

            result = self.sandbox.run(code)
            trace.n_tool_calls += 1
            output_block = f"\n```output\n{result.output_for_model}\n```\n"
            trace.full_text += output_block
            trace.steps.append(TIRStep("tool", code, tool_result=result))

            if budget_left <= 0:
                trace.saturated = True
                break

        if not trace.finished:
            trace.final_answer = extract_boxed_answer(trace.full_text)
            if trace.final_answer is None:
                trace.saturated = True

        return trace
