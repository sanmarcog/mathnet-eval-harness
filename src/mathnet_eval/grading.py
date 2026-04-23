"""Answer extraction + hybrid grader for MathNet responses.

The pipeline is intentionally layered so we can see which method caught
each correct answer — useful when we later decide whether to invest in
better normalization or LLM-as-judge coverage.

1. `extract_answer(text)`  -> pull the claimed answer out of the response
2. `normalize(s)`          -> strip LaTeX delimiters, whitespace, etc.
3. `symbolic_equal(a, b)`  -> parse with sympy, check mathematical equality
4. `judge_equal(pred, gold, problem)` -> ask Claude if two answers agree

`grade(response_text, gold, problem)` runs the layers in order and stops
at the first hit, returning `{correct, method, predicted}` so analysis
code can see *how* each problem was scored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---- Extraction -------------------------------------------------------------

_FINAL_ANSWER_PATTERNS = [
    # Explicit "Final answer:" (our system prompt asks for this).
    re.compile(r"[Ff]inal\s*[Aa]nswer\s*[:=]\s*(.+?)\s*$", re.MULTILINE),
    # \boxed{...} — standard olympiad convention.
    re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"),
    # "The answer is ..." variants.
    re.compile(r"[Tt]he\s+(?:final\s+)?answer\s+is\s+(.+?)\s*[.$]", re.MULTILINE),
]


def extract_answer(response_text: str) -> str | None:
    """Return the claimed answer string, or None if no marker found.

    Tries several conventions in order: `Final answer: X`, `\\boxed{X}`,
    `The answer is X`. Prefers the *last* match in the text, since reasoning
    steps sometimes contain intermediate "answers" that shouldn't count.
    """
    best: str | None = None
    best_pos = -1
    for pat in _FINAL_ANSWER_PATTERNS:
        for m in pat.finditer(response_text):
            if m.start() > best_pos:
                best_pos = m.start()
                best = m.group(1)
    return best.strip() if best else None


# ---- Normalization ----------------------------------------------------------

_LATEX_DELIM = re.compile(r"\$+|\\\[|\\\]|\\\(|\\\)")
_LATEX_LRIGHT = re.compile(r"\\left|\\right")
_LATEX_SPACING = re.compile(r"\\[,;!>:]|~|\\quad|\\qquad")
_WHITESPACE = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Strip LaTeX delimiters/spacing, collapse whitespace, lowercase-agnostic
    trimming. Not a substitute for symbolic equality — cheap preprocessing
    to make direct string compare slightly more forgiving."""
    s = _LATEX_DELIM.sub("", s)
    s = _LATEX_LRIGHT.sub("", s)
    s = _LATEX_SPACING.sub(" ", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = _WHITESPACE.sub(" ", s).strip()
    return s


# ---- Symbolic equality ------------------------------------------------------

_LATEX_FRAC = re.compile(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_LATEX_SQRT = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_LATEX_POW_BRACE = re.compile(r"\^\s*\{([^{}]+)\}")


def _latex_to_sympy_str(s: str) -> str:
    """Best-effort LaTeX -> Python/sympy-parseable string. Only handles the
    patterns we actually see in MathNet `final_answer` values: \\frac, \\sqrt,
    ^{...}, \\cdot / \\times, \\pi. Anything else passes through unchanged
    and sympify will just reject it."""
    s = normalize(s)
    s = _LATEX_FRAC.sub(r"(\1)/(\2)", s)
    s = _LATEX_SQRT.sub(r"sqrt(\1)", s)
    s = _LATEX_POW_BRACE.sub(r"**(\1)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\pi", "pi")
    return s


def _to_sympy(s: str):
    """Try to parse `s` as a sympy expression. Returns the expression or None."""
    try:
        from sympy import sympify  # lazy; heavy import
    except Exception:
        return None

    candidates = [s, _latex_to_sympy_str(s)]
    try:
        from sympy.parsing.latex import parse_latex
        # parse_latex needs antlr4; if missing this call raises at import-site.
        for form in candidates:
            try:
                return parse_latex(form)
            except Exception:
                continue
    except Exception:
        pass

    for form in candidates:
        try:
            return sympify(form)
        except Exception:
            continue
    return None


def symbolic_equal(a: str, b: str) -> bool:
    """Return True iff `a` and `b` parse to mathematically equal sympy
    expressions. False on any parse failure or inequality."""
    ea, eb = _to_sympy(a), _to_sympy(b)
    if ea is None or eb is None:
        return False
    try:
        from sympy import simplify
        return bool(simplify(ea - eb) == 0)
    except Exception:
        return False


# ---- LLM-as-judge -----------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a strict mathematical equivalence judge. Given two candidate "
    "answers to an olympiad problem, decide whether they represent the "
    "same mathematical object (same numeric value, same set of solutions, "
    "same expression up to trivial rearrangement). Output exactly one token: "
    "YES or NO."
)


def _judge_prompt(predicted: str, gold: str, problem: str | None) -> str:
    parts = [
        "# Problem",
        problem or "(problem omitted — judge based on answers alone)",
        "",
        "# Gold answer",
        gold,
        "",
        "# Candidate answer",
        predicted,
        "",
        "Do these represent the same mathematical answer? Reply YES or NO.",
    ]
    return "\n".join(parts)


def judge_equal(
    predicted: str,
    gold: str,
    problem: str | None = None,
    *,
    judge_model: str = "sonnet-4-6",
) -> bool:
    """Ask Claude whether `predicted` and `gold` agree. Lazy import keeps
    the grading module usable in environments where the inference harness
    / API keys are absent (e.g. unit tests of `normalize`)."""
    from .inference import generate

    resp = generate(
        _judge_prompt(predicted, gold, problem),
        model=judge_model,
        max_tokens=4,
        temperature=0.0,
        system=_JUDGE_SYSTEM,
    )
    return resp.text.strip().upper().startswith("YES")


# ---- Public API -------------------------------------------------------------

@dataclass
class Grade:
    correct: bool
    method: str     # "exact" | "normalized" | "symbolic" | "judge" | "miss"
    predicted: str | None


def grade(
    response_text: str,
    gold_answer: str,
    problem: str | None = None,
    *,
    use_judge: bool = False,
    judge_model: str = "sonnet-4-6",
) -> Grade:
    """Score a response against the gold answer with a layered pipeline.

    Returns which layer (if any) caught it, so analysis can see how much
    of correctness relies on the LLM-judge vs. cheap string / symbolic checks.
    """
    pred = extract_answer(response_text)
    if pred is None:
        return Grade(correct=False, method="miss", predicted=None)

    # Layer 1: exact match
    if pred.strip() == gold_answer.strip():
        return Grade(True, "exact", pred)

    # Layer 2: cheap normalization
    if normalize(pred) == normalize(gold_answer):
        return Grade(True, "normalized", pred)

    # Layer 3: symbolic equality via sympy
    if symbolic_equal(pred, gold_answer):
        return Grade(True, "symbolic", pred)

    # Layer 4 (opt-in): LLM judge
    if use_judge:
        if judge_equal(pred, gold_answer, problem, judge_model=judge_model):
            return Grade(True, "judge", pred)

    return Grade(False, "miss", pred)
