"""Collect OpenAI safety-filter rejections and their problem text.

Scans the run logs for `invalid_prompt` errors (OpenAI's 400 response when
the reasoning-API safety filter flags a prompt), looks each failing
problem's text up in the eval split, and writes a markdown artifact
suitable for two downstream uses:

    1. Methodology caveats — per-model "filtered / total" counts for the
       Day-2 NOTES denominator accounting.
    2. Blog-post side observation — "what kinds of olympiad math tripped
       OpenAI's safety filter?" (LaTeX patterns? Translated phrasings?
       Specific topics?). Ten minutes of pattern-spotting at the end.

Usage:
    /gscratch/.../qlora/bin/python scripts/collect_openai_flags.py \
        --logs-dir logs \
        --split data/splits/eval.jsonl \
        --out results/full/openai-flagged-problems.md

Safe to re-run; output is overwritten each invocation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


# Match run_eval.py's per-problem error line:
#   [11/500] id=0df4  ERROR: BadRequestError("Error code: 400 - {'error': {'message': ...
#   [135/500] id=03fi  ERROR: ...
_ERR_LINE = re.compile(
    r"\[(?P<idx>\d+)/(?P<total>\d+)\]\s+id=(?P<id>\S+)\s+ERROR:\s+(?P<exc>.+)$"
)
_INVALID_PROMPT = re.compile(r"invalid_prompt", re.IGNORECASE)
# Grab the "message" field out of the nested dict-ish string.
_MSG_FIELD = re.compile(r"'message':\s*'([^']+)'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir", type=Path, default=Path("logs"))
    p.add_argument("--split", type=Path, default=Path("data/splits/eval.jsonl"))
    p.add_argument(
        "--out", type=Path, default=Path("results/full/openai-flagged-problems.md")
    )
    p.add_argument(
        "--openai-only", action="store_true", default=True,
        help="Filter to OpenAI invalid_prompt errors only (default).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    splits = {}
    for line in args.split.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        splits[r["id"]] = r

    # Walk each log file. Filename stem is the model alias.
    flagged: dict[str, list[dict]] = defaultdict(list)
    per_model_total: dict[str, int] = {}

    for log_path in sorted(args.logs_dir.glob("*.log")):
        model = log_path.stem
        lines = log_path.read_text(errors="replace").splitlines()
        # Best-effort "total processed so far" per model -- count how many
        # [X/N] problem lines we see (success or error). The latest index
        # reached gives the denominator for the filter rate.
        last_idx = 0
        total_declared = 0
        for line in lines:
            m = _ERR_LINE.search(line) or re.search(r"\[(\d+)/(\d+)\]", line)
            if m:
                try:
                    last_idx = max(last_idx, int(m.group(1)))
                    total_declared = max(total_declared, int(m.group(2)))
                except (IndexError, ValueError):
                    pass
            m_err = _ERR_LINE.search(line)
            if not m_err:
                continue
            exc_text = m_err.group("exc")
            if args.openai_only and not _INVALID_PROMPT.search(exc_text):
                continue
            msg = _MSG_FIELD.search(exc_text)
            pid = m_err.group("id")
            flagged[model].append(
                {
                    "idx": int(m_err.group("idx")),
                    "total": int(m_err.group("total")),
                    "id": pid,
                    "message": msg.group(1) if msg else exc_text[:200],
                    "problem": splits.get(pid, {}),
                }
            )
        per_model_total[model] = last_idx

    lines: list[str] = []
    lines.append("# OpenAI safety-filter flagged problems\n")
    lines.append(
        "Collected from `logs/*.log` during the Day-2 full-eval run. "
        "These are MathNet problems that OpenAI's reasoning API rejected with "
        "`invalid_prompt` (a 400, not a transient 429/5xx). Our retry logic "
        "correctly does not retry these.\n"
    )
    lines.append(
        "Purpose: (a) denominator accounting for the methodology caveat "
        "(`accuracy = n_correct / n_scored` rather than / N), and (b) "
        "blog-post side observation about what kinds of olympiad math trip "
        "the safety filter.\n"
    )

    lines.append("## Per-model filter rate\n")
    lines.append("| Model | Processed so far | Flagged | Rate |")
    lines.append("|---|---|---|---|")
    for model in sorted(set(per_model_total) | set(flagged)):
        n_total = per_model_total.get(model, 0)
        n_flag = len(flagged.get(model, []))
        rate = f"{100*n_flag/n_total:.2f}%" if n_total else "—"
        lines.append(f"| `{model}` | {n_total} | {n_flag} | {rate} |")
    lines.append("")

    if not any(flagged.values()):
        lines.append("_No `invalid_prompt` errors observed._\n")
    else:
        for model in sorted(flagged):
            if not flagged[model]:
                continue
            lines.append(f"\n## {model} — {len(flagged[model])} flagged\n")
            for f in flagged[model]:
                p = f["problem"] or {}
                lines.append(f"### `{f['id']}`  *(#{f['idx']} of {f['total']})*")
                lines.append("")
                if p:
                    lines.append(
                        f"- **Country**: {p.get('country','?')}   "
                        f"**Competition**: {p.get('competition','?')}   "
                        f"**Language**: {p.get('language','?')}   "
                        f"**Type**: {p.get('problem_type','?')}"
                    )
                    topics = p.get("topics_flat") or []
                    if topics:
                        lines.append(f"- **Topics**: {', '.join(topics)}")
                    lines.append(f"- **Gold answer**: `{p.get('final_answer','?')}`")
                lines.append("")
                lines.append(f"**OpenAI error**: {f['message']}")
                lines.append("")
                if p.get("problem_markdown"):
                    lines.append("**Problem**:\n")
                    lines.append("```")
                    lines.append(p["problem_markdown"])
                    lines.append("```")
                lines.append("\n---\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    total_flagged = sum(len(v) for v in flagged.values())
    print(f"flagged problems: {total_flagged}  across {len([m for m in flagged if flagged[m]])} model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
