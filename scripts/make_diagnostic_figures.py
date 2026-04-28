"""Diagnostic figures supporting the negative result.

Three figures, all reading the per-problem `*.graded.json` outputs:

  A. Miss-mode decomposition  —  base + Run 2/3/4 broken into 4 outcome
     buckets. The `saturated, never boxed` segment is the convergence-
     failure mode. If it grows from base to Run 4, the fine-tune trained
     the model to think *longer* rather than to think *better* — which is
     the central diagnosis of this writeup.

  B. Run 4 vs base transition matrix  —  paired 2×2 over the 500
     problems. Highlights regressions vs improvements; the gap is the
     paired delta.

  C. Per-competition delta  —  ft_correct - base_correct, sorted, with
     a minimum-N threshold so single-problem competitions don't dominate.

Saturation cutoff is 16,384 (the actual eval ceiling, identical across
all four Qwen3-1.7B runs — verified from the sbatch configs).

    python scripts/make_diagnostic_figures.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from mathnet_eval import SATURATION_CUTOFF

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGURES = RESULTS / "figures"

CREAM = "#F3E8D0"
INK = "#2B241A"
RUST = "#BC4E30"
FOREST = "#3A5A40"
SLATE = "#4A6C8C"
TAN = "#C99E6B"
GRID = "#D4C6A8"
SAND = "#E8D4A8"
DEEP_RED = "#8B2E1F"

SAT_CUTOFF = SATURATION_CUTOFF

RUNS = [
    ("qwen3-1.7b-base", "Qwen3-1.7B base"),
    ("qwen3-1.7b-run2", "Run 2"),
    ("qwen3-1.7b-run3", "Run 3"),
    ("qwen3-1.7b-run4", "Run 4"),
]


def base_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": CREAM,
        "axes.facecolor": CREAM,
        "savefig.facecolor": CREAM,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
    })


def load_graded(d: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not d.exists():
        return out
    for f in d.glob("*.graded.json"):
        if f.name.startswith("summary"):
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        pid = data.get("id")
        if pid is None:
            continue
        method = (data.get("grade") or {}).get("method", "miss")
        out[pid] = {
            "correct": method != "miss",
            "method": method,
            "competition": (data.get("competition") or "?").strip() or "?",
            "output_tokens": (data.get("usage") or {}).get("output_tokens", 0),
            "has_boxed": "\\boxed{" in (data.get("response_text") or ""),
        }
    return out


def classify_outcome(row: dict) -> str:
    if row["correct"]:
        return "correct"
    if row["output_tokens"] >= SAT_CUTOFF:
        return "sat_no_boxed" if not row["has_boxed"] else "sat_boxed_wrong"
    return "wrong_committed"


def figure_a_miss_mode() -> Path | None:
    rows = []
    for slug, label in RUNS:
        graded = load_graded(RESULTS / "full" / slug)
        if not graded:
            print(f"[A] skipping {slug}: no graded JSONs in {RESULTS / 'full' / slug}")
            continue
        counts: dict[str, int] = defaultdict(int)
        for r in graded.values():
            counts[classify_outcome(r)] += 1
        rows.append((slug, label, len(graded), counts))

    if len(rows) < 2:
        print("[A] need at least 2 runs with data; aborting")
        return None

    keys = ["correct", "wrong_committed", "sat_boxed_wrong", "sat_no_boxed"]
    palette = {
        "correct":         FOREST,
        "wrong_committed": TAN,
        "sat_boxed_wrong": RUST,
        "sat_no_boxed":    DEEP_RED,
    }
    pretty = {
        "correct":         "correct",
        "wrong_committed": "wrong, committed (output < 16K)",
        "sat_boxed_wrong": "saturated at 16K, boxed but wrong",
        "sat_no_boxed":    "saturated at 16K, never boxed  (convergence failure)",
    }

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ys = np.arange(len(rows))[::-1]
    lefts = np.zeros(len(rows))
    for k in keys:
        widths = np.array([r[3].get(k, 0) / r[2] * 100 for r in rows])
        ax.barh(ys, widths, left=lefts, color=palette[k],
                edgecolor=INK, linewidth=0.5, height=0.65)
        for y, w, left in zip(ys, widths, lefts):
            if w >= 5:
                ax.text(left + w / 2, y, f"{w:.0f}%",
                        va="center", ha="center",
                        color=INK if k in ("correct", "wrong_committed") else CREAM,
                        fontsize=10, fontweight="bold" if k == "sat_no_boxed" else "normal")
        lefts += widths

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[1]}  (N={r[2]})" for r in rows], fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of problems (%)", fontsize=11)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_title(
        "Outcome decomposition: correct  /  wrong-but-committed  /  saturated-without-answer",
        fontsize=13, pad=14, loc="left",
    )
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=palette[k],
                             edgecolor=INK, linewidth=0.5) for k in keys]
    ax.legend(handles, [pretty[k] for k in keys],
              loc="lower center", bbox_to_anchor=(0.5, -0.36),
              ncol=2, frameon=False, fontsize=9.5)

    fig.text(0.01, 0.04,
             "All four runs use the same eval config (vLLM, thinking-on, max_new_tokens=16384, "
             "temperature=0). Saturated = output_tokens ≥ 16,384. The deep-red segment is the "
             "convergence-failure mode the writeup investigates.",
             fontsize=8.5, color=INK, alpha=0.75)

    plt.tight_layout(rect=(0, 0.10, 1, 1))
    out = FIGURES / "miss_mode_decomposition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def figure_b_transition_matrix(
    ft_slug: str = "qwen3-1.7b-run4",
    base_slug: str = "qwen3-1.7b-base",
) -> Path | None:
    base = load_graded(RESULTS / "full" / base_slug)
    ft = load_graded(RESULTS / "full" / ft_slug)
    shared = sorted(set(base) & set(ft))
    if not shared:
        print(f"[B] no shared IDs between {ft_slug} and {base_slug}")
        return None

    n = len(shared)
    both_correct = sum(1 for i in shared if base[i]["correct"] and ft[i]["correct"])
    base_only    = sum(1 for i in shared if base[i]["correct"] and not ft[i]["correct"])
    ft_only      = sum(1 for i in shared if not base[i]["correct"] and ft[i]["correct"])
    neither      = sum(1 for i in shared if not base[i]["correct"] and not ft[i]["correct"])

    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    ax.set_xlim(-0.20, 2.20)
    ax.set_ylim(-0.20, 2.40)
    ax.set_axis_off()

    # rows: Run 4 ✓ (top), Run 4 ✗ (bottom)   cols: base ✓ (left), base ✗ (right)
    cells = [
        (0, 1, both_correct, "both correct",          FOREST,  CREAM),
        (1, 1, base_only,    "base only\n(regressed)", DEEP_RED, CREAM),
        (0, 0, ft_only,      "Run 4 only\n(improved)", TAN,     INK),
        (1, 0, neither,      "neither",                GRID,    INK),
    ]
    for col, row, count, label, fc, txt_color in cells:
        x = col * 1.05
        y = row * 1.05
        ax.add_patch(Rectangle((x, y), 1.0, 1.0,
                               facecolor=fc, edgecolor=INK, linewidth=1.2))
        ax.text(x + 0.5, y + 0.62, str(count),
                ha="center", va="center", fontsize=30, fontweight="bold", color=txt_color)
        ax.text(x + 0.5, y + 0.26, label,
                ha="center", va="center", fontsize=11, color=txt_color)

    ax.text(0.5,  2.18, "base ✓", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(1.55, 2.18, "base ✗", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(-0.12, 1.55, "Run 4 ✓", ha="right", va="center",
            fontsize=12, fontweight="bold", rotation=90)
    ax.text(-0.12, 0.55, "Run 4 ✗", ha="right", va="center",
            fontsize=12, fontweight="bold", rotation=90)

    delta_pp = (ft_only - base_only) / n * 100
    ax.set_title(
        f"Paired transition matrix — Run 4 vs base, n={n}\n"
        f"regressions {base_only}  vs  improvements {ft_only}    →    "
        f"net Δ = {delta_pp:+.1f} pp",
        fontsize=12.5, pad=10, loc="left",
    )

    out = FIGURES / "transition_matrix_run4_vs_base.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def figure_c_per_topic_delta(
    ft_slug: str = "qwen3-1.7b-run4",
    base_slug: str = "qwen3-1.7b-base",
    min_n: int = 3,
) -> Path | None:
    base = load_graded(RESULTS / "full" / base_slug)
    ft = load_graded(RESULTS / "full" / ft_slug)
    shared = sorted(set(base) & set(ft))
    if not shared:
        print(f"[C] no shared IDs between {ft_slug} and {base_slug}")
        return None

    by_comp: dict[str, dict] = defaultdict(lambda: {"n": 0, "ft": 0, "base": 0})
    for i in shared:
        c = base[i]["competition"] or ft[i]["competition"] or "?"
        by_comp[c]["n"] += 1
        by_comp[c]["ft"] += int(ft[i]["correct"])
        by_comp[c]["base"] += int(base[i]["correct"])

    rows = [(c, v["n"], v["ft"] - v["base"]) for c, v in by_comp.items() if v["n"] >= min_n]
    rows.sort(key=lambda r: r[2])
    if not rows:
        print(f"[C] no competitions with n>={min_n}")
        return None

    fig, ax = plt.subplots(figsize=(10, max(4.5, len(rows) * 0.32)))
    ys = np.arange(len(rows))[::-1]
    deltas = [r[2] for r in rows]
    colors = [DEEP_RED if d < 0 else (FOREST if d > 0 else GRID) for d in deltas]
    ax.barh(ys, deltas, color=colors, edgecolor=INK, linewidth=0.6, height=0.72)
    for y, (c, n, d) in zip(ys, rows):
        if d >= 0:
            ax.text(d + 0.08, y, f"  {d:+d}  (N={n})",
                    va="center", ha="left", fontsize=8.5, color=INK)
        else:
            ax.text(d - 0.08, y, f"{d:+d}  (N={n})  ",
                    va="center", ha="right", fontsize=8.5, color=INK)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0][:60] for r in rows], fontsize=8.5)
    ax.set_xlabel("Run 4 correct − base correct  (problems)", fontsize=11)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_title(
        f"Per-competition delta, Run 4 vs base  (competitions with N ≥ {min_n})",
        fontsize=13, pad=12, loc="left",
    )
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    pad = max(abs(min(deltas)), max(deltas)) + 1
    ax.set_xlim(-pad, pad)

    plt.tight_layout()
    out = FIGURES / "per_competition_delta_run4.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    base_style()
    figure_a_miss_mode()
    figure_b_transition_matrix()
    figure_c_per_topic_delta()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
