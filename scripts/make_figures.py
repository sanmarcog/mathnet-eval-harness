"""Generate the headline figures for the repo (scoreboard, grader paths,
architecture, data funnel). Single script, no extra dependencies beyond
matplotlib. Patagonia-inspired palette: warm cream background, rust
accents, forest/slate secondaries, deep warm-brown ink.

    python scripts/make_figures.py

Reads per-model accuracy / method counts from results/full/*/summary.json.
Writes PNGs to results/figures/ and docs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGURES = RESULTS / "figures"
DOCS = REPO / "docs"

CREAM = "#F3E8D0"
INK = "#2B241A"
RUST = "#BC4E30"
FOREST = "#3A5A40"
SLATE = "#4A6C8C"
TAN = "#C99E6B"
GRID = "#D4C6A8"
SAND = "#E8D4A8"

# grader-path sequential ramp: light (exact, cheap) -> dark (judge, expensive)
# -> red (miss, wrong)
RAMP = {
    "exact": "#EED9A8",
    "normalized": "#C9A573",
    "symbolic": "#A06E47",
    "judge": "#604530",
    "miss": "#8B2E1F",
}

MODEL_ORDER = [
    ("opus-4-7", "Claude Opus 4.7"),
    ("gemini-3-pro", "Gemini 3 Pro"),
    ("sonnet-4-6", "Claude Sonnet 4.6"),
    ("gpt-5.4", "GPT-5.4"),
    ("gpt-5.4-mini", "GPT-5.4 Mini"),
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


def load_summaries() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for slug, _ in MODEL_ORDER:
        p = RESULTS / "full" / slug / "summary.json"
        out[slug] = json.loads(p.read_text())
    return out


def scoreboard() -> None:
    data = load_summaries()
    labels = [name for _, name in MODEL_ORDER]
    accs = [data[s]["accuracy"] * 100 for s, _ in MODEL_ORDER]
    ns = [data[s]["n_scored"] for s, _ in MODEL_ORDER]

    labels = labels + ["Qwen-2.5-1.5B QLoRA  (ours)"]
    accs = accs + [0]
    ns = ns + [None]
    ys = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    bar_colors = [RUST] * 5 + [GRID]
    bars = ax.barh(ys, accs, color=bar_colors, edgecolor=INK, linewidth=0.8, height=0.72)

    for i, (bar, acc, n) in enumerate(zip(bars, accs, ns)):
        if n is None:
            ax.text(1.0, bar.get_y() + bar.get_height() / 2,
                    "  pending Run 2",
                    va="center", ha="left", color=INK, fontsize=10, fontstyle="italic")
            continue
        ax.text(acc + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%   (N={n})",
                va="center", ha="left", color=INK, fontsize=10)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("MathNet accuracy (%)", fontsize=11)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Frontier models on 500 MathNet problems", fontsize=14, pad=14, loc="left")

    fig.text(0.01, 0.03,
             "Opus on N=100 spot-check; Gemini with thinking_budget=4096 on partial N=240; "
             "denominators are n_scored.",
             fontsize=8.5, color=INK, alpha=0.75)

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    out = FIGURES / "scoreboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def grader_paths() -> None:
    data = load_summaries()
    paths = ["exact", "normalized", "symbolic", "judge", "miss"]
    labels = [name for _, name in MODEL_ORDER]
    matrix = []
    for slug, _ in MODEL_ORDER:
        mc = data[slug]["method_counts"]
        n = data[slug]["n_scored"]
        row = [mc.get(p, 0) / n * 100 for p in paths]
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ys = np.arange(len(labels))[::-1]
    lefts = np.zeros(len(labels))
    for i, p in enumerate(paths):
        widths = matrix[:, i]
        ax.barh(ys, widths, left=lefts, color=RAMP[p],
                edgecolor=INK, linewidth=0.5, height=0.72, label=p)
        for y, w, left in zip(ys, widths, lefts):
            if w >= 6:
                ax.text(left + w / 2, y, f"{w:.0f}%", va="center", ha="center",
                        color=INK if p in ("exact", "normalized") else CREAM,
                        fontsize=9)
        lefts += widths

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of problems (%)", fontsize=11)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Grader-path distribution per model", fontsize=14, pad=14, loc="left")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    legend_labels = [
        ("exact",      "exact string match"),
        ("normalized", "normalized (latex, case, punct)"),
        ("symbolic",   "symbolic (sympy)"),
        ("judge",      "LLM judge (Sonnet 4.6)"),
        ("miss",       "miss (wrong or unparseable)"),
    ]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=RAMP[k], edgecolor=INK, linewidth=0.5)
               for k, _ in legend_labels]
    ax.legend(handles, [f"{k}  —  {desc}" for k, desc in legend_labels],
              loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=3, frameon=False, fontsize=9.5)

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    out = FIGURES / "grader_paths.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _rbox(ax, x, y, w, h, text, fc=SAND, ec=INK, fontsize=10, fontweight="normal"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                         facecolor=fc, edgecolor=ec, linewidth=1.1)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, fontweight=fontweight,
            wrap=True)
    return (x + w / 2, y + h / 2)


def _arrow(ax, x1, y1, x2, y2, color=INK):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>,head_length=7,head_width=5",
                        linewidth=1.3, color=color, mutation_scale=1)
    ax.add_patch(a)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_axis_off()

    ax.text(0.25, 7.55, "Pipeline", fontsize=16, fontweight="bold", color=INK)

    # Row 1: data
    _rbox(ax, 0.3, 5.9, 2.2, 1.0,
          "MathNet\n27,817 problems", fc=SAND, fontsize=10.5, fontweight="bold")
    _rbox(ax, 3.2, 5.9, 2.2, 1.0,
          "Filter\ntext / English / has-answer", fc=CREAM, fontsize=10)
    _rbox(ax, 6.1, 5.9, 2.6, 1.0,
          "Stratified split\n500 eval  /  3,596 en train\n14,585 multilingual train",
          fc=CREAM, fontsize=9.5)
    _arrow(ax, 2.5, 6.4, 3.2, 6.4)
    _arrow(ax, 5.4, 6.4, 6.1, 6.4)

    # Row 2: inference harness (container + header + provider sub-boxes)
    _rbox(ax, 0.3, 3.0, 8.4, 2.0,
          "", fc="#DCE4D7", fontsize=1)
    ax.text(4.5, 4.65, "Inference harness",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(4.5, 4.25,
            "unified client  .  tenacity retry  .  SHA-256 disk cache  .  per-model log",
            ha="center", va="center", fontsize=9.2, color=INK, alpha=0.8)
    providers = ["Anthropic", "OpenAI", "Google", "local HF", "vLLM"]
    px = 0.5
    py = 3.2
    pw = 1.55
    ph = 0.65
    for name in providers:
        _rbox(ax, px, py, pw, ph, name, fc=CREAM, fontsize=9.5)
        px += pw + 0.08
    _arrow(ax, 7.4, 5.9, 5.0, 5.0)

    # Row 3: grader
    _rbox(ax, 0.3, 1.4, 8.4, 1.1,
          "", fc="#E3D5C2", fontsize=1)
    ax.text(4.5, 2.20, "4-layer grader",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(4.5, 1.75,
            "exact  >  normalized  >  sympy  >  LLM judge",
            ha="center", va="center", fontsize=10.5, color=INK, alpha=0.85)
    _arrow(ax, 4.5, 3.0, 4.5, 2.5)

    # Row 3b: sink
    _rbox(ax, 9.0, 1.4, 2.8, 1.1,
          "Per-problem\nJSON results  +  summary.json",
          fc=SAND, fontsize=9.8, fontweight="bold")
    _arrow(ax, 8.7, 1.95, 9.0, 1.95)

    ax.text(0.3, 0.55,
            "Cache key: SHA-256(model  .  prompt  .  sampling params) "
            "— survives preemption, makes reruns free.",
            fontsize=9, color=INK, alpha=0.8, fontstyle="italic")

    out = DOCS / "architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=CREAM)
    plt.close(fig)
    print(f"wrote {out}")


def data_funnel() -> None:
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.set_axis_off()

    ax.text(0.5, 12.4, "Data funnel", fontsize=16, fontweight="bold", color=INK)
    ax.text(0.5, 11.9, "from MathNet to the Week-1 train / eval splits",
            fontsize=10.5, color=INK, alpha=0.75, fontstyle="italic")

    stages = [
        (11.0, "MathNet (full)", "27,817", "all competitions, languages, modalities", SAND),
        (9.2, "Text + has_final_answer", "22,669",
         "drop non-textual and answerless rows", CREAM),
        (7.4, "English only", "6,708",
         "Week-1 scope: single-language for clean grading", CREAM),
        (5.6, "Stratified pool", "4,096",
         "dedup by competition, keep evaluable set", CREAM),
    ]
    box_x = 2.5
    box_w = 5.2
    box_h = 1.1
    for y, title, number, detail, fc in stages:
        _rbox(ax, box_x, y - box_h / 2, box_w, box_h,
              f"{title}\n{number}", fc=fc, fontsize=11, fontweight="bold")
        ax.text(box_x + box_w + 0.2, y, detail,
                fontsize=9.5, color=INK, alpha=0.8, va="center")

    # arrows between main stages
    for (y1, *_), (y2, *_) in zip(stages[:-1], stages[1:]):
        _arrow(ax, box_x + box_w / 2, y1 - box_h / 2, box_x + box_w / 2, y2 + box_h / 2)

    # Splits row (two boxes at bottom)
    ax.text(box_x + box_w / 2, 4.3, "split",
            ha="center", fontsize=9, color=INK, alpha=0.7)
    _arrow(ax, box_x + box_w / 2, 5.05, 2.1, 3.6)
    _arrow(ax, box_x + box_w / 2, 5.05, 7.9, 3.6)

    _rbox(ax, 1.0, 2.6, 2.8, 1.1, "Eval split\n500", fc=RUST, ec=INK, fontsize=11, fontweight="bold")
    ax.text(1.0, 2.3, "held out for scoring all models", fontsize=9, color=INK, alpha=0.75)

    _rbox(ax, 6.5, 2.6, 2.8, 1.1, "Train (English)\n3,596", fc=FOREST, ec=INK, fontsize=11, fontweight="bold")
    for patch in ax.patches:  # set white text on dark fills
        pass
    # write over the eval / train boxes with cream text
    ax.texts[-3].set_color(CREAM)  # "Eval split\n500"
    ax.texts[-1].set_color(CREAM)  # "Train (English)\n3,596"

    ax.text(6.5, 2.3, "QLoRA fine-tune base signal", fontsize=9, color=INK, alpha=0.75)

    # multilingual augmentation branching off train
    _arrow(ax, 7.9, 2.6, 7.9, 1.5)
    _rbox(ax, 6.0, 0.4, 3.8, 1.1,
          "Train (multilingual aug.)\n14,585",
          fc=SLATE, ec=INK, fontsize=11, fontweight="bold")
    ax.texts[-1].set_color(CREAM)
    ax.text(6.0, 0.05, "4× via non-English MathNet rows (Run 2)", fontsize=8.8, color=INK, alpha=0.75)

    out = DOCS / "data_funnel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=CREAM)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    base_style()
    scoreboard()
    grader_paths()
    architecture()
    data_funnel()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
