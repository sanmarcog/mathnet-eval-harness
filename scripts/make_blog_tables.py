"""Render the two LinkedIn-blog tables as PNGs.

LinkedIn Articles strip HTML tables; image embeds are the reliable path.
Outputs match the project's Patagonia palette so they slot in next to
figures A/B.

    python scripts/make_blog_tables.py

Writes:
    results/figures/blog_scoreboard.png
    results/figures/blog_runs.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "results" / "figures"

CREAM = "#F3E8D0"
INK = "#2B241A"
SAND = "#E8D4A8"
ROW_LIGHT = "#FBF3DD"
ROW_DARK = "#F3E8D0"
HIGHLIGHT_GREEN = "#DCE8D2"
HIGHLIGHT_RUST = "#F4D5C2"


def render_table(headers, rows, col_widths, highlights, out_path, figsize, fontsize=11):
    """Render a styled table to PNG using matplotlib.table.

    highlights: dict mapping body-row-index (0-based) -> hex fill colour.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    fig.patch.set_facecolor(CREAM)

    n_cols = len(headers)
    cell_colours = []
    for i in range(len(rows)):
        if i in highlights:
            cell_colours.append([highlights[i]] * n_cols)
        else:
            cell_colours.append([ROW_LIGHT if i % 2 == 0 else ROW_DARK] * n_cols)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        cellColours=cell_colours,
        colColours=[SAND] * n_cols,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    # Row height scale: multi-line cells need extra vertical space.
    has_multiline = any("\n" in str(c) for row in rows for c in row)
    table.scale(1, 2.4 if has_multiline else 1.7)

    # styling pass
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(INK)
        cell.set_linewidth(0.5)
        cell.PAD = 0.04
        if row == 0:
            cell.set_text_props(weight="bold", color=INK)
            cell.set_height(cell.get_height() * 1.05)
        else:
            cell.set_text_props(color=INK)

    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=CREAM)
    plt.close(fig)
    print(f"wrote {out_path}")


def scoreboard_table():
    headers = ["Model", "N scored", "Accuracy"]
    col_widths = [0.62, 0.18, 0.20]
    rows = [
        ["Claude Opus 4.7  (spot-check)",            "100",       "84.0%"],
        ["Gemini 3 Pro  (partial)",                  "240 / 300", "73.3%"],
        ["Claude Sonnet 4.6",                         "500",       "65.0%"],
        ["GPT-5.4",                                   "495 / 500", "57.8%"],
        ["Qwen3-1.7B base  (open, thinking-on, 16K)", "500",       "36.8%"],
        ["GPT-5.4 Mini",                              "498 / 500", "36.7%"],
    ]
    # highlight the Qwen3 base row in green (the open-weights anchor)
    highlights = {4: HIGHLIGHT_GREEN}
    render_table(headers, rows, col_widths, highlights,
                 FIGURES / "blog_scoreboard.png",
                 figsize=(11, 3.3))


def runs_table():
    headers = ["Run", "Single change vs. previous", "Paired Δ vs. base"]
    col_widths = [0.08, 0.64, 0.28]
    rows = [
        ["Run 1", "tutorial-default QLoRA on Qwen2.5-1.5B-Instruct\n(LoRA r=16, LR 2e-4, eff batch 16)",                  "−6 pp   (n=150)"],
        ["Run B", "+ completion_only_loss=True\n(single-variable loss-masking fix)",                                      "−8 pp   (n=150)"],
        ["Run 2", "switched to Qwen3-1.7B; recipe-matched Alibaba Qwen2.5-Math 1.5B\n(LR cut 10×, batch up 8×, base swapped)",
                                                                                                                            "−33.8 pp   (n=500)"],
        ["Run 3", "+ boxed-answer augmentation",                                                                          "−33.0 pp   (n=500)"],
        ["Run 4", "self-distilled — base's own correct outputs,\nfull <think> traces preserved",                          "−8.0 pp   (n=500, p ≈ 10⁻⁴)"],
        ["Run 5", "Dr. GRPO bias-corrected RL\n(LR 1e-6, group=4, 200 steps, thinking-on)",                                "−4.2 pp   (n=500, p ≈ 0.024)"],
    ]
    # highlight Run 5 row (the latest result row)
    highlights = {5: HIGHLIGHT_RUST}
    render_table(headers, rows, col_widths, highlights,
                 FIGURES / "blog_runs.png",
                 figsize=(14, 4.6), fontsize=10.5)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    scoreboard_table()
    runs_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
