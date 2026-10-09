"""Regenerate the selection figure with no OOD-informed quantity and no hard-coding.

The previous version set the "deployed" marker to the maximum OOD lift over the
four families and hard-coded three datasets' values plus Adult's floor and
ceiling. Everything here is read from exp8_selection_summary.csv, which applies
a source-domain cross-validation rule over the whole 48-cell grid.

Three things are drawn per dataset:
  * the candidate range, floor to ceiling, over all 48 (variant, family) cells
  * the four within-family deployed lifts -- what Tables 2 and 3 report
  * the single cell a cross-family source-CV rule selects

The gap between the last two markers and the ceiling is the point of the
figure: source-domain selection does not reach the best candidate, and the
accompanying margin/SE statistics show it cannot be expected to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

RESULTS = Path(sys.argv[1])
OUTDIR = Path(sys.argv[2])

VERMILLION = "#D55E00"
BLUISH_GREEN = "#009E73"
GREY = "#777777"
BLUE = "#0072B2"

DS_ORDER = ["Boston Housing", "Cancer", "Heart Disease", "Diabetes", "Adult"]
FAMS = [("prior_mean", "Prior-Mean"), ("independent", "Independent-Prior"),
        ("correlated", "Correlated-Prior"), ("mixed", "Mixed")]

summary = pd.read_csv(RESULTS / "exp8_selection_summary.csv").set_index("dataset")

fig, ax = plt.subplots(figsize=(7.6, 4.1))

for i, ds in enumerate(DS_ORDER):
    s = summary.loc[ds]
    ax.plot([i, i], [s.floor, s.ceiling], color=GREY, lw=1.4,
            solid_capstyle="round", zorder=2)
    ax.plot([i], [s.floor], "_", ms=13, color=GREY, mew=1.6, zorder=3)
    ax.plot([i], [s.ceiling], "_", ms=13, color=GREY, mew=1.6, zorder=3)

    for k, (fam, _) in enumerate(FAMS):
        ax.plot([i - 0.17 + 0.113 * k], [s[f"within_{fam}_lift"]], "o", ms=4.0,
                color=BLUE, alpha=0.85, markeredgecolor="white", mew=0.5, zorder=4)

    ax.plot([i], [s.cross_family_lift], "D", ms=8, color=BLUISH_GREEN,
            markeredgecolor="black", mew=0.7, zorder=5)
    ax.annotate(f"{s.cross_family_lift:+.1f}%", (i + 0.17, s.cross_family_lift),
                va="center", ha="left", fontsize=8)
    ax.annotate(f"{s.ceiling:+.1f}%", (i, s.ceiling), xytext=(0, 7),
                textcoords="offset points", va="bottom", ha="center",
                fontsize=7.5, color=GREY)
    ax.annotate(f"{s.floor:+.1f}%", (i, s.floor), xytext=(0, -8),
                textcoords="offset points", va="top", ha="center",
                fontsize=7.5, color=GREY)

ax.axhline(0, color="black", lw=0.9, zorder=1)
ax.set_xticks(range(len(DS_ORDER)))
ax.set_xticklabels(DS_ORDER)
ax.set_ylabel("OOD lift over baseline (%)")
ax.set_xlim(-0.5, len(DS_ORDER) - 0.35)
ax.margins(y=0.16)

ax.legend(handles=[
    Line2D([], [], color=GREY, lw=1.4, label="range over all 48 candidates"),
    Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor="white",
           mew=0.5, label="within-family deployed (Tables 2-3)"),
    Line2D([], [], marker="D", ls="", color=BLUISH_GREEN,
           markeredgecolor="black", mew=0.7,
           label="cross-family, selected on source CV"),
], loc="upper right", frameon=False, fontsize=8)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUTDIR / f"selection_value.{ext}", dpi=200, bbox_inches="tight")
print("wrote", OUTDIR / "selection_value.pdf")
