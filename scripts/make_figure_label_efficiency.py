#!/usr/bin/env python3
"""Paper figure: the label-efficiency story + the temporal-vs-spectral 2x2.

(A) Sleep label-efficiency   (B) Seizure label-efficiency   (C) Pretraining gain by task

    python scripts/make_figure_label_efficiency.py --out results/figures/fig_label_efficiency.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# validated categorical slots 1/2/3 (see dataviz palette); diverging blue<->red for (C)
C_PC, C_RAW, C_RAND = "#2a78d6", "#1baf7a", "#eda100"
C_POS, C_NEG, C_ZERO = "#2a78d6", "#e34948", "#f0efec"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"

FRACS = [1, 5, 10, 25, 50, 100]
# (A) sleep — accuracy %, EXP-0009 §4c
SLEEP = {"pc": [70.94, 72.51, 72.64, 72.75, 72.65, 72.63],
         "raw": [67.04, 67.80, 67.66, 67.86, 67.85, 67.86],
         "rand": [58.20, 60.75, 61.89, 62.51, 62.79, 62.86]}
# (B) seizure — ROC-AUC, EXP-0015 §4c
SEIZ = {"pc": [0.797, 0.809, 0.808, 0.814, 0.820, 0.822],
        "raw": [0.731, 0.790, 0.795, 0.803, 0.804, 0.806],
        "rand": [0.692, 0.740, 0.734, 0.740, 0.737, 0.740]}
# (C) PC - random at full labels, in each task's native accuracy points
GAIN = [("Sleep\n(temporal)", 9.77), ("Seizure\n(temporal)", 8.10),
        ("Motor imagery\n(spectral)", -1.27), ("Emotion\n(spectral)", -3.18)]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=1.0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _curve(ax, data, ylab, title, fmt, label_dy):
    for key, color, name in (("pc", C_PC, "PhysioFM-S (PC-pretrained)"),
                             ("raw", C_RAW, "raw-DE (linear ceiling)"),
                             ("rand", C_RAND, "random-init (no pretraining)")):
        y = data[key]
        ax.plot(FRACS, y, color=color, lw=2.0, marker="o", ms=5.5,
                mfc=color, mec=SURFACE, mew=1.4, zorder=3, label=name, clip_on=False)
        # direct label at the right end (relief for the low-contrast slots)
        ax.annotate(name.split(" (")[0], xy=(FRACS[-1], y[-1]),
                    xytext=(8, label_dy[key]), textcoords="offset points",
                    color=color, fontsize=8.5, fontweight="bold", va="center")
    ax.set_xscale("log")
    ax.set_xticks(FRACS)
    ax.set_xticklabels([f"{f}%" for f in FRACS])
    ax.minorticks_off()
    ax.set_xlabel("Labelled training data (log scale)", fontsize=9.5, color=INK2)
    ax.set_ylabel(ylab, fontsize=9.5, color=INK2)
    ax.set_title(title, fontsize=11, color=INK, fontweight="bold", loc="left", pad=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
    _style(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/fig_label_efficiency.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.05, right=0.965, top=0.80, bottom=0.16, wspace=0.55)

    # (A) sleep
    _curve(axes[0], SLEEP, "Accuracy (%)", "A · Sleep staging — label efficiency",
           lambda v, _: f"{v:.0f}", {"pc": 6, "raw": 0, "rand": -4})
    axes[0].set_ylim(55, 76)

    # (B) seizure
    _curve(axes[1], SEIZ, "ROC-AUC", "B · Seizure detection — label efficiency",
           lambda v, _: f"{v:.2f}", {"pc": 6, "raw": -2, "rand": -6})
    axes[1].set_ylim(0.66, 0.845)

    # (C) pretraining gain — diverging by polarity, zero reference
    ax = axes[2]
    names = [g[0] for g in GAIN]
    vals = [g[1] for g in GAIN]
    xs = np.arange(len(vals))
    cols = [C_POS if v > 0 else C_NEG for v in vals]
    ax.bar(xs, vals, width=0.62, color=cols, edgecolor=SURFACE, linewidth=2.0, zorder=3)
    ax.axhline(0, color=INK2, lw=1.2, zorder=4)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:+.1f}", xy=(x, v), xytext=(0, 5 if v > 0 else -13),
                    textcoords="offset points", ha="center", fontsize=9.5,
                    fontweight="bold", color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8.8, color=INK2)
    ax.set_ylabel("PC − random-init (accuracy points)", fontsize=9.5, color=INK2)
    ax.set_title("C · Pretraining gain: temporal vs spectral", fontsize=11,
                 color=INK, fontweight="bold", loc="left", pad=10)
    ax.set_ylim(-6, 13)
    _style(ax)
    ax.grid(False, axis="x")
    ax.annotate("PC pretraining helps", xy=(0.5, 11.4), fontsize=8.5, color=C_POS,
                ha="center", style="italic")
    ax.annotate("null", xy=(2.5, -5.0), fontsize=8.5, color=C_NEG, ha="center", style="italic")

    fig.suptitle("Predictive-coding pretraining pays off in proportion to a task's temporal structure — "
                 "and most where labels are scarce",
                 fontsize=12.5, color=INK, fontweight="bold", x=0.05, ha="left", y=0.955)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
