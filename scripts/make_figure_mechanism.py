#!/usr/bin/env python3
"""Paper figure 2 — the mechanism: WHY predictive coding helps only where it does.

(A) Sleep order-shuffle control  — the gain is causally temporal
(B) Emotion smoothing flip       — remove the dynamics, the gain goes with them
(C) Seizure paired effects       — what is (and isn't) statistically real

    python scripts/make_figure_mechanism.py --out results/figures/fig_mechanism.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_PC, C_RAW, C_RAND = "#2a78d6", "#1baf7a", "#eda100"
C_SIG, C_NS = "#2a78d6", "#9a9995"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"

# (A) sleep shuffle control — accuracy %, EXP-0009 §4e
SHUFFLE = {"PhysioFM-S": (72.6, 67.4, C_PC),
           "raw-DE": (67.9, 67.9, C_RAW),
           "random-init": (62.9, 61.2, C_RAND)}
# (B) emotion smoothing flip — SEED-IV, SAME trials/labels/folds, only the feature
# variant changes. Matched protocol, 3-seed means (scripts/run_parity.sh).
FLIP = [("LDS-smoothed DE\n(0.08% within-trial variance)", 2.38),
        ("un-smoothed DE\n(17.6% within-trial variance)", 11.01)]
# (C) seizure paired per-patient effects (AUC diff), EXP-0015 §4c
PAIRED = [  # label, diff, p, wins, n
    ("vs random-init\n@ 1% labels",   0.105, 0.0002, 22),
    ("vs random-init\n@ 100% labels", 0.082, 0.0063, 17),
    ("vs raw-DE\n@ 1% labels",        0.066, 0.0166, 18),
    ("vs raw-DE\n@ 100% labels",      0.016, 0.4611, 11),
]


def _style(ax, xgrid=False):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=1.0)
    ax.grid(True, axis="x" if xgrid else "y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/fig_mechanism.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), facecolor=SURFACE)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.78, bottom=0.20, wspace=0.50)

    # ---- (A) shuffle control -------------------------------------------------
    ax = axes[0]
    xs = np.arange(len(SHUFFLE)); w = 0.34
    for i, (name, (normal, shuf, col)) in enumerate(SHUFFLE.items()):
        ax.bar(i - w / 2, normal, w, color=col, edgecolor=SURFACE, lw=2.0, zorder=3)
        ax.bar(i + w / 2, shuf, w, color=col, edgecolor=SURFACE, lw=2.0, zorder=3,
               alpha=0.42, hatch="////")
        ax.annotate(f"{normal:.1f}", (i - w / 2, normal), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        ax.annotate(f"{shuf:.1f}", (i + w / 2, shuf), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    # the collapse arrow: PC falls to the raw-DE line when time is scrambled
    ax.axhline(67.9, color=C_RAW, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.annotate("", xy=(0 + w / 2, 68.2), xytext=(0 - w / 2, 72.2),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4,
                                connectionstyle="arc3,rad=-0.25"), zorder=5)
    ax.annotate("shuffling time erases\nPC's entire edge —\ndown to the raw-DE line",
                xy=(0.42, 70.6), fontsize=8.3, color=INK2, style="italic")
    ax.set_xticks(xs); ax.set_xticklabels(list(SHUFFLE), fontsize=9, color=INK2)
    ax.set_ylim(55, 77); ax.set_ylabel("Sleep accuracy (%)", fontsize=9.5, color=INK2)
    ax.set_title("A · Order-shuffle control (sleep, per-epoch labels)", fontsize=11, color=INK,
                 fontweight="bold", loc="left", pad=10)
    _style(ax); ax.grid(False, axis="x")
    solid = matplotlib.patches.Patch(fc=INK2, ec=SURFACE, label="normal")
    hatch = matplotlib.patches.Patch(fc=INK2, ec=SURFACE, alpha=0.42, hatch="////",
                                     label="epoch order shuffled")
    ax.legend(handles=[solid, hatch], loc="lower left", frameon=False, fontsize=8.3,
              labelcolor=INK2, handlelength=1.4)

    # ---- (B) smoothing flip --------------------------------------------------
    ax = axes[1]
    labs = [f[0] for f in FLIP]; vals = [f[1] for f in FLIP]
    cols = ["#9ec5f4", C_PC]
    ax.bar(np.arange(2), vals, width=0.55, color=cols, edgecolor=SURFACE, lw=2.0, zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(f"+{v:.1f}", (i, v), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.annotate("", xy=(1, 10.0), xytext=(0, 3.4),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4,
                                connectionstyle="arc3,rad=0.22"), zorder=5)
    ax.annotate("restore the dynamics,\nthe PC gain returns (4.6×)",
                xy=(0.28, 8.2), fontsize=8.3, color=INK2, style="italic")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labs, fontsize=8.5, color=INK2)
    ax.set_ylim(0, 13.5)
    ax.set_ylabel("PC − random-init (accuracy points)", fontsize=9.5, color=INK2)
    ax.set_title("B · Smoothing flip (emotion, SEED-IV)", fontsize=11, color=INK,
                 fontweight="bold", loc="left", pad=10)
    _style(ax); ax.grid(False, axis="x")

    # ---- (C) paired effects --------------------------------------------------
    ax = axes[2]
    ys = np.arange(len(PAIRED))[::-1]
    for y, (lab, diff, p, wins) in zip(ys, PAIRED):
        sig = p < 0.05
        col = C_SIG if sig else C_NS
        ax.plot([0, diff], [y, y], color=col, lw=2.0, zorder=3, alpha=0.55)
        ax.plot(diff, y, "o", ms=10, mfc=col, mec=SURFACE, mew=1.6, zorder=4)
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.annotate(f"{stars}   p={p:.3f}   {wins}/24 patients",
                    xy=(diff, y), xytext=(14, 0), textcoords="offset points",
                    va="center", fontsize=8.4, color=INK if sig else C_NS,
                    fontweight="bold" if sig else "normal")
    ax.axvline(0, color=INK2, lw=1.2, zorder=2)
    ax.set_yticks(ys); ax.set_yticklabels([p[0] for p in PAIRED], fontsize=8.6, color=INK2)
    ax.set_xlim(-0.012, 0.235)
    ax.set_ylim(-0.75, 3.5)
    ax.set_xlabel("Paired ΔAUC (PC − comparator), 24 matched patients", fontsize=9.5, color=INK2)
    ax.set_title("C · What is statistically real (seizure)", fontsize=11, color=INK,
                 fontweight="bold", loc="left", pad=10)
    _style(ax, xgrid=True); ax.grid(False, axis="y")
    ax.annotate("the only comparison that is NOT significant:\n"
                "PC ties the linear ceiling when labels are abundant",
                xy=(0.012, 0.0), xytext=(0.052, -0.58), fontsize=8.0, color=C_NS,
                style="italic", va="center",
                arrowprops=dict(arrowstyle="->", color=C_NS, lw=1.0,
                                connectionstyle="arc3,rad=0.2"))

    fig.suptitle("The mechanism: on per-epoch-label tasks the gain is temporal — destroy the dynamics "
                 "and it disappears; restore them and it returns",
                 fontsize=12.5, color=INK, fontweight="bold", x=0.045, ha="left", y=0.945)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
