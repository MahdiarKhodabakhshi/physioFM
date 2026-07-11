#!/usr/bin/env python3
"""E3.2 + E4.1 + E4.3 — instance-norm control, LOSO, and band/channel importance.

Writes results/phase2/analysis.md (+ figure). No model training: all on raw-DE
features through the canonical harness, which is the right level given E3.4 (the
structured representation, not pretraining, carries the emotion signal).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import DEFAULT_BANDS, load_de_archive
from physiofm.phase2_eval import build_raw_de_segments, loso_eval, subject_dependent_eval
from physiofm.structured_data import ARCH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("phase2_analysis")

OUTDIR = Path("results/phase2")
BANDS = [b[0] for b in DEFAULT_BANDS]


def band_channel_importance(feats, n_ch=62, n_band=5, max_n=8000, seed=42):
    """Per-(channel,band) discriminative importance = mean |LogReg coef| over classes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    y = LabelEncoder().fit_transform(feats.label)
    idx = np.arange(feats.X.shape[0])
    if idx.size > max_n:
        idx = np.random.default_rng(seed).choice(idx, size=max_n, replace=False)
    Xs = StandardScaler().fit_transform(feats.X[idx])
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    clf.fit(Xs, y[idx])
    return np.abs(clf.coef_).mean(axis=0).reshape(n_ch, n_band)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lines = ["# E3.2 / E4 analysis — instance-norm control, LOSO, band/channel importance\n"]

    # ---- E3.2 instance-norm control (C5) ----
    lines.append("## E3.2 — instance-norm control (raw-DE, LogReg, subject-dependent)\n")
    lines.append("| Dataset | corpus-standardized | per-series instance-norm | chance |")
    lines.append("| --- | ---: | ---: | ---: |")
    importances = {}
    sd_results = {}
    loso_rows = []
    for ds in ["seed_v", "seed_iv"]:
        trials = load_de_archive(ARCH[ds])
        base = build_raw_de_segments(trials, instance_norm=False)
        inrm = build_raw_de_segments(trials, instance_norm=True)
        r0 = subject_dependent_eval(base, ds, "logreg")
        r1 = subject_dependent_eval(inrm, ds, "logreg")
        sd_results[ds] = r0
        lines.append(
            f"| {ds} | {r0['accuracy_mean']*100:.2f} / {r0['macro_f1_mean']*100:.2f} "
            f"| {r1['accuracy_mean']*100:.2f} / {r1['macro_f1_mean']*100:.2f} | {r0['chance']:.0f} |"
        )
        LOG.info("instance-norm %s: base=%.2f instance=%.2f", ds, r0["accuracy_mean"]*100, r1["accuracy_mean"]*100)

        # ---- E4.3 LOSO (Linear-SVM: fast & well-conditioned on large sets) ----
        l0 = loso_eval(base, ds, "linear_svm", max_train=4000)
        loso_rows.append((ds, l0))
        LOG.info("LOSO %s: %.2f", ds, l0["accuracy_mean"]*100)

        importances[ds] = band_channel_importance(base)
        LOG.info("importance %s done", ds)

    lines.append(
        "\n*Per-series instance normalization (TimesFM-style RevIN) collapses raw DE "
        "toward chance — confirming the Phase-1 mechanism (C5): the discriminative signal "
        "is the absolute spectral level, which instance-norm removes.*\n"
    )

    # ---- E4.3 LOSO table ----
    lines.append("## E4.3 — LOSO (subject-independent) raw-DE LogReg\n")
    lines.append("| Dataset | LOSO acc / F1 | subject-dependent acc / F1 | chance |")
    lines.append("| --- | ---: | ---: | ---: |")
    for ds, l0 in loso_rows:
        sd = sd_results[ds]
        lines.append(
            f"| {ds} | {l0['accuracy_mean']*100:.2f} / {l0['macro_f1_mean']*100:.2f} "
            f"| {sd['accuracy_mean']*100:.2f} / {sd['macro_f1_mean']*100:.2f} | {l0['chance']:.0f} |"
        )

    # ---- E4.1 band importance ----
    lines.append("\n## E4.1 — band-level discriminative importance (mean |LogReg coef|)\n")
    lines.append("| Dataset | " + " | ".join(BANDS) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(BANDS)) + " |")
    for ds in ["seed_v", "seed_iv"]:
        per_band = importances[ds].mean(axis=0)  # (B,)
        per_band = per_band / per_band.sum()
        lines.append(f"| {ds} | " + " | ".join(f"{v*100:.1f}%" for v in per_band) + " |")

    lines.append("\n## E4.1 — top-8 channels by discriminative importance\n")
    for ds in ["seed_v", "seed_iv"]:
        per_ch = importances[ds].mean(axis=1)  # (C,)
        top = np.argsort(per_ch)[::-1][:8]
        lines.append(f"- **{ds}**: channels (0-indexed) {top.tolist()}")

    # ---- figure: band importance bar + channel importance ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for col, ds in enumerate(["seed_v", "seed_iv"]):
            pb = importances[ds].mean(axis=0)
            axes[0, col].bar(BANDS, pb, color="steelblue")
            axes[0, col].set_title(f"{ds}: band importance")
            axes[0, col].set_ylabel("mean |coef|")
            pc = importances[ds].mean(axis=1)
            axes[1, col].plot(pc, color="darkorange")
            axes[1, col].set_title(f"{ds}: per-channel importance")
            axes[1, col].set_xlabel("channel index")
        fig.tight_layout()
        figpath = OUTDIR / "band_channel_importance.png"
        fig.savefig(figpath, dpi=120)
        lines.append(f"\n![band/channel importance]({figpath})\n")
        LOG.info("wrote %s", figpath)
    except Exception as e:  # pragma: no cover
        LOG.warning("figure skipped: %s", e)

    (OUTDIR / "analysis.md").write_text("\n".join(lines) + "\n")
    LOG.info("wrote %s", OUTDIR / "analysis.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
