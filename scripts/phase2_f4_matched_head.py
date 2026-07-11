#!/usr/bin/env python3
"""F4 — Matched downstream head (fair PC-SSL-style comparison).

Applies ONE identical downstream protocol — frozen encoder + the same
2-hidden-layer balanced MLP head (class weighting via balanced oversampling,
per-fold validation early stopping), same clean PC-SSL subject-dependent splits
— to three feature sets:

  * raw DE (310-d, the non-FM ceiling)
  * PhysioFM-S, PC-pretrained (frozen encoder embeddings)
  * PhysioFM-S, random-init    (frozen encoder embeddings)

so the comparison isolates the *representation*, not the classifier. ``logreg``
(linear probe) and ``mlp`` (the un-balanced sklearn MLP) are reported alongside
``mlp_bal`` for context.

Decision rule (F4):
  * raw-DE + MLP reaches the 80s        -> the PC-SSL gap is mostly the head.
  * PhysioFM-S + MLP > raw-DE + MLP     -> representation value the linear probe hid.
  * all three tie near the linear ceil  -> the head is not the lever; static ceiling.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import load_de_archive
from physiofm.phase2_eval import build_raw_de_segments, subject_dependent_eval
from physiofm.structured_data import ARCH, load_standardizer
from scripts.phase2_extract_eval import extract, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("f4_matched_head")

OUTDIR = Path("results/phase2/followup/f4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_dir", required=True)
    ap.add_argument("--random_dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=["seed_v", "seed_iv"])
    ap.add_argument("--classifiers", nargs="+", default=["logreg", "mlp", "mlp_bal"])
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre_model, pre_args = load_model(Path(args.pretrained_dir), device)
    pre_mean, pre_std = load_standardizer(Path(args.pretrained_dir) / "standardizer.npz")
    rnd_model, rnd_args = load_model(Path(args.random_dir), device)
    rnd_mean, rnd_std = load_standardizer(Path(args.random_dir) / "standardizer.npz")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        trials = load_de_archive(ARCH[ds])
        providers = {
            "raw_de": build_raw_de_segments(trials),
            "physiofm_pretrained": extract(pre_model, pre_args, pre_mean, pre_std, trials, device),
            "physiofm_random_init": extract(rnd_model, rnd_args, rnd_mean, rnd_std, trials, device),
        }
        for feat_name, feats in providers.items():
            for clf in args.classifiers:
                res = subject_dependent_eval(feats, ds, classifier=clf)
                LOG.info("RESULT %s %s %s acc=%.2f±%.2f f1=%.2f±%.2f (chance=%.1f)",
                         ds, feat_name, clf, res["accuracy_mean"] * 100, res["accuracy_std"] * 100,
                         res["macro_f1_mean"] * 100, res["macro_f1_std"] * 100, res["chance"])
                rows.append((ds, feat_name, clf, res))

    with (OUTDIR / "f4_matched_head.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "features", "classifier", "acc_mean", "acc_std", "f1_mean", "f1_std", "chance"])
        for ds, feat, clf, r in rows:
            w.writerow([ds, feat, clf, f"{r['accuracy_mean']*100:.2f}", f"{r['accuracy_std']*100:.2f}",
                        f"{r['macro_f1_mean']*100:.2f}", f"{r['macro_f1_std']*100:.2f}", f"{r['chance']:.1f}"])

    lines = ["# F4 — Matched downstream head (frozen encoder + identical 2-layer MLP)\n",
             "Segment-level, subject-dependent PC-SSL splits, seed 42. acc % / macro-F1 %.\n"]
    for ds in args.datasets:
        chance = next(r["chance"] for d, _, _, r in rows if d == ds)
        lines.append(f"## {ds} (chance {chance:.0f}%)\n")
        clfs = args.classifiers
        lines.append("| Features | " + " | ".join(clfs) + " |")
        lines.append("| --- | " + " | ".join(["---:"] * len(clfs)) + " |")
        for feat in ["raw_de", "physiofm_pretrained", "physiofm_random_init"]:
            cells = []
            for clf in clfs:
                r = next(rr for d, fe, c, rr in rows if d == ds and fe == feat and c == clf)
                cells.append(f"{r['accuracy_mean']*100:.2f} / {r['macro_f1_mean']*100:.2f}")
            lines.append(f"| {feat} | " + " | ".join(cells) + " |")
        lines.append("")
    (OUTDIR / "f4_matched_head.md").write_text("\n".join(lines) + "\n")
    LOG.info("wrote %s", OUTDIR / "f4_matched_head.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
