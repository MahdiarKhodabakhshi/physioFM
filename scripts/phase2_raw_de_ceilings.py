#!/usr/bin/env python3
"""E0.1 — Raw-DE nonlinear ceilings (no foundation model).

Segment-level classification of the raw 310-d DE window with the PC-SSL
subject-dependent splits, across LogReg / Linear-SVM / RBF-SVM / MLP. This is
the honest non-FM ceiling the Phase 2 transformer must beat. If RBF-SVM already
approaches PC-SSL, the FM's value proposition is reframed.

Writes results/phase2/raw_de_ceilings.csv and .md. Does not touch Phase 1 files.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import load_de_archive
from physiofm.phase2_eval import build_raw_de_segments, subject_dependent_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("e0_ceilings")

ARCH = {
    "seed_v": "data/physiofm/de_features/seed_v_de_LDS.npz",
    "seed_iv": "data/physiofm/de_features/seed_iv_de_LDS.npz",
}
OUTDIR = Path("results/phase2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["seed_v", "seed_iv"])
    ap.add_argument(
        "--classifiers",
        nargs="+",
        default=["logreg", "linear_svm", "rbf_svm", "mlp"],
    )
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = []
    for ds in args.datasets:
        trials = load_de_archive(ARCH[ds])
        feats = build_raw_de_segments(trials)
        LOG.info(
            "%s: %d segments x %d dims, %d labeled trials",
            ds,
            feats.X.shape[0],
            feats.X.shape[1],
            len(feats.trials),
        )
        for clf in args.classifiers:
            res = subject_dependent_eval(feats, ds, classifier=clf)
            LOG.info(
                "RESULT %s %s acc=%.2f±%.2f f1=%.2f±%.2f (runs=%d, chance=%.1f)",
                ds,
                clf,
                res["accuracy_mean"] * 100,
                res["accuracy_std"] * 100,
                res["macro_f1_mean"] * 100,
                res["macro_f1_std"] * 100,
                res["runs"],
                res["chance"],
            )
            results.append(res)

    csv_path = OUTDIR / "raw_de_ceilings.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "classifier", "runs", "acc_mean", "acc_std", "f1_mean", "f1_std", "chance"])
        for r in results:
            w.writerow(
                [
                    r["dataset"],
                    r["classifier"],
                    r["runs"],
                    f"{r['accuracy_mean'] * 100:.2f}",
                    f"{r['accuracy_std'] * 100:.2f}",
                    f"{r['macro_f1_mean'] * 100:.2f}",
                    f"{r['macro_f1_std'] * 100:.2f}",
                    f"{r['chance']:.1f}",
                ]
            )
    LOG.info("wrote %s", csv_path)

    md_path = OUTDIR / "raw_de_ceilings.md"
    with md_path.open("w") as f:
        f.write("# E0.1 — Raw-DE segment-level ceilings (subject-dependent, PC-SSL splits)\n\n")
        f.write("Seed 42. Accuracy % / macro-F1 % (mean ± std across subject×fold runs).\n\n")
        for ds in args.datasets:
            chance = next(r["chance"] for r in results if r["dataset"] == ds)
            f.write(f"## {ds} (chance {chance:.1f}%)\n\n")
            f.write("| Classifier | Runs | Accuracy % | Macro-F1 % |\n")
            f.write("| --- | ---: | ---: | ---: |\n")
            for r in results:
                if r["dataset"] != ds:
                    continue
                f.write(
                    f"| raw-DE → {r['classifier']} | {r['runs']} | "
                    f"{r['accuracy_mean'] * 100:.2f} ± {r['accuracy_std'] * 100:.2f} | "
                    f"{r['macro_f1_mean'] * 100:.2f} ± {r['macro_f1_std'] * 100:.2f} |\n"
                )
            f.write("\n")
    LOG.info("wrote %s", md_path)


if __name__ == "__main__":
    main()
