#!/usr/bin/env python3
"""F10 — Representation ceiling / linear-saturation test.

Root-cause diagnostic: is there ANY headroom above the raw-DE linear ceiling? We
classify the per-window 310-d DE with linear (LogReg, Linear-SVM) vs nonlinear
(RBF-SVM, balanced MLP) heads, under two data regimes:

  * subject_dependent — PC-SSL folds, ~600 labels/fold (small data)
  * loso              — leave-one-subject-out, large train (subsampled to 8000)

If nonlinear ~= linear in BOTH regimes, the static per-window DE->emotion map is
linearly saturated => no representational headroom existed for the proposed FM to
capture (root cause B). If nonlinear >> linear under large data, headroom exists
and the failure was the method, not the task.

No model; reuses the canonical eval harness so the numbers are ladder-comparable.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import load_de_archive
from physiofm.phase2_eval import (
    build_raw_de_segments,
    loso_eval,
    subject_dependent_eval,
)
from physiofm.structured_data import ARCH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("f10_ceiling")

OUTDIR = Path("results/phase2/followup/f10")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["seed_v", "seed_iv", "seed_iv_raw"])
    ap.add_argument("--classifiers", nargs="+",
                    default=["logreg", "linear_svm", "rbf_svm", "mlp_bal"])
    ap.add_argument("--protocols", nargs="+",
                    default=["subject_dependent", "loso"])
    ap.add_argument("--max_train", type=int, default=8000, help="LOSO train cap")
    ap.add_argument("--out_dir", default=str(OUTDIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in args.datasets:
        feats = build_raw_de_segments(load_de_archive(ARCH[ds]))
        LOG.info("%s: %d segments x %d dims", ds, feats.X.shape[0], feats.X.shape[1])
        for proto in args.protocols:
            for clf in args.classifiers:
                if proto == "loso":
                    res = loso_eval(feats, ds, classifier=clf, max_train=args.max_train)
                else:
                    res = subject_dependent_eval(feats, ds, classifier=clf)
                LOG.info("RESULT %-12s %-18s %-10s acc=%.2f±%.2f f1=%.2f±%.2f (chance=%.1f, runs=%d)",
                         ds, proto, clf, res["accuracy_mean"] * 100, res["accuracy_std"] * 100,
                         res["macro_f1_mean"] * 100, res["macro_f1_std"] * 100,
                         res["chance"], res["runs"])
                rows.append((ds, proto, clf, res))

    csv_path = out_dir / "f10_ceiling.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "protocol", "classifier", "kind", "runs",
                    "acc_mean", "acc_std", "f1_mean", "f1_std", "chance"])
        for ds, proto, clf, res in rows:
            kind = "linear" if clf in ("logreg", "linear_svm") else "nonlinear"
            w.writerow([ds, proto, clf, kind, res["runs"],
                        f"{res['accuracy_mean'] * 100:.2f}", f"{res['accuracy_std'] * 100:.2f}",
                        f"{res['macro_f1_mean'] * 100:.2f}", f"{res['macro_f1_std'] * 100:.2f}",
                        f"{res['chance']:.1f}"])

    md_path = out_dir / "f10_ceiling.md"
    with md_path.open("w") as f:
        f.write("# F10 — Representation ceiling / linear-saturation (acc % / macro-F1 %)\n\n")
        for ds in args.datasets:
            f.write(f"## {ds}\n\n| protocol | classifier | kind | acc % | macro-F1 % |\n")
            f.write("| --- | --- | --- | ---: | ---: |\n")
            for d, proto, clf, res in rows:
                if d != ds:
                    continue
                kind = "linear" if clf in ("logreg", "linear_svm") else "nonlinear"
                f.write(f"| {proto} | {clf} | {kind} | "
                        f"{res['accuracy_mean'] * 100:.2f} ± {res['accuracy_std'] * 100:.2f} | "
                        f"{res['macro_f1_mean'] * 100:.2f} ± {res['macro_f1_std'] * 100:.2f} |\n")
            f.write("\n")
    LOG.info("wrote %s and %s", csv_path, md_path)


if __name__ == "__main__":
    main()
