#!/usr/bin/env python3
"""F13 label-efficiency curve on sleep staging (the Option-A headline metric).

Reuses the frozen F13 encoders (PC-pretrained vs matched random-init) and the
raw-DE ceiling. For each label fraction, within each subject-disjoint fold we
subsample the *training* epochs (stratified by stage) to that fraction, fit the
classifier, and test on the full held-out fold. Features are extracted ONCE per
encoder (the expensive part); the fraction sweep is cheap on top.

Hypothesis (from EXP-0009 §5): the PC-pretrained FM's advantage over raw-DE and
random-init should GROW as labels shrink — SSL gains concentrate in the low-label
regime. At full labels the FM already wins on sleep (72.6 vs 67.9 raw / 62.9 rand).

    python scripts/phase2_f13_label_curves.py \
        --pc_dir   results/phase3/f13/pc/scratch_pin1_pout16_linear \
        --rand_dir results/phase3/f13/rand/scratch_pin1_pout16_linear \
        --raw --classifier logreg --k 5 \
        --label_fracs 0.01 0.05 0.1 0.25 0.5 1.0 --subsample_seeds 3
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.phase2_eval import CLASSIFIERS
from physiofm.sleep_edf import LABEL_NAMES
from scripts.phase2_f13_sleep import (
    _load_recordings,
    extract_model_features,
    extract_raw_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("f13_label_curves")


def stratified_subsample(y_sub, frac, rng):
    """Indices (into y_sub) of a per-class fraction of the rows; >=1 per class."""
    keep = []
    for c in np.unique(y_sub):
        idx_c = np.where(y_sub == c)[0]
        n = max(1, int(round(frac * len(idx_c))))
        keep.append(rng.choice(idx_c, size=min(n, len(idx_c)), replace=False))
    return np.concatenate(keep)


def curve_eval(X, subj, y, k, classifier, fracs, subsample_seeds):
    """Per-fraction acc/F1/kappa (mean±std over the SAME subject-disjoint folds
    as the headline F13; low-label noise reduced by averaging subsample seeds)."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    subjects = np.array(sorted(set(subj.tolist())))
    rng_fold = np.random.default_rng(42)  # identical split to phase2_f13_sleep
    folds = np.array_split(rng_fold.permutation(subjects), k)
    factory = CLASSIFIERS[classifier]

    out = {}
    for frac in fracs:
        fold_acc, fold_f1, fold_kap = [], [], []
        for fi, test_subj in enumerate(folds):
            te = np.isin(subj, test_subj)
            tr_idx = np.where(~te)[0]
            if len(tr_idx) == 0 or te.sum() == 0:
                continue
            seeds = [0] if frac >= 1.0 else list(range(subsample_seeds))
            a_s, f_s, k_s = [], [], []
            for s in seeds:
                if frac >= 1.0:
                    sel = tr_idx
                else:
                    rng = np.random.default_rng(1000 * s + fi)
                    sel = tr_idx[stratified_subsample(y[tr_idx], frac, rng)]
                clf = factory()
                clf.fit(X[sel], y[sel])
                pred = clf.predict(X[te])
                a_s.append(accuracy_score(y[te], pred))
                f_s.append(f1_score(y[te], pred, average="macro", zero_division=0))
                k_s.append(cohen_kappa_score(y[te], pred))
            fold_acc.append(np.mean(a_s))
            fold_f1.append(np.mean(f_s))
            fold_kap.append(np.mean(k_s))
        a, f, kp = np.array(fold_acc), np.array(fold_f1), np.array(fold_kap)
        out[frac] = {
            "folds": len(fold_acc),
            "n_labels_full": None,  # filled by caller
            "acc_mean": a.mean() * 100, "acc_std": a.std() * 100,
            "f1_mean": f.mean() * 100, "f1_std": f.std() * 100,
            "kappa_mean": kp.mean(), "kappa_std": kp.std(),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_dir", default=None)
    ap.add_argument("--rand_dir", default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--classifier", default="logreg")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--label_fracs", type=float, nargs="+",
                    default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--subsample_seeds", type=int, default=3,
                    help="subsample draws averaged per fold at frac<1 (denoise)")
    ap.add_argument("--out_dir", default="results/phase3/f13")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trials, labels = _load_recordings()
    n_ep = sum(t.values.shape[0] for t in trials)
    LOG.info("sleep corpus: %d recordings, %d epochs, %d subjects",
             len(trials), n_ep, len({t.subject for t in trials}))

    feature_sets = {}
    if args.raw:
        feature_sets["raw_de"] = extract_raw_features(trials, labels)
    if args.pc_dir:
        feature_sets["physiofm_pc"] = extract_model_features(Path(args.pc_dir), trials, labels, device)
    if args.rand_dir:
        feature_sets["physiofm_rand"] = extract_model_features(Path(args.rand_dir), trials, labels, device)
    if not feature_sets:
        raise SystemExit("nothing to evaluate: pass --raw and/or --pc_dir/--rand_dir")

    fracs = sorted(args.label_fracs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (X, subj, y) in feature_sets.items():
        res = curve_eval(X, subj, y, args.k, args.classifier, fracs, args.subsample_seeds)
        for frac in fracs:
            r = res[frac]
            LOG.info("RESULT %-14s frac=%-5.2f acc=%.2f±%.2f f1=%.2f±%.2f kappa=%.3f±%.3f (%d folds)",
                     name, frac, r["acc_mean"], r["acc_std"], r["f1_mean"], r["f1_std"],
                     r["kappa_mean"], r["kappa_std"], r["folds"])
            rows.append((name, frac, r))

    csv_path = out_dir / "f13_label_curves.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["features", "label_frac", "folds", "acc_mean", "acc_std",
                    "f1_mean", "f1_std", "kappa_mean", "kappa_std"])
        for name, frac, r in rows:
            w.writerow([name, f"{frac:.2f}", r["folds"], f"{r['acc_mean']:.2f}", f"{r['acc_std']:.2f}",
                        f"{r['f1_mean']:.2f}", f"{r['f1_std']:.2f}",
                        f"{r['kappa_mean']:.3f}", f"{r['kappa_std']:.3f}"])
    LOG.info("wrote %s  (chance=%.1f%%, classifier=%s, subsample_seeds=%d)",
             csv_path, 100.0 / len(LABEL_NAMES), args.classifier, args.subsample_seeds)


if __name__ == "__main__":
    main()
