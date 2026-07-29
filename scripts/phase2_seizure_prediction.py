#!/usr/bin/env python3
"""Seizure PREDICTION, patient-specific protocol (the standard for this task).

A first attempt used leave-one-patient-out and every arm — including the raw-DE baseline —
sat at chance (AUC 0.49-0.55). That is expected: pre-ictal signatures are highly individual,
so cross-patient prediction is near-impossible and essentially all published seizure-
prediction work is PATIENT-SPECIFIC. A task nothing can solve cannot discriminate between
hypotheses, so this script uses the correct protocol.

Protocol: within each patient, split by SEIZURE EVENT (leave-one-seizure-out over the
pre-ictal blocks, with interictal split alongside), so train and test never share a seizure.
Pretraining stays unsupervised over the whole corpus, unchanged.

The scientific question is unchanged (EXP-0017): seizure prediction is the one EEG task whose
downstream objective IS forecasting, so our diagnosis predicts PC pretraining should help
here even fine-tuned, unlike detection.

    python scripts/phase2_seizure_prediction.py --pc_dir ... --rand_dir ... --raw
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.chbmit import load_chbmit_labels
from physiofm.de import load_de_archive
from physiofm.phase2_eval import CLASSIFIERS
from physiofm.structured_data import ARCH
from scripts.phase2_f13_sleep import extract_model_features, extract_raw_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("seiz_pred")

PRED_LABELS = "data/physiofm/de_features/chbmit_pred_labels.npz"


def _load():
    trials = load_de_archive(ARCH["chbmit"])
    labels, patient, file_idx, key = load_chbmit_labels(PRED_LABELS)
    return trials, labels


def block_ids(labels):
    """Tag each contiguous pre-ictal run with a distinct block id (one per seizure event),
    so we can hold out a whole event rather than random epochs from it."""
    out = []
    nxt = 0
    for lab in labels:
        b = np.full(len(lab), -1, dtype=np.int64)
        inside = False
        for i, v in enumerate(lab):
            if v == 1:
                if not inside:
                    nxt += 1; inside = True
                b[i] = nxt
            else:
                inside = False
        out.append(b)
    return out, nxt


def eval_patient(X, y, blk, interictal_grp, classifier, min_blocks=3):
    """Leave-one-seizure-event-out within a patient. Returns per-fold AUC/bal-acc."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    blocks = sorted(set(blk[blk > 0].tolist()))
    if len(blocks) < min_blocks:
        return None
    inter_idx = np.where(y == 0)[0]
    if inter_idx.size < 50:
        return None
    rng = np.random.default_rng(42)
    inter_split = rng.permutation(inter_idx)
    parts = np.array_split(inter_split, len(blocks))  # interictal split alongside events

    res = []
    for i, b in enumerate(blocks):
        te_pre = np.where(blk == b)[0]
        te_int = parts[i]
        tr_pre = np.where((blk > 0) & (blk != b))[0]
        tr_int = np.concatenate([parts[j] for j in range(len(blocks)) if j != i])
        tr = np.concatenate([tr_pre, tr_int]); te = np.concatenate([te_pre, te_int])
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = CLASSIFIERS[classifier]()
        clf.fit(X[tr], y[tr])
        try:
            score = clf.predict_proba(X[te])[:, 1]
        except Exception:
            score = clf.decision_function(X[te])
        res.append((balanced_accuracy_score(y[te], clf.predict(X[te])),
                    roc_auc_score(y[te], score)))
    return np.array(res) if res else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_dir", default=None)
    ap.add_argument("--rand_dir", default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--classifier", default="logreg")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_csv", default="results/phase3/f18/f18_seizure_prediction.csv")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trials, labels = _load()
    blks, n_ev = block_ids(labels)
    LOG.info("seizure prediction: %d recordings, %d pre-ictal events", len(trials), n_ev)

    feats = {}
    if args.raw:
        feats["raw_de"] = extract_raw_features(trials, labels)
    if args.pc_dir:
        feats["physiofm_pc"] = extract_model_features(Path(args.pc_dir), trials, labels, device, args.batch_size)
    if args.rand_dir:
        feats["physiofm_rand"] = extract_model_features(Path(args.rand_dir), trials, labels, device, args.batch_size)

    # per-epoch block ids aligned with the flattened feature arrays
    blk_flat = np.concatenate([b[:min(len(b), len(l))] for b, l in zip(blks, labels)])

    rows = []
    for name, (X, subj, y) in feats.items():
        keep = y >= 0
        Xk, sk, yk, bk = X[keep], subj[keep], y[keep], blk_flat[: len(y)][keep]
        per_pat = []
        for p in sorted(set(sk.tolist())):
            m = sk == p
            r = eval_patient(Xk[m], yk[m], bk[m], None, args.classifier)
            if r is not None:
                per_pat.append(r.mean(axis=0))
        a = np.array(per_pat)
        LOG.info("RESULT %-14s bal_acc=%.2f±%.2f auc=%.3f±%.3f (%d patients)",
                 name, a[:, 0].mean() * 100, a[:, 0].std() * 100, a[:, 1].mean(), a[:, 1].std(), len(a))
        rows.append((name, len(a), a[:, 0].mean() * 100, a[:, 0].std() * 100,
                     a[:, 1].mean(), a[:, 1].std()))

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["features", "patients", "bal_acc", "bal_acc_std", "auc", "auc_std"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.3f}", f"{r[5]:.3f}"])
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
