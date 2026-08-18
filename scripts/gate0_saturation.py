#!/usr/bin/env python3
"""Gate 0 (docs/NEXT_PHASE_PLAN.md): does LESS COMPRESSION restore representational headroom?

Requirement R1: the discriminative information must NOT be linearly accessible from the
input, otherwise a linear probe on the raw features already reads it optimally and no
encoder (pretrained or not) has anything to add — F10 showed exactly this for DE on emotion.
Here we run the same linear-saturation test on the per-epoch-label tasks (sleep, seizure)
for two feature sets built from the SAME recordings/epochs/labels:

    de    5 DE bands per channel      (10-d sleep / 90-d seizure)
    tf64  64 log-spaced log-power bins (128-d sleep / 1152-d seizure)

and three heads on each: logreg (linear), mlp_bal (balanced 2-layer MLP), hgb (gradient-
boosted trees). Protocols mirror the model evaluators exactly (sleep: subject-disjoint
5-fold, seed 42; seizure: leave-one-patient-out with a stratified training cap).

    headroom(feature) = best nonlinear − linear
Decision rule (pre-registered in EXP-0020): headroom(tf64) > headroom(de) by >= 2 points
AND best-nonlinear(tf64) > linear(tf64) by >= 2 points => R1 satisfied for tf64 -> proceed
to pretraining on tf64. Otherwise tf64 is saturated like DE and the pilot fails.
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

from physiofm.de import load_de_archive
from physiofm.phase2_eval import CLASSIFIERS, SEED
from physiofm.structured_data import ARCH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("gate0")


def make_hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, early_stopping=True, class_weight="balanced",
        random_state=SEED))


HEADS = {"logreg": CLASSIFIERS["logreg"], "mlp_bal": CLASSIFIERS["mlp_bal"], "hgb": make_hgb}
LINEAR = {"logreg"}


def load_task(task, arch_key):
    trials = load_de_archive(ARCH[arch_key])
    if task == "sleep":
        from physiofm.sleep_edf import load_sleep_labels
        labels, subj, night, key = load_sleep_labels("data/physiofm/de_features/sleep_edf_labels.npz")
    else:
        from physiofm.chbmit import load_chbmit_labels
        labels, subj, fidx, key = load_chbmit_labels("data/physiofm/de_features/chbmit_labels.npz")
    assert len(trials) == len(labels)
    X, S, Y = [], [], []
    for t, lab in zip(trials, labels):
        v = np.asarray(t.values, np.float32).reshape(t.values.shape[0], -1)
        n = min(v.shape[0], lab.shape[0])
        X.append(v[:n]); S.append(np.full(n, t.subject, np.int64)); Y.append(lab[:n])
    X, S, Y = np.concatenate(X), np.concatenate(S), np.concatenate(Y)
    k = Y >= 0
    return X[k], S[k], Y[k]


def _fit_sleep_fold(X, y, subj, test_subj, head, max_train, seed):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    te = np.isin(subj, test_subj); tr = np.where(~te)[0]
    if max_train and tr.size > max_train:
        tr = np.random.default_rng(seed).choice(tr, size=max_train, replace=False)
    clf = HEADS[head]()
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return (accuracy_score(y[te], pred) * 100, f1_score(y[te], pred, average="macro") * 100,
            cohen_kappa_score(y[te], pred))


def _fit_seizure_fold(X, y, subj, patient, head, max_train, seed):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    te = subj == patient
    if te.sum() == 0 or len(np.unique(y[te])) < 2:
        return None
    tr = np.where(~te)[0]
    if max_train and tr.size > max_train:
        rng = np.random.default_rng(seed + int(patient))
        pos = tr[y[tr] == 1]; neg = tr[y[tr] == 0]
        n_pos = min(pos.size, max_train // 2)
        n_neg = min(neg.size, max_train - n_pos)
        tr = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
    clf = HEADS[head]()
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    if hasattr(clf, "predict_proba"):
        score = clf.predict_proba(X[te])[:, 1]
    elif hasattr(clf, "decision_function"):
        score = clf.decision_function(X[te])
    else:  # _BalancedMLP wrapper exposes only predict(); use its inner MLP's probabilities
        inner = clf[-1].clf_
        score = inner.predict_proba(clf[:-1].transform(X[te]))[:, 1]
    return (balanced_accuracy_score(y[te], pred) * 100, roc_auc_score(y[te], score), 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sleep", "seizure"], required=True)
    ap.add_argument("--arch_keys", nargs="+", default=None)
    ap.add_argument("--heads", nargs="+", default=["logreg", "mlp_bal", "hgb"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_train", type=int, default=0, help="cap on training epochs per fold (0=all)")
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--out_dir", default="results/phase4/gate0")
    args = ap.parse_args()
    from joblib import Parallel, delayed

    keys = args.arch_keys or (["sleep_edf", "sleep_edf_tf64"] if args.task == "sleep" else ["chbmit", "chbmit_tf64"])
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in keys:
        X, S, Y = load_task(args.task, key)
        LOG.info("%s/%s: %d epochs x %d dims, %d subjects", args.task, key, X.shape[0], X.shape[1], len(set(S.tolist())))
        groups = np.array(sorted(set(S.tolist())))
        if args.task == "sleep":
            folds = np.array_split(np.random.default_rng(42).permutation(groups), args.k)  # == f13 folds
        else:
            folds = [np.array([g]) for g in groups]                                        # LOPO
        for head in args.heads:
            fn = _fit_sleep_fold if args.task == "sleep" else _fit_seizure_fold
            res = Parallel(n_jobs=args.n_jobs, max_nbytes="1M")(
                delayed(fn)(X, Y, S, f if args.task == "sleep" else f[0], head, args.max_train, SEED) for f in folds)
            res = [r for r in res if r is not None]
            a = np.array(res)
            kind = "linear" if head in LINEAR else "nonlinear"
            LOG.info("RESULT %-16s %-8s %-9s m1=%.2f±%.2f m2=%.3f (%d folds)",
                     key, head, kind, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), len(res))
            rows.append((key, head, kind, len(res), a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 2].mean(),
                         a[:, 0].tolist()))
    m1, m2 = ("acc", "macro_f1") if args.task == "sleep" else ("bal_acc", "auc")
    with (out_dir / f"saturation_{args.task}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch_key", "head", "kind", "folds", f"{m1}_mean", f"{m1}_std", f"{m2}_mean", "kappa_mean", "per_fold_m1"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.2f}", f"{r[5]:.2f}", f"{r[6]:.3f}", f"{r[7]:.3f}",
                        ";".join(f"{v:.2f}" for v in r[8])])
    # headroom summary
    with (out_dir / f"saturation_{args.task}.md").open("w") as fh:
        fh.write(f"# Gate 0 linear-saturation test — {args.task}\n\n| features | head | kind | {m1} | {m2} |\n|---|---|---|---:|---:|\n")
        for r in rows:
            fh.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[4]:.2f} ± {r[5]:.2f} | {r[6]:.3f} |\n")
        fh.write("\n| features | linear | best nonlinear | headroom |\n|---|---:|---:|---:|\n")
        for key in keys:
            lin = [r[4] for r in rows if r[0] == key and r[2] == "linear"]
            non = [r[4] for r in rows if r[0] == key and r[2] == "nonlinear"]
            if lin and non:
                fh.write(f"| {key} | {lin[0]:.2f} | {max(non):.2f} | {max(non) - lin[0]:+.2f} |\n")
    LOG.info("wrote %s", out_dir / f"saturation_{args.task}.csv")


if __name__ == "__main__":
    main()
