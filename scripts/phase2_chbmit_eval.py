#!/usr/bin/env python3
"""F17 — CHB-MIT seizure-detection eval (the truer 2nd dynamic task).

Per-epoch binary (seizure/interictal) classification, **leave-one-patient-out**
(patient-disjoint). Seizures are ~1% of epochs, so we report imbalance-aware metrics
— balanced accuracy, sensitivity (seizure recall), specificity, ROC-AUC — not raw
accuracy. Same PC-vs-random-vs-raw ladder as sleep (EXP-0009) and MI (EXP-0014).

    python scripts/phase2_chbmit_eval.py \
        --pc_dir results/phase3/f17/pc/scratch_pin1_pout16_linear \
        --rand_dir results/phase3/f17/rand/scratch_pin1_pout16_linear \
        --raw --out_dir results/phase3/f17
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
LOG = logging.getLogger("f17_chbmit")

LABELS_ARCH = "data/physiofm/de_features/chbmit_labels.npz"
LABELS_OVERRIDE = None  # set by --labels


def _load():
    trials = load_de_archive(ARCH["chbmit"])
    labels, patient, file_idx, key = load_chbmit_labels(LABELS_OVERRIDE or LABELS_ARCH)
    if len(trials) != len(labels):
        raise SystemExit(f"archive/labels misalignment: {len(trials)} vs {len(labels)}")
    return trials, labels


def _fit_patient(X, subj, y, p, classifier, label_frac=1.0, sseed=0):
    """One held-out patient. At label_frac<1 the *training* epochs are stratified-
    subsampled (keep the fraction of seizure and of interictal epochs separately, so
    rare seizures aren't lost) — the label-efficiency knob."""
    from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

    te = subj == p
    if te.sum() == 0 or len(np.unique(y[te])) < 2:
        return None  # need both classes in the held-out patient for sens/spec/AUC
    tr_idx = np.where(subj != p)[0]
    if tr_idx.size == 0:
        return None
    if label_frac < 1.0:
        rng = np.random.default_rng(1000 * sseed + int(p))
        parts = []
        for c in (0, 1):
            idx_c = tr_idx[y[tr_idx] == c]
            if idx_c.size == 0:
                continue
            k = max(1, int(round(label_frac * idx_c.size)))
            parts.append(rng.choice(idx_c, size=min(k, idx_c.size), replace=False))
        tr_idx = np.concatenate(parts)
    if len(np.unique(y[tr_idx])) < 2:
        return None
    clf = CLASSIFIERS[classifier]()
    clf.fit(X[tr_idx], y[tr_idx])
    pred = clf.predict(X[te])
    try:
        score = clf.predict_proba(X[te])[:, 1]
    except Exception:
        score = clf.decision_function(X[te])
    return (
        balanced_accuracy_score(y[te], pred),
        recall_score(y[te], pred, pos_label=1, zero_division=0),   # sensitivity
        recall_score(y[te], pred, pos_label=0, zero_division=0),   # specificity
        roc_auc_score(y[te], score),
    )


def lopo_eval(X, subj, y, classifier, n_jobs=-1, fracs=(1.0,), subsample_seeds=1):
    """Per-fraction LOPO metrics (mean over patients, seed-averaged at frac<1)."""
    from collections import defaultdict

    from joblib import Parallel, delayed

    patients = sorted(set(subj.tolist()))
    tasks = []
    for frac in fracs:
        seeds = [0] if frac >= 1.0 else list(range(subsample_seeds))
        for p in patients:
            for s in seeds:
                tasks.append((frac, p, s))
    raw = Parallel(n_jobs=n_jobs, max_nbytes="1M")(
        delayed(_fit_patient)(X, subj, y, p, classifier, frac, s) for (frac, p, s) in tasks
    )
    bucket = defaultdict(lambda: defaultdict(list))  # frac -> patient -> [(4,) over seeds]
    for (frac, p, s), r in zip(tasks, raw):
        if r is not None:
            bucket[frac][p].append(r)
    out = {}
    for frac in fracs:
        pats = sorted(bucket[frac].keys())
        per_patient = {p: np.mean(bucket[frac][p], axis=0) for p in pats}  # seed-avg per patient
        a = np.array([per_patient[p] for p in pats]) if pats else np.zeros((0, 4))
        out[frac] = {"patients": len(pats),
                     "bal_acc": a[:, 0].mean() * 100 if len(a) else 0.0,
                     "bal_acc_std": a[:, 0].std() * 100 if len(a) else 0.0,
                     "sens": a[:, 1].mean() * 100 if len(a) else 0.0,
                     "spec": a[:, 2].mean() * 100 if len(a) else 0.0,
                     "auc": a[:, 3].mean() if len(a) else 0.0,
                     "auc_std": a[:, 3].std() if len(a) else 0.0,
                     "per_patient": per_patient}  # for paired significance tests
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_dir", default=None)
    ap.add_argument("--rand_dir", default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--classifiers", nargs="+", default=["logreg"])
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=8)  # long seizure recordings
    ap.add_argument("--label_fracs", type=float, nargs="+", default=[1.0],
                    help="label-efficiency sweep (stratified train subsample)")
    ap.add_argument("--subsample_seeds", type=int, default=3)
    ap.add_argument("--out_dir", default="results/phase3/f17")
    ap.add_argument("--tag", default="")
    ap.add_argument("--labels", default=None, help="alternative per-epoch label archive")
    args = ap.parse_args()
    if args.labels:
        global LABELS_OVERRIDE; LABELS_OVERRIDE = args.labels

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trials, labels = _load()
    n_ep = sum(t.values.shape[0] for t in trials)
    n_seiz = sum(int((l == 1).sum()) for l in labels)
    LOG.info("CHB-MIT: %d recordings, %d epochs, %d seizure (%.2f%%), %d patients, n_cb=%d",
             len(trials), n_ep, n_seiz, 100 * n_seiz / max(n_ep, 1),
             len({t.subject for t in trials}), trials[0].values.shape[1] * trials[0].values.shape[2])

    keep_mask = None  # set after features are built (label -1 = excluded)
    feats = {}
    if args.raw:
        feats["raw_de"] = extract_raw_features(trials, labels)
    if args.pc_dir:
        feats["physiofm_pc"] = extract_model_features(Path(args.pc_dir), trials, labels, device, args.batch_size)
    if args.rand_dir:
        feats["physiofm_rand"] = extract_model_features(Path(args.rand_dir), trials, labels, device, args.batch_size)
    if not feats:
        raise SystemExit("nothing to evaluate")

    # drop epochs the protocol excludes (SPH / ictal / post-ictal, label -1)
    for name in list(feats):
        X, subj, y = feats[name]
        k = y >= 0
        feats[name] = (X[k], subj[k], y[k])
    LOG.info("after excluding label<0: %d epochs (%.1f%% positive)",
             int((next(iter(feats.values()))[2] >= 0).sum()),
             100 * float(next(iter(feats.values()))[2].mean()))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fracs = sorted(args.label_fracs)
    rows = []
    for name, (X, subj, y) in feats.items():
        for clf in args.classifiers:
            res = lopo_eval(X, subj, y, clf, args.n_jobs, fracs, args.subsample_seeds)
            for frac in fracs:
                r = res[frac]
                LOG.info("RESULT %-14s %-8s frac=%.2f bal_acc=%.2f±%.2f sens=%.2f spec=%.2f auc=%.3f±%.3f (%d pat)",
                         name, clf, frac, r["bal_acc"], r["bal_acc_std"], r["sens"], r["spec"],
                         r["auc"], r["auc_std"], r["patients"])
                rows.append((name, clf, frac, r))

    with (out_dir / f"f17_chbmit{args.tag}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["features", "classifier", "label_frac", "patients", "bal_acc", "bal_acc_std",
                    "sensitivity", "specificity", "auc", "auc_std"])
        for name, clf, frac, r in rows:
            w.writerow([name, clf, f"{frac:.2f}", r["patients"], f"{r['bal_acc']:.2f}",
                        f"{r['bal_acc_std']:.2f}", f"{r['sens']:.2f}", f"{r['spec']:.2f}",
                        f"{r['auc']:.3f}", f"{r['auc_std']:.3f}"])

    # per-patient values (for paired significance tests: pc vs raw, pc vs rand)
    pp_path = out_dir / f"f17_chbmit{args.tag}_perpatient.csv"
    with pp_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["features", "classifier", "label_frac", "patient",
                    "bal_acc", "sensitivity", "specificity", "auc"])
        for name, clf, frac, r in rows:
            for p, v in sorted(r["per_patient"].items()):
                w.writerow([name, clf, f"{frac:.2f}", int(p), f"{v[0]*100:.4f}",
                            f"{v[1]*100:.4f}", f"{v[2]*100:.4f}", f"{v[3]:.4f}"])
    LOG.info("wrote %s and %s", out_dir / f"f17_chbmit{args.tag}.csv", pp_path)


if __name__ == "__main__":
    main()
