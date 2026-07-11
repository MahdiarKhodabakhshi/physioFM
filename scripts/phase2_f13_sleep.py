#!/usr/bin/env python3
"""F13 — Pre-registered temporal-PC test on sleep staging (Sleep-EDF).

The pre-registered prediction (docs/experiments/EXP-0009): on a genuinely dynamic
task, the PC-pretrained PhysioFM-S encoder beats the matched random-init encoder —
unlike static emotion DE, where they tie. We freeze each encoder, extract one
hidden state per 30 s epoch, and classify sleep stage (W/N1/N2/N3/REM) under a
**subject-disjoint** k-fold (the sleep-staging convention), reporting accuracy,
macro-F1, and Cohen's kappa.

    python scripts/phase2_f13_sleep.py \
        --pc_dir   results/phase3/f13/pretrain/scratch_pin1_pout16_linear \
        --rand_dir results/phase3/f13/pretrain_rand/scratch_pin1_pout16_linear \
        --raw --classifiers logreg mlp_bal --k 5
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

# Ops: pin BLAS to 1 thread/process so the joblib fold fan-out doesn't oversubscribe.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import load_de_archive
from physiofm.phase2_eval import CLASSIFIERS
from physiofm.sleep_edf import LABEL_NAMES, load_sleep_labels
from physiofm.structured_data import ARCH, collate_pad, load_standardizer, standardize

# reuse the exact model-load pattern from the emotion extractor
from scripts.phase2_extract_eval import load_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("f13_sleep")

LABELS_ARCH = "data/physiofm/de_features/sleep_edf_labels.npz"


def _load_recordings():
    """Return aligned (trials, per_epoch_labels) for the sleep corpus."""
    trials = load_de_archive(ARCH["sleep_edf"])
    labels, subj, night, key = load_sleep_labels(LABELS_ARCH)
    if len(trials) != len(labels):
        raise SystemExit(f"archive/labels misalignment: {len(trials)} vs {len(labels)}")
    return trials, labels


def extract_model_features(model_dir: Path, trials, labels, device, batch_size: int = 16):
    """Per-epoch frozen-encoder features. Returns X (N,d), subject (N,), y (N,).

    Batched with length-bucketing for GPU throughput: recordings are sorted by
    length and padded within a batch. Causal attention + the right-pad mask make
    the valid-position features identical to one-at-a-time encoding (verified in
    `--self_check`), so batching is a pure speedup, not an approximation.
    """
    import torch

    model, margs = load_model(model_dir, device)
    model.eval()
    mean, std = load_standardizer(model_dir / "standardizer.npz")
    p_in = margs["p_in"]

    seqs = [standardize(t.values, mean, std) for t in trials]  # (T, n_cb) each
    order = sorted(range(len(seqs)), key=lambda i: seqs[i].shape[0])  # length buckets
    feats: list = [None] * len(seqs)
    with torch.no_grad():
        for b0 in range(0, len(order), batch_size):
            idxs = order[b0 : b0 + batch_size]
            batch = [seqs[i] for i in idxs]
            x, mask = collate_pad(batch)  # (B, Tmax, n_cb), (B, Tmax)
            h = model.encode(x.to(device), mask.to(device)).float().cpu().numpy()  # (B, Pmax, d)
            for j, i in enumerate(idxs):
                t = batch[j].shape[0]
                p = t // p_in
                hi = h[j, :p]
                feats[i] = hi if p_in == 1 else np.repeat(hi, p_in, axis=0)[:t]

    X, subj, y = [], [], []
    for i, (t, lab) in enumerate(zip(trials, labels)):
        emb = feats[i]
        n = min(emb.shape[0], lab.shape[0])
        X.append(emb[:n]); subj.append(np.full(n, t.subject, np.int64)); y.append(lab[:n])
    return np.concatenate(X), np.concatenate(subj), np.concatenate(y)


def extract_raw_features(trials, labels):
    """Per-epoch raw-DE features (the linear ceiling baseline)."""
    X, subj, y = [], [], []
    for t, lab in zip(trials, labels):
        v = np.asarray(t.values, np.float32).reshape(t.values.shape[0], -1)
        n = min(v.shape[0], lab.shape[0])
        X.append(v[:n]); subj.append(np.full(n, t.subject, np.int64)); y.append(lab[:n])
    return np.concatenate(X), np.concatenate(subj), np.concatenate(y)


def _kfold_fit_one(X, y, subj, test_subj, classifier):
    """One held-out-subject fold — module-level so joblib memmaps X/y once."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    te = np.isin(subj, test_subj)
    tr = ~te
    if tr.sum() == 0 or te.sum() == 0:
        return None
    clf = CLASSIFIERS[classifier]()
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return (accuracy_score(y[te], pred),
            f1_score(y[te], pred, average="macro", zero_division=0),
            cohen_kappa_score(y[te], pred))


def subject_kfold_eval(X, subj, y, k, classifier, n_jobs=-1):
    """Subject-disjoint k-fold; mean±std of acc / macro-F1 / Cohen kappa (%).
    Folds are independent -> fan out across cores (results identical to serial)."""
    from joblib import Parallel, delayed

    subjects = np.array(sorted(set(subj.tolist())))
    rng = np.random.default_rng(42)
    folds = np.array_split(rng.permutation(subjects), k)

    res = Parallel(n_jobs=n_jobs, max_nbytes="1M")(
        delayed(_kfold_fit_one)(X, y, subj, ts, classifier) for ts in folds
    )
    res = [r for r in res if r is not None]
    accs = [r[0] for r in res]
    f1s = [r[1] for r in res]
    kaps = [r[2] for r in res]
    a, f, kp = np.array(accs), np.array(f1s), np.array(kaps)
    return {
        "runs": len(accs),
        "acc_mean": a.mean() * 100, "acc_std": a.std() * 100,
        "f1_mean": f.mean() * 100, "f1_std": f.std() * 100,
        "kappa_mean": kp.mean(), "kappa_std": kp.std(),
        "acc_folds": (a * 100).tolist(),
        "f1_folds": (f * 100).tolist(),
        "kappa_folds": kp.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_dir", default=None, help="PC-pretrained model dir")
    ap.add_argument("--rand_dir", default=None, help="matched random-init model dir")
    ap.add_argument("--raw", action="store_true", help="also run the raw-DE ceiling")
    ap.add_argument("--classifiers", nargs="+", default=["logreg"])
    ap.add_argument("--k", type=int, default=5, help="subject-disjoint folds")
    ap.add_argument("--out_dir", default="results/phase3/f13")
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. _seed1)")
    ap.add_argument("--batch_size", type=int, default=16, help="recordings per encode batch")
    ap.add_argument("--n_jobs", type=int, default=-1, help="parallel fold fits (-1=all cores)")
    ap.add_argument("--self_check", action="store_true",
                    help="verify batched==unbatched extraction on a few recordings, then exit")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trials, labels = _load_recordings()

    if args.self_check:
        md = Path(args.pc_dir or args.rand_dir)
        sub = trials[:12], labels[:12]
        Xb, _, yb = extract_model_features(md, *sub, device, batch_size=8)
        Xu, _, yu = extract_model_features(md, *sub, device, batch_size=1)
        ok = np.allclose(Xb, Xu, atol=1e-4) and np.array_equal(yb, yu)
        LOG.info("SELF_CHECK batched==unbatched: %s (max|Δ|=%.2e, shapes %s/%s)",
                 ok, float(np.abs(Xb - Xu).max()), Xb.shape, Xu.shape)
        raise SystemExit(0 if ok else 1)
    n_ep = sum(t.values.shape[0] for t in trials)
    LOG.info("sleep corpus: %d recordings, %d epochs, n_cb=%d, %d subjects",
             len(trials), n_ep, trials[0].values.shape[1] * trials[0].values.shape[2],
             len({t.subject for t in trials}))

    feature_sets = {}
    if args.raw:
        feature_sets["raw_de"] = extract_raw_features(trials, labels)
    if args.pc_dir:
        feature_sets["physiofm_pc"] = extract_model_features(Path(args.pc_dir), trials, labels, device, args.batch_size)
    if args.rand_dir:
        feature_sets["physiofm_rand"] = extract_model_features(Path(args.rand_dir), trials, labels, device, args.batch_size)
    if not feature_sets:
        raise SystemExit("nothing to evaluate: pass --raw and/or --pc_dir/--rand_dir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (X, subj, y) in feature_sets.items():
        for clf in args.classifiers:
            r = subject_kfold_eval(X, subj, y, args.k, clf, args.n_jobs)
            LOG.info("RESULT %-14s %-8s acc=%.2f±%.2f f1=%.2f±%.2f kappa=%.3f±%.3f (%d folds)",
                     name, clf, r["acc_mean"], r["acc_std"], r["f1_mean"], r["f1_std"],
                     r["kappa_mean"], r["kappa_std"], r["runs"])
            rows.append((name, clf, r))

    csv_path = out_dir / f"f13_sleep{args.tag}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["features", "classifier", "folds", "acc_mean", "acc_std",
                    "f1_mean", "f1_std", "kappa_mean", "kappa_std"])
        for name, clf, r in rows:
            w.writerow([name, clf, r["runs"], f"{r['acc_mean']:.2f}", f"{r['acc_std']:.2f}",
                        f"{r['f1_mean']:.2f}", f"{r['f1_std']:.2f}",
                        f"{r['kappa_mean']:.3f}", f"{r['kappa_std']:.3f}"])

    # per-fold values (for paired significance tests across seeds/models)
    perfold_path = out_dir / f"f13_sleep{args.tag}_perfold.csv"
    with perfold_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["features", "classifier", "fold", "acc", "f1", "kappa"])
        for name, clf, r in rows:
            for i in range(r["runs"]):
                w.writerow([name, clf, i, f"{r['acc_folds'][i]:.4f}",
                            f"{r['f1_folds'][i]:.4f}", f"{r['kappa_folds'][i]:.4f}"])
    LOG.info("wrote %s and %s  (chance=%.1f%%, %d classes: %s)",
             csv_path, perfold_path, 100.0 / len(LABEL_NAMES), len(LABEL_NAMES), LABEL_NAMES)


if __name__ == "__main__":
    main()
