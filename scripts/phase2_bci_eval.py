#!/usr/bin/env python3
"""F16 — motor-imagery (BCI-IV-2a) eval: the 2nd dynamic task.

Trial-level 4-way classification under the canonical **session-holdout** protocol
(train on session T, test on session E, per subject) — leakage-free by construction.
Each trial's DE-window sequence is encoded by the frozen model and mean-pooled to one
vector; raw-DE mean-pool is the linear ceiling. Reports acc / macro-F1 / kappa as
mean ± std across the 9 subjects, for physiofm_pc vs physiofm_rand vs raw_de — the
same PC-vs-random-vs-raw ladder as sleep (EXP-0009), on a second dynamic task.

    python scripts/phase2_bci_eval.py \
        --pc_dir results/phase3/f16/pc/scratch_pin1_pout16_linear \
        --rand_dir results/phase3/f16/rand/scratch_pin1_pout16_linear \
        --raw --classifiers logreg --out_dir results/phase3/f16
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

from physiofm.bci_iv_2a import CLASS_NAMES
from physiofm.de import load_de_archive
from physiofm.phase2_eval import CLASSIFIERS
from physiofm.structured_data import ARCH, collate_pad, load_standardizer, standardize
from scripts.phase2_extract_eval import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("f16_bci")


def _load():
    trials = load_de_archive(ARCH["bci_iv_2a"])
    subj = np.array([t.subject for t in trials], np.int64)
    sess = np.array([t.session for t in trials], np.int64)  # 1=T(train) 2=E(test)
    y = np.array([t.label for t in trials], np.int64)
    return trials, subj, sess, y


def extract_trial_model(model_dir, trials, device, batch_size=256):
    """One mean-pooled encoder vector per trial. Returns X (N, d)."""
    import torch

    model, _ = load_model(model_dir, device)
    model.eval()
    mean, std = load_standardizer(model_dir / "standardizer.npz")
    seqs = [standardize(t.values, mean, std) for t in trials]
    feats = []
    with torch.no_grad():
        for b0 in range(0, len(seqs), batch_size):
            x, mask = collate_pad(seqs[b0 : b0 + batch_size])
            h = model.encode(x.to(device), mask.to(device)).float().cpu().numpy()  # (B,P,d)
            feats.append(h.mean(axis=1))  # mean-pool over windows
    return np.concatenate(feats)


def extract_trial_raw(trials):
    """Mean-pooled raw DE per trial. Returns X (N, n_cb)."""
    return np.stack([np.asarray(t.values, np.float32).reshape(t.values.shape[0], -1).mean(0)
                     for t in trials])


def _fit_subject(X, subj, sess, y, s, classifier):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    tr = (subj == s) & (sess == 1)
    te = (subj == s) & (sess == 2)
    if tr.sum() == 0 or te.sum() == 0:
        return None
    clf = CLASSIFIERS[classifier]()
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return (accuracy_score(y[te], pred),
            f1_score(y[te], pred, average="macro", zero_division=0),
            cohen_kappa_score(y[te], pred))


def session_holdout_eval(X, subj, sess, y, classifier, n_jobs=-1):
    from joblib import Parallel, delayed

    subjects = sorted(set(subj.tolist()))
    res = Parallel(n_jobs=n_jobs, max_nbytes="1M")(
        delayed(_fit_subject)(X, subj, sess, y, s, classifier) for s in subjects
    )
    res = [r for r in res if r is not None]
    a = np.array([r[0] for r in res]); f = np.array([r[1] for r in res]); k = np.array([r[2] for r in res])
    return {"subjects": len(res),
            "acc_mean": a.mean() * 100, "acc_std": a.std() * 100,
            "f1_mean": f.mean() * 100, "f1_std": f.std() * 100,
            "kappa_mean": k.mean(), "kappa_std": k.std(),
            "acc_subj": (a * 100).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_dir", default=None)
    ap.add_argument("--rand_dir", default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--classifiers", nargs="+", default=["logreg"])
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--out_dir", default="results/phase3/f16")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trials, subj, sess, y = _load()
    LOG.info("BCI-IV-2a: %d trials, %d subjects, n_cb=%d, classes=%s",
             len(trials), len(set(subj.tolist())),
             trials[0].values.shape[1] * trials[0].values.shape[2], CLASS_NAMES)

    feats = {}
    if args.raw:
        feats["raw_de"] = extract_trial_raw(trials)
    if args.pc_dir:
        feats["physiofm_pc"] = extract_trial_model(Path(args.pc_dir), trials, device)
    if args.rand_dir:
        feats["physiofm_rand"] = extract_trial_model(Path(args.rand_dir), trials, device)
    if not feats:
        raise SystemExit("nothing to evaluate")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, X in feats.items():
        for clf in args.classifiers:
            r = session_holdout_eval(X, subj, sess, y, clf, args.n_jobs)
            LOG.info("RESULT %-14s %-8s acc=%.2f±%.2f f1=%.2f±%.2f kappa=%.3f±%.3f (%d subj)",
                     name, clf, r["acc_mean"], r["acc_std"], r["f1_mean"], r["f1_std"],
                     r["kappa_mean"], r["kappa_std"], r["subjects"])
            rows.append((name, clf, r))

    with (out_dir / f"f16_bci{args.tag}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["features", "classifier", "subjects", "acc_mean", "acc_std",
                    "f1_mean", "f1_std", "kappa_mean", "kappa_std"])
        for name, clf, r in rows:
            w.writerow([name, clf, r["subjects"], f"{r['acc_mean']:.2f}", f"{r['acc_std']:.2f}",
                        f"{r['f1_mean']:.2f}", f"{r['f1_std']:.2f}",
                        f"{r['kappa_mean']:.3f}", f"{r['kappa_std']:.3f}"])
    LOG.info("wrote %s (chance=25%%, 4 classes)", out_dir / f"f16_bci{args.tag}.csv")


if __name__ == "__main__":
    main()
