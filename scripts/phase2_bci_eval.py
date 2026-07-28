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


def extract_trial_model(model_dir, trials, device, batch_size=256,
                        shuffle_time=False, shuffle_seed=0):
    """One mean-pooled encoder vector per trial. Returns X (N, d).

    shuffle_time permutes each trial's DE windows BEFORE encoding — the pre-registered
    NEGATIVE control. The encoder is causal, so each window's hidden state depends on
    which windows precede it; scrambling order therefore changes the pooled feature IF
    the encoder is using temporal context. Prediction for MI (a static spatial-spectral
    task): ~no change, unlike sleep where shuffling erased the whole gain.
    """
    import torch

    model, _ = load_model(model_dir, device)
    model.eval()
    mean, std = load_standardizer(model_dir / "standardizer.npz")
    seqs = [standardize(t.values, mean, std) for t in trials]
    if shuffle_time:
        rng = np.random.default_rng(shuffle_seed)
        seqs = [s[rng.permutation(s.shape[0])] for s in seqs]
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


def _fit_subject(X, subj, sess, y, s, classifier, label_frac=1.0, sseed=0):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    te = (subj == s) & (sess == 2)
    tr_idx = np.where((subj == s) & (sess == 1))[0]
    if tr_idx.size == 0 or te.sum() == 0:
        return None
    if label_frac < 1.0:  # stratified subsample of the training session's trials
        rng = np.random.default_rng(1000 * sseed + int(s))
        parts = []
        for c in np.unique(y[tr_idx]):
            idx_c = tr_idx[y[tr_idx] == c]
            n = max(1, int(round(label_frac * idx_c.size)))
            parts.append(rng.choice(idx_c, size=min(n, idx_c.size), replace=False))
        tr_idx = np.concatenate(parts)
    if len(np.unique(y[tr_idx])) < 2:
        return None
    clf = CLASSIFIERS[classifier]()
    clf.fit(X[tr_idx], y[tr_idx])
    pred = clf.predict(X[te])
    return (int(s),
            accuracy_score(y[te], pred),
            f1_score(y[te], pred, average="macro", zero_division=0),
            cohen_kappa_score(y[te], pred))


def session_holdout_eval(X, subj, sess, y, classifier, n_jobs=-1, fracs=(1.0,), subsample_seeds=3):
    """Per-fraction session-holdout metrics (mean over subjects, seed-averaged)."""
    from collections import defaultdict

    from joblib import Parallel, delayed

    subjects = sorted(set(subj.tolist()))
    tasks = []
    for frac in fracs:
        seeds = [0] if frac >= 1.0 else list(range(subsample_seeds))
        for s in subjects:
            for sd in seeds:
                tasks.append((frac, s, sd))
    raw = Parallel(n_jobs=n_jobs, max_nbytes="1M")(
        delayed(_fit_subject)(X, subj, sess, y, s, classifier, frac, sd) for (frac, s, sd) in tasks
    )
    bucket = defaultdict(lambda: defaultdict(list))  # frac -> subject -> [(acc,f1,kappa)]
    for (frac, s, sd), r in zip(tasks, raw):
        if r is not None:
            bucket[frac][r[0]].append(r[1:])
    out = {}
    for frac in fracs:
        per_subj = {s: np.mean(v, axis=0) for s, v in bucket[frac].items()}
        a = np.array(list(per_subj.values())) if per_subj else np.zeros((0, 3))
        out[frac] = {"subjects": len(per_subj),
                     "acc_mean": a[:, 0].mean() * 100 if len(a) else 0.0,
                     "acc_std": a[:, 0].std() * 100 if len(a) else 0.0,
                     "f1_mean": a[:, 1].mean() * 100 if len(a) else 0.0,
                     "f1_std": a[:, 1].std() * 100 if len(a) else 0.0,
                     "kappa_mean": a[:, 2].mean() if len(a) else 0.0,
                     "kappa_std": a[:, 2].std() if len(a) else 0.0,
                     "per_subject": per_subj}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_dir", default=None)
    ap.add_argument("--rand_dir", default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--classifiers", nargs="+", default=["logreg"])
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--label_fracs", type=float, nargs="+", default=[1.0])
    ap.add_argument("--subsample_seeds", type=int, default=3)
    ap.add_argument("--shuffle_time", action="store_true",
                    help="negative control: scramble window order before encoding")
    ap.add_argument("--shuffle_seed", type=int, default=0)
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
        feats["physiofm_pc"] = extract_trial_model(
            Path(args.pc_dir), trials, device, shuffle_time=args.shuffle_time, shuffle_seed=args.shuffle_seed)
    if args.rand_dir:
        feats["physiofm_rand"] = extract_trial_model(
            Path(args.rand_dir), trials, device, shuffle_time=args.shuffle_time, shuffle_seed=args.shuffle_seed)
    if not feats:
        raise SystemExit("nothing to evaluate")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fracs = sorted(args.label_fracs)
    rows = []
    for name, X in feats.items():
        for clf in args.classifiers:
            res = session_holdout_eval(X, subj, sess, y, clf, args.n_jobs, fracs, args.subsample_seeds)
            for frac in fracs:
                r = res[frac]
                LOG.info("RESULT %-14s %-8s frac=%.2f acc=%.2f±%.2f f1=%.2f±%.2f kappa=%.3f±%.3f (%d subj%s)",
                         name, clf, frac, r["acc_mean"], r["acc_std"], r["f1_mean"], r["f1_std"],
                         r["kappa_mean"], r["kappa_std"], r["subjects"],
                         ", SHUFFLED" if args.shuffle_time else "")
                rows.append((name, clf, frac, r))

    with (out_dir / f"f16_bci{args.tag}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["features", "classifier", "label_frac", "subjects", "acc_mean", "acc_std",
                    "f1_mean", "f1_std", "kappa_mean", "kappa_std"])
        for name, clf, frac, r in rows:
            w.writerow([name, clf, f"{frac:.2f}", r["subjects"], f"{r['acc_mean']:.2f}",
                        f"{r['acc_std']:.2f}", f"{r['f1_mean']:.2f}", f"{r['f1_std']:.2f}",
                        f"{r['kappa_mean']:.3f}", f"{r['kappa_std']:.3f}"])
    with (out_dir / f"f16_bci{args.tag}_persubject.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["features", "classifier", "label_frac", "subject", "acc", "f1", "kappa"])
        for name, clf, frac, r in rows:
            for s, v in sorted(r["per_subject"].items()):
                w.writerow([name, clf, f"{frac:.2f}", s, f"{v[0]*100:.4f}",
                            f"{v[1]*100:.4f}", f"{v[2]:.4f}"])
    LOG.info("wrote %s (+ per-subject; chance=25%%)", out_dir / f"f16_bci{args.tag}.csv")


if __name__ == "__main__":
    main()
