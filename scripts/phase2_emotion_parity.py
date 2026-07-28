#!/usr/bin/env python3
"""Emotion evaluated under the SAME analyses as sleep/seizure (protocol parity).

The 4-task claim ("PC pretraining helps in proportion to temporal structure") is a
comparison ACROSS tasks, so every task must get the same analyses. Sleep (EXP-0009)
and seizure (EXP-0015) have: batched frozen-encoder extraction, label-efficiency
curves, an order-shuffle control, per-fold outputs and paired tests. Emotion's
results predate that harness. This script closes the gap.

Protocol is unchanged from the frozen harness: segment-level, PC-SSL
subject-dependent folds (3 folds x subjects), StandardScaler + balanced logreg.
Adds --shuffle_time (the pre-registered NEGATIVE control: on a task with no
temporal gain, shuffling should cost ~nothing) and --label_fracs.

    python scripts/phase2_emotion_parity.py \
        --pc_dir results/.../pc/<tag> --rand_dir results/.../rand/<tag> --raw \
        --datasets seed_iv seed_iv_raw --label_fracs 0.1 1.0 --out_dir results/phase3/parity
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
from physiofm.phase2_eval import (
    CHANCE,
    CLASSIFIERS,
    FOLD_MASK,
    N_FOLDS,
    SegmentFeatures,
    base_dataset,
    build_raw_de_segments,
)
from physiofm.structured_data import ARCH, collate_pad, load_standardizer, standardize
from scripts.phase2_extract_eval import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("emotion_parity")


def extract_model_segments(model_dir: Path, trials, device, batch_size: int = 32,
                           shuffle_time: bool = False, shuffle_seed: int = 0) -> SegmentFeatures:
    """Batched frozen-encoder per-window features (matches the sleep/seizure path).

    shuffle_time permutes each trial's windows BEFORE encoding (scrambling the causal
    context the encoder sees) and inverse-permutes the features afterwards, so labels
    and ids stay aligned. Content is untouched; only temporal order changes.
    """
    import torch

    model, margs = load_model(model_dir, device)
    model.eval()
    mean, std = load_standardizer(model_dir / "standardizer.npz")
    p_in = margs["p_in"]
    if shuffle_time and p_in != 1:
        raise SystemExit("--shuffle_time assumes p_in=1 (per-window alignment)")

    keep = [t for t in trials if t.label is not None and t.values.shape[0] >= 1]
    seqs = [standardize(t.values, mean, std) for t in keep]
    perms = None
    if shuffle_time:
        rng = np.random.default_rng(shuffle_seed)
        perms = [rng.permutation(s.shape[0]) for s in seqs]
        seqs = [s[perms[i]] for i, s in enumerate(seqs)]

    order = sorted(range(len(seqs)), key=lambda i: seqs[i].shape[0])  # length buckets
    feats: list = [None] * len(seqs)
    with torch.no_grad():
        for b0 in range(0, len(order), batch_size):
            idxs = order[b0 : b0 + batch_size]
            batch = [seqs[i] for i in idxs]
            x, mask = collate_pad(batch)
            h = model.encode(x.to(device), mask.to(device)).float().cpu().numpy()
            for j, i in enumerate(idxs):
                t_len = batch[j].shape[0]
                p = t_len // p_in
                hi = h[j, :p]
                feats[i] = hi if p_in == 1 else np.repeat(hi, p_in, axis=0)[:t_len]
    if shuffle_time:
        for i in range(len(feats)):
            feats[i] = feats[i][np.argsort(perms[i])]

    X, subj, trid, lab = [], [], [], []
    for i, t in enumerate(keep):
        e = feats[i]
        n = e.shape[0]
        X.append(e.astype(np.float32))
        subj.append(np.full(n, t.subject, np.int64))
        trid.append(np.full(n, t.trial, np.int64))
        lab.append(np.full(n, t.label, np.int64))
    return SegmentFeatures(X=np.concatenate(X), subject=np.concatenate(subj),
                           trial=np.concatenate(trid), label=np.concatenate(lab), trials=keep)


def _stratified_idx(y_sub, frac, rng):
    keep = []
    for c in np.unique(y_sub):
        idx_c = np.where(y_sub == c)[0]
        n = max(1, int(round(frac * len(idx_c))))
        keep.append(rng.choice(idx_c, size=min(n, len(idx_c)), replace=False))
    return np.concatenate(keep)


def _fit_one(X, y, subject, trial, tsubj, ttrial, fold_fn, s, f, classifier, frac, sseed):
    """One (subject, fold) fit at a given label fraction."""
    from sklearn.metrics import accuracy_score, f1_score

    tr_t, te_t = fold_fn(tsubj, ttrial, s, f)
    train_trials, test_trials = set(ttrial[tr_t].tolist()), set(ttrial[te_t].tolist())
    a = (subject == s) & np.isin(trial, list(train_trials))
    b = (subject == s) & np.isin(trial, list(test_trials))
    if a.sum() == 0 or b.sum() == 0:
        return None
    tr_idx = np.where(a)[0]
    if frac < 1.0:
        rng = np.random.default_rng(10000 * sseed + 100 * int(s) + f)
        tr_idx = tr_idx[_stratified_idx(y[tr_idx], frac, rng)]
    if len(np.unique(y[tr_idx])) < 2:
        return None
    clf = CLASSIFIERS[classifier]()
    clf.fit(X[tr_idx], y[tr_idx])
    pred = clf.predict(X[b])
    return (int(s), f,
            accuracy_score(y[b], pred) * 100,
            f1_score(y[b], pred, average="macro", zero_division=0) * 100)


def parity_eval(feats: SegmentFeatures, dataset: str, classifier="logreg",
                fracs=(1.0,), subsample_seeds=3, n_jobs=-1):
    """Subject-dependent eval (identical splits to the frozen harness), per fraction."""
    from collections import defaultdict

    from joblib import Parallel, delayed
    from sklearn.preprocessing import LabelEncoder

    ds = base_dataset(dataset)
    fold_fn = FOLD_MASK[ds]
    y = LabelEncoder().fit_transform(feats.label)
    tsubj = np.array([t.subject for t in feats.trials])
    ttrial = np.array([t.trial for t in feats.trials])
    subjects = sorted(set(tsubj.tolist()))

    tasks = []
    for frac in fracs:
        seeds = [0] if frac >= 1.0 else list(range(subsample_seeds))
        for s in subjects:
            for f in range(N_FOLDS):
                for sd in seeds:
                    tasks.append((frac, s, f, sd))
    LOG.info("%s: %d fits (%d fracs x %d subj x %d folds), n_jobs=%d",
             dataset, len(tasks), len(fracs), len(subjects), N_FOLDS, n_jobs)
    raw = Parallel(n_jobs=n_jobs, max_nbytes="1M")(
        delayed(_fit_one)(feats.X, y, feats.subject, feats.trial, tsubj, ttrial,
                          fold_fn, s, f, classifier, frac, sd)
        for (frac, s, f, sd) in tasks
    )
    bucket = defaultdict(lambda: defaultdict(list))  # frac -> (subj,fold) -> [(acc,f1)]
    for (frac, s, f, sd), r in zip(tasks, raw):
        if r is not None:
            bucket[frac][(r[0], r[1])].append((r[2], r[3]))

    out = {}
    for frac in fracs:
        per_fold = {k: np.mean(v, axis=0) for k, v in bucket[frac].items()}  # seed-avg
        arr = np.array(list(per_fold.values())) if per_fold else np.zeros((0, 2))
        out[frac] = {"runs": len(per_fold),
                     "acc_mean": arr[:, 0].mean() if len(arr) else 0.0,
                     "acc_std": arr[:, 0].std() if len(arr) else 0.0,
                     "f1_mean": arr[:, 1].mean() if len(arr) else 0.0,
                     "f1_std": arr[:, 1].std() if len(arr) else 0.0,
                     "per_fold": per_fold}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_dir", default=None)
    ap.add_argument("--rand_dir", default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["seed_iv"])
    ap.add_argument("--classifier", default="logreg")
    ap.add_argument("--label_fracs", type=float, nargs="+", default=[1.0])
    ap.add_argument("--subsample_seeds", type=int, default=3)
    ap.add_argument("--shuffle_time", action="store_true")
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--out_dir", default="results/phase3/parity")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fracs = sorted(args.label_fracs)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows, perfold_rows = [], []

    for ds in args.datasets:
        trials = load_de_archive(ARCH[ds])
        feature_sets = {}
        if args.raw:
            feature_sets["raw_de"] = build_raw_de_segments(trials)
        if args.pc_dir:
            feature_sets["physiofm_pc"] = extract_model_segments(
                Path(args.pc_dir), trials, device, args.batch_size, args.shuffle_time, args.shuffle_seed)
        if args.rand_dir:
            feature_sets["physiofm_rand"] = extract_model_segments(
                Path(args.rand_dir), trials, device, args.batch_size, args.shuffle_time, args.shuffle_seed)
        if not feature_sets:
            raise SystemExit("nothing to evaluate")

        for name, feats in feature_sets.items():
            res = parity_eval(feats, ds, args.classifier, fracs, args.subsample_seeds, args.n_jobs)
            for frac in fracs:
                r = res[frac]
                LOG.info("RESULT %-8s %-14s frac=%.2f acc=%.2f±%.2f f1=%.2f±%.2f (%d folds, chance=%.1f%s)",
                         ds, name, frac, r["acc_mean"], r["acc_std"], r["f1_mean"], r["f1_std"],
                         r["runs"], CHANCE[base_dataset(ds)], ", SHUFFLED" if args.shuffle_time else "")
                rows.append((ds, name, frac, r))
                for (s, f), v in sorted(r["per_fold"].items()):
                    perfold_rows.append((ds, name, f"{frac:.2f}", s, f, v[0], v[1]))

    with (out_dir / f"emotion_parity{args.tag}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "features", "label_frac", "folds", "acc_mean", "acc_std", "f1_mean", "f1_std"])
        for ds, name, frac, r in rows:
            w.writerow([ds, name, f"{frac:.2f}", r["runs"], f"{r['acc_mean']:.2f}",
                        f"{r['acc_std']:.2f}", f"{r['f1_mean']:.2f}", f"{r['f1_std']:.2f}"])
    with (out_dir / f"emotion_parity{args.tag}_perfold.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "features", "label_frac", "subject", "fold", "acc", "f1"])
        for row in perfold_rows:
            w.writerow([row[0], row[1], row[2], row[3], row[4], f"{row[5]:.4f}", f"{row[6]:.4f}"])
    LOG.info("wrote %s (+ per-fold)", out_dir / f"emotion_parity{args.tag}.csv")


if __name__ == "__main__":
    main()
