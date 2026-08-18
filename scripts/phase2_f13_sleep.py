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
from physiofm.structured_data import ARCH, TOKENS_PER_EPOCH, collate_pad, load_standardizer, standardize

# reuse the exact model-load pattern from the emotion extractor
from scripts.phase2_extract_eval import load_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("f13_sleep")

LABELS_ARCH = "data/physiofm/de_features/sleep_edf_labels.npz"
# Next-phase plan: evaluators can be pointed at a sibling archive of the SAME recordings
# (sleep_edf_tf64 = 64-bin time-frequency tokens; sleep_edf_raw = raw 200 ms tokens).
# The label companion is identical across them (asserted at build time).
ARCH_KEY = "sleep_edf"


def _load_recordings():
    """Return aligned (trials, per_epoch_labels) for the sleep corpus."""
    trials = load_de_archive(ARCH[ARCH_KEY])
    labels, subj, night, key = load_sleep_labels(LABELS_ARCH)
    if len(trials) != len(labels):
        raise SystemExit(f"archive/labels misalignment: {len(trials)} vs {len(labels)}")
    return trials, labels


def extract_model_features(model_dir: Path, trials, labels, device, batch_size: int = 16,
                           shuffle_time: bool = False, shuffle_seed: int = 0,
                           tokens_per_epoch: int = 1, max_len: int = 0, merge_every: int = 1):
    """Per-epoch frozen-encoder features. Returns X (N,d), subject (N,), y (N,).

    ``merge_every`` m > 1 (per-electrode ablation): consecutive groups of m trials are the m
    channel-sequences of ONE recording; their per-epoch features are averaged into one.

    ``tokens_per_epoch`` > 1 (raw-token archives): the encoder runs over the token
    sequence and the per-epoch feature is the mean of that epoch's token states.
    ``max_len`` > 0: encode in contiguous chunks of that many tokens (memory bound for
    raw nights; each chunk sees only its own context, exactly as in pretraining/fine-tuning).

    Batched with length-bucketing for GPU throughput: recordings are sorted by
    length and padded within a batch. Causal attention + the right-pad mask make
    the valid-position features identical to one-at-a-time encoding (verified in
    `--self_check`), so batching is a pure speedup, not an approximation.

    ``shuffle_time`` (pre-registered order control): permute each recording's epochs
    BEFORE encoding so the causal transformer's temporal context is scrambled, then
    inverse-permute the features back so they still align with the true labels. If the
    PC advantage is temporal, shuffled PC should collapse toward random-init.
    """
    import torch

    model, margs = load_model(model_dir, device)
    model.eval()
    mean, std = load_standardizer(model_dir / "standardizer.npz")
    p_in = margs["p_in"]
    if shuffle_time and (p_in != 1 or tokens_per_epoch != 1):
        raise SystemExit("--shuffle_time assumes p_in=1 and one token per epoch")

    seqs = [standardize(t.values, mean, std) for t in trials]  # (T, n_cb) each
    perms = None
    if shuffle_time:
        rng = np.random.default_rng(shuffle_seed)
        perms = [rng.permutation(s.shape[0]) for s in seqs]
        seqs = [s[perms[i]] for i, s in enumerate(seqs)]

    # chunking (raw tokens): split each sequence into contiguous max_len-token pieces,
    # encode the pieces, pool to per-epoch features INSIDE the loop (token states of a raw
    # corpus would not fit host RAM), and re-assemble in order.
    if max_len and tokens_per_epoch > 1:
        assert max_len % tokens_per_epoch == 0, "max_len must be a multiple of tokens_per_epoch"
    pieces = []  # (seq_idx, start, array)
    for i, s in enumerate(seqs):
        if max_len and s.shape[0] > max_len:
            for st in range(0, s.shape[0], max_len):
                pieces.append((i, st, s[st:st + max_len]))
        else:
            pieces.append((i, 0, s))
    order = sorted(range(len(pieces)), key=lambda k: pieces[k][2].shape[0])  # length buckets
    piece_feats: list = [None] * len(pieces)
    with torch.no_grad():
        for b0 in range(0, len(order), batch_size):
            idxs = order[b0 : b0 + batch_size]
            batch = [pieces[k][2] for k in idxs]
            x, mask = collate_pad(batch)  # (B, Tmax, n_cb), (B, Tmax)
            h = model.encode(x.to(device), mask.to(device)).float()  # (B, Pmax, d) on device
            for j, k in enumerate(idxs):
                t = batch[j].shape[0]
                p = t // p_in
                hi = h[j, :p]
                if p_in != 1:
                    hi = hi.repeat_interleave(p_in, dim=0)[:t]
                if tokens_per_epoch > 1:  # pool token states -> one feature per labelled epoch
                    n_ep = hi.shape[0] // tokens_per_epoch
                    hi = hi[: n_ep * tokens_per_epoch].reshape(n_ep, tokens_per_epoch, -1).mean(1)
                piece_feats[k] = hi.cpu().numpy()
    feats: list = [None] * len(seqs)
    by_seq: dict = {}
    for k, (i, st, _) in enumerate(pieces):
        by_seq.setdefault(i, []).append((st, k))
    for i, lst in by_seq.items():
        feats[i] = np.concatenate([piece_feats[k] for st, k in sorted(lst)], 0)

    if shuffle_time:  # undo the permutation so features re-align with true labels
        for i in range(len(feats)):
            feats[i] = feats[i][np.argsort(perms[i])]

    if merge_every > 1:
        m = merge_every
        assert len(feats) % m == 0, "merge_every must divide the number of sequences"
        feats = [np.mean([feats[i * m + j] for j in range(m)], axis=0) for i in range(len(feats) // m)]
        for i in range(len(feats)):
            for j in range(1, m):
                assert np.array_equal(labels[i * m], labels[i * m + j]), "merged sequences must share labels"
        trials = [trials[i * m] for i in range(len(feats))]
        labels = [labels[i * m] for i in range(len(feats))]

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
    ap.add_argument("--shuffle_time", action="store_true",
                    help="pre-registered order control: scramble epoch order before encoding")
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--self_check", action="store_true",
                    help="verify batched==unbatched extraction on a few recordings, then exit")
    # ---- next-phase plan ----
    ap.add_argument("--arch_key", default="sleep_edf", help="archive key in structured_data.ARCH")
    ap.add_argument("--labels", default=None, help="per-epoch label companion (default: DE one)")
    ap.add_argument("--tokens_per_epoch", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=0, help="encode in chunks of this many tokens (0=whole)")
    ap.add_argument("--latent_dir", default=None, help="latent-objective model dir (arm physiofm_latent)")
    ap.add_argument("--arm", nargs=2, action="append", default=[], metavar=("NAME", "DIR"),
                    help="additional (name, model_dir) arms")
    ap.add_argument("--merge_every", type=int, default=1,
                    help="per-electrode ablation: average features of every m consecutive sequences")
    args = ap.parse_args()
    global ARCH_KEY, LABELS_ARCH
    ARCH_KEY = args.arch_key
    if args.labels:
        LABELS_ARCH = args.labels
    tpe = args.tokens_per_epoch or TOKENS_PER_EPOCH.get(args.arch_key, 1)

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
        if tpe != 1:
            raise SystemExit("--raw (features straight to the classifier) needs one token per epoch")
        feature_sets["raw_de"] = extract_raw_features(trials, labels)
    arms = []
    if args.pc_dir: arms.append(("physiofm_pc", args.pc_dir))
    if args.latent_dir: arms.append(("physiofm_latent", args.latent_dir))
    if args.rand_dir: arms.append(("physiofm_rand", args.rand_dir))
    arms += [(n, d) for n, d in args.arm]
    for name, mdir in arms:
        feature_sets[name] = extract_model_features(
            Path(mdir), trials, labels, device, args.batch_size,
            args.shuffle_time, args.shuffle_seed, tpe, args.max_len, args.merge_every)
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
