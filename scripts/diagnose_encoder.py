#!/usr/bin/env python3
"""Diagnostic: is the encoder ADDING information, or lossily RE-ENCODING raw DE?

Motivation. Our margin over the raw-DE linear baseline shrinks monotonically with the
dimensionality of raw DE (sleep 10-d: +5.2; seizure 90-d: +3.1; MI 110-d: -9.4;
emotion 310-d: -3.5). Two very different explanations fit that:

  (H1) BOTTLENECK. The 256-d encoder cannot preserve high-dimensional per-window spectral
       detail, so on wide inputs it destroys information the linear probe needed. The
       pretraining is fine; the architecture throws away the discriminative signal.
  (H2) NO NEW INFORMATION. The encoder only re-expresses what is already in the current
       window and contributes no temporal context, so it can never beat raw features
       except where raw features are very low-dimensional.

The decisive test is CONCATENATION. Give the classifier raw-DE and the encoder features
together:
  * concat > raw   -> the encoder carries COMPLEMENTARY information (temporal context).
                      Our losses are then a bottleneck/readout problem (H1) and are FIXABLE.
  * concat ~ raw   -> the encoder is redundant (H2). The approach adds nothing beyond
                      the features themselves, which is a much deeper problem.

Also reports a random-projection control (raw-DE -> 256-d random nonlinear map) to separate
"deep encoder" effects from plain dimensionality change.

    python scripts/diagnose_encoder.py --task sleep --pc_dir ... --rand_dir ...
    python scripts/diagnose_encoder.py --task emotion --dataset seed_iv --pc_dir ... --rand_dir ...
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("diagnose")


def random_projection(X, dim=256, seed=0):
    """Raw features -> random 256-d nonlinear map (a 'random encoder' with no sequence model).
    Isolates dimensionality/nonlinearity from anything the transformer does."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0 / np.sqrt(X.shape[1]), size=(X.shape[1], dim)).astype(np.float32)
    b = rng.normal(0, 0.01, size=dim).astype(np.float32)
    return np.tanh(X @ W + b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sleep", "emotion", "mi"], required=True)
    ap.add_argument("--dataset", default=None, help="emotion only: seed_iv | seed_iv_raw")
    ap.add_argument("--pc_dir", required=True)
    ap.add_argument("--rand_dir", default=None)
    ap.add_argument("--classifier", default="logreg")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--out_csv", default="results/phase3/diagnose_encoder.csv")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets = {}

    if args.task == "sleep":
        from scripts.phase2_f13_sleep import (_load_recordings, extract_model_features,
                                              extract_raw_features, subject_kfold_eval)
        trials, labels = _load_recordings()
        Xr, subj, y = extract_raw_features(trials, labels)
        Xp, _, _ = extract_model_features(Path(args.pc_dir), trials, labels, device, 16)
        sets["raw_de"] = Xr
        sets["physiofm_pc"] = Xp
        sets["concat_raw+pc"] = np.concatenate([Xr, Xp], axis=1)
        sets["raw_randproj256"] = random_projection(Xr)
        if args.rand_dir:
            Xd, _, _ = extract_model_features(Path(args.rand_dir), trials, labels, device, 16)
            sets["physiofm_rand"] = Xd
        evaluate = lambda X: subject_kfold_eval(X, subj, y, 5, args.classifier, args.n_jobs)
        fmt = lambda r: (r["acc_mean"], r["acc_std"])

    elif args.task == "emotion":
        from physiofm.de import load_de_archive
        from physiofm.phase2_eval import build_raw_de_segments
        from physiofm.structured_data import ARCH
        from scripts.phase2_emotion_parity import extract_model_segments, parity_eval
        ds = args.dataset or "seed_iv"
        trials = load_de_archive(ARCH[ds])
        fr = build_raw_de_segments(trials)
        fp = extract_model_segments(Path(args.pc_dir), trials, device, 32)
        sets["raw_de"] = fr.X
        sets["physiofm_pc"] = fp.X
        sets["concat_raw+pc"] = np.concatenate([fr.X, fp.X], axis=1)
        sets["raw_randproj256"] = random_projection(fr.X)
        if args.rand_dir:
            sets["physiofm_rand"] = extract_model_segments(Path(args.rand_dir), trials, device, 32).X

        def evaluate(X, _fr=fr, _ds=ds):
            import copy
            f2 = copy.copy(_fr); f2.X = X
            return parity_eval(f2, _ds, args.classifier, (1.0,), 1, args.n_jobs)[1.0]
        fmt = lambda r: (r["acc_mean"], r["acc_std"])

    else:  # mi
        from scripts.phase2_bci_eval import (_load, extract_trial_model, extract_trial_raw,
                                             session_holdout_eval)
        trials, subj, sess, y = _load()
        Xr = extract_trial_raw(trials)
        Xp = extract_trial_model(Path(args.pc_dir), trials, device)
        sets["raw_de"] = Xr
        sets["physiofm_pc"] = Xp
        sets["concat_raw+pc"] = np.concatenate([Xr, Xp], axis=1)
        sets["raw_randproj256"] = random_projection(Xr)
        if args.rand_dir:
            sets["physiofm_rand"] = extract_trial_model(Path(args.rand_dir), trials, device)
        evaluate = lambda X: session_holdout_eval(X, subj, sess, y, args.classifier,
                                                  args.n_jobs, (1.0,), 1)[1.0]
        fmt = lambda r: (r["acc_mean"], r["acc_std"])

    rows = []
    for name, X in sets.items():
        r = evaluate(X)
        m, s = fmt(r)
        LOG.info("DIAG %-8s %-18s dim=%4d  acc=%.2f±%.2f", args.task, name, X.shape[1], m, s)
        rows.append((args.task, args.dataset or "", name, X.shape[1], m, s))

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["task", "dataset", "features", "dim", "acc_mean", "acc_std"])
        for row in rows:
            w.writerow([row[0], row[1], row[2], row[3], f"{row[4]:.2f}", f"{row[5]:.2f}"])
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
