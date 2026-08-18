#!/usr/bin/env python3
"""E1 — Extract PhysioFM-S per-window embeddings and zero-shot evaluate.

Freezes the pretrained encoder, extracts one embedding per DE window (segment),
and classifies with the canonical subject-dependent harness (PC-SSL splits).
This is the make-or-break Gate 2 signal: does PhysioFM-S beat Phase 1 (chance)
and the raw-DE linear ceiling?
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import load_de_archive
from physiofm.phase2_eval import SegmentFeatures, loso_eval, subject_dependent_eval
from physiofm.physiofm_s import PhysioFMS
from physiofm.structured_data import ARCH, load_standardizer, standardize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("phase2_eval")


def load_model(model_dir: Path, device: str):
    import torch

    ckpt = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
    a = ckpt["args"]
    model = PhysioFMS(
        n_cb=ckpt["n_cb"], p_in=a["p_in"], p_out=a["p_out"], variant=a["variant"],
        hidden=a["hidden"], layers=a["layers"], heads=a["heads"],
        embedder=a.get("embedder", "linear"), causal=bool(a.get("causal", 1)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, a


def extract(model, args, mean, std, trials, device) -> SegmentFeatures:
    import torch

    p_in = args["p_in"]
    X, subj, trid, lab = [], [], [], []
    kept = []
    with torch.no_grad():
        for t in trials:
            if t.label is None or t.values.shape[0] < 1:
                continue
            seq = standardize(t.values, mean, std)  # (T, n_cb)
            x = torch.from_numpy(seq).unsqueeze(0).to(device)
            h = model.encode(x).squeeze(0).float().cpu().numpy()  # (P, d)
            if p_in == 1:
                win_emb = h  # one per window
            else:
                win_emb = np.repeat(h, p_in, axis=0)[: seq.shape[0]]  # assign patch emb to its windows
            n = win_emb.shape[0]
            X.append(win_emb.astype(np.float32))
            subj.append(np.full(n, t.subject, dtype=np.int64))
            trid.append(np.full(n, t.trial, dtype=np.int64))
            lab.append(np.full(n, t.label, dtype=np.int64))
            kept.append(t)
    return SegmentFeatures(
        X=np.concatenate(X), subject=np.concatenate(subj),
        trial=np.concatenate(trid), label=np.concatenate(lab), trials=kept,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=["seed_v", "seed_iv"])
    ap.add_argument("--classifiers", nargs="+", default=["logreg", "linear_svm"])
    ap.add_argument("--protocol", choices=["subject_dependent", "loso"], default="subject_dependent")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(args.model_dir)
    model, margs = load_model(model_dir, device)
    mean, std = load_standardizer(model_dir / "standardizer.npz")
    LOG.info("loaded %s (variant=%s p_in=%d p_out=%d)", model_dir, margs["variant"], margs["p_in"], margs["p_out"])

    results = []
    for ds in args.datasets:
        trials = load_de_archive(ARCH[ds])
        feats = extract(model, margs, mean, std, trials, device)
        LOG.info("%s: %d segments x %d dims", ds, feats.X.shape[0], feats.X.shape[1])
        for clf in args.classifiers:
            if args.protocol == "loso":
                res = loso_eval(feats, ds, classifier=clf, max_train=4000)
            else:
                res = subject_dependent_eval(feats, ds, classifier=clf)
            LOG.info(
                "RESULT %s %s %s [%s] acc=%.2f±%.2f f1=%.2f±%.2f (chance=%.1f)",
                margs["variant"], ds, clf, args.protocol,
                res["accuracy_mean"] * 100, res["accuracy_std"] * 100,
                res["macro_f1_mean"] * 100, res["macro_f1_std"] * 100, res["chance"],
            )
            results.append((margs["variant"], ds, clf, res))

    out_csv = Path(args.out_csv) if args.out_csv else model_dir / "eval_zeroshot.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "dataset", "classifier", "runs", "acc_mean", "acc_std", "f1_mean", "f1_std", "chance"])
        for variant, ds, clf, res in results:
            w.writerow([
                variant, ds, clf, res["runs"],
                f"{res['accuracy_mean'] * 100:.2f}", f"{res['accuracy_std'] * 100:.2f}",
                f"{res['macro_f1_mean'] * 100:.2f}", f"{res['macro_f1_std'] * 100:.2f}",
                f"{res['chance']:.1f}",
            ])
    LOG.info("wrote %s", out_csv)


if __name__ == "__main__":
    main()
