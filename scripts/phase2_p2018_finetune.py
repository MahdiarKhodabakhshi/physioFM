#!/usr/bin/env python3
"""Physio2018 — SleePyCo-split 5-fold end-to-end fine-tuning (third sleep corpus).

Protocol (pinned from the SleePyCo repo + paper by the survey agents, 2026-08-27):
  * SleePyCo's published fold file (data/physiofm/splits/p2018_sleepyco_folds.npy, MIT):
    per fold, test = contiguous 199/198-record block, val = 50 records, train = rest.
  * Metrics POOLED over all test epochs across the 5 folds (the XSleepNet/SleePyCo
    convention): plain accuracy, macro-F1, unweighted Cohen kappa — plus balanced
    accuracy / weighted-F1 for our own cross-dataset table. Per-fold rows also logged.
  * Best fine-tune epoch by val kappa (val never enters training).
  * Per-fold pretraining arms (pc / rand) live in <pretrain_root>/fold{k}/{pc,rand} —
    pretrained only on that fold's non-test recordings (scripts/prepare_p2018_folds.py).

Published rows on this exact split family (single-channel C3-M2, 100 Hz, raw):
  SleePyCo 80.9 acc / 78.9 MF1 / kappa .737 ; XSleepNet2 80.3 / 78.6 / .732 (random
  folds, protocol-matched).

    python scripts/phase2_p2018_finetune.py --arch_key p2018_tf64 \
        --pretrain_root results/phase4/p2018/pretrain --out_csv results/phase4/p2018/finetune.csv
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
from physiofm.physio2018 import load_folds
from physiofm.sleep_edf import load_sleep_labels
from physiofm.structured_data import ARCH, load_standardizer, standardize
from scripts.phase2_hmc_finetune import run_split
from scripts.phase2_sleep_finetune import N_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("p2018_ft")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch_key", default="p2018_tf64")
    ap.add_argument("--labels", default="data/physiofm/tf_features/p2018_labels.npz")
    ap.add_argument("--pretrain_root", default="results/phase4/p2018/pretrain")
    ap.add_argument("--arms", nargs="+", default=["pc", "rand"])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--mode", choices=["full", "io", "head"], default="full")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=400)
    ap.add_argument("--ft_seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--head", choices=["linear", "context"], default="linear")
    ap.add_argument("--lookahead", type=int, default=-1)
    ap.add_argument("--out_csv", default="results/phase4/p2018/finetune.csv")
    args = ap.parse_args()

    import torch
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 cohen_kappa_score, f1_score)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch_path = ARCH.get(args.arch_key, "data/physiofm/tf_features/p2018_tf64_c3.npz"
                         if args.arch_key == "p2018_tf64_c3" else None)
    assert arch_path, f"unknown arch_key {args.arch_key}"
    trials = load_de_archive(arch_path)
    labels, subj, night, key = load_sleep_labels(args.labels)
    assert len(trials) == len(labels) == 994
    folds = load_folds()

    rows = []
    for arm in args.arms:
        pooled_p, pooled_g = [], []
        for k in args.folds:
            f = folds[k - 1]
            mdir = Path(args.pretrain_root) / f"fold{k}" / arm
            ckpt = torch.load(mdir / "model.pt", map_location=device, weights_only=False)
            mean, std = load_standardizer(mdir / "standardizer.npz")
            seqs = {i: standardize(trials[i].values, mean, std)
                    for part in ("train", "val", "test") for i in f[part]}

            tr = [(seqs[i], labels[i]) for i in f["train"]]
            ev = {"val":  [(seqs[i], labels[i]) for i in f["val"]],
                  "test": [(seqs[i], labels[i]) for i in f["test"]]}
            allc = np.concatenate([labels[i] for i in f["train"]])
            cnt = np.bincount(allc[allc >= 0], minlength=N_CLASSES).astype(np.float64)
            class_w = (cnt.sum() / (N_CLASSES * np.maximum(cnt, 1))).tolist()

            res = run_split(ckpt, tr, ev, args.mode, args.epochs, args.lr, args.batch,
                            args.max_len, device, class_w, args.ft_seed,
                            collect=("test",), head_kind=args.head,
                            lookahead=None if args.lookahead < 0 else args.lookahead)
            best_ep = res.pop("_best_epoch")
            m = res["test"]
            LOG.info("RESULT %s fold%d best_ep=%d test acc=%.2f bac=%.2f kappa=%.3f mf1=%.2f wf1=%.2f (n=%d)",
                     arm, k, best_ep, m["acc"], m["bac"], m["kappa"], m["mf1"], m["wf1"], m["n"])
            rows.append([f"physiofm_{arm}", f"{args.mode}{args.tag}", f"fold{k}",
                         f"{m['acc']:.2f}", f"{m['bac']:.2f}", f"{m['kappa']:.4f}",
                         f"{m['mf1']:.2f}", f"{m['wf1']:.2f}", m["n"]])
            pooled_p.append(res["_pred"]["test"][0]); pooled_g.append(res["_pred"]["test"][1])

        p = np.concatenate(pooled_p); g = np.concatenate(pooled_g)
        LOG.info("POOLED %s (%d folds): acc=%.2f bac=%.2f kappa=%.4f mf1=%.2f wf1=%.2f (n=%d)",
                 arm, len(args.folds), accuracy_score(g, p) * 100,
                 balanced_accuracy_score(g, p) * 100, cohen_kappa_score(g, p),
                 f1_score(g, p, average="macro", zero_division=0) * 100,
                 f1_score(g, p, average="weighted", zero_division=0) * 100, len(g))
        rows.append([f"physiofm_{arm}", f"{args.mode}{args.tag}", "pooled",
                     f"{accuracy_score(g, p) * 100:.2f}",
                     f"{balanced_accuracy_score(g, p) * 100:.2f}",
                     f"{cohen_kappa_score(g, p):.4f}",
                     f"{f1_score(g, p, average='macro', zero_division=0) * 100:.2f}",
                     f"{f1_score(g, p, average='weighted', zero_division=0) * 100:.2f}", len(g)])

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["features", "mode", "fold", "acc", "bal_acc", "kappa", "macro_f1", "weighted_f1", "n_epochs"])
        w.writerows(rows)
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
