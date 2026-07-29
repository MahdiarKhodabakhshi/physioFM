#!/usr/bin/env python3
"""Fine-tuned sleep evaluation — is the FROZEN PROBE costing us the gap to SOTA?

Every published method we are compared against fine-tunes end-to-end; we report a frozen
encoder + logistic regression. That choice was made on emotion, where per-fold SGD was
unstable on ~600 labels/fold — reasoning that does NOT transfer to sleep (195k labelled
epochs). This script tests it directly on the same subject-disjoint folds.

Modes (what is trainable):
  full : encoder + head (standard fine-tuning, what SOTA baselines do)
  io   : structured input/output blocks + head; transformer layers frozen
  head : head only (a gradient-trained linear probe — the closest analogue of the
         frozen sklearn probe, isolating optimizer effects from representation effects)

Compared arms: PC-pretrained vs matched random-init, so the fine-tuned setting still
answers "does pretraining help" as well as "does fine-tuning close the gap".

    python scripts/phase2_sleep_finetune.py --pc_dir ... --rand_dir ... --mode full --epochs 8
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.physiofm_s import PhysioFMS
from physiofm.structured_data import collate_pad, load_standardizer, standardize
from scripts.phase2_f13_sleep import _load_recordings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("sleep_ft")

N_CLASSES = 5


def build(ckpt, device):
    import torch.nn as nn

    a = ckpt["args"]
    enc = PhysioFMS(n_cb=ckpt["n_cb"], p_in=a["p_in"], p_out=a["p_out"], variant=a["variant"],
                    hidden=a["hidden"], layers=a["layers"], heads=a["heads"],
                    embedder=a.get("embedder", "linear"))
    enc.load_state_dict(ckpt["state_dict"])
    head = nn.Linear(enc.d, N_CLASSES)
    return enc.to(device), head.to(device)


def trainable(enc, head, mode):
    if mode == "full":
        ps = list(enc.parameters()) + list(head.parameters())
    elif mode == "io":
        for p in enc.layers.parameters():
            p.requires_grad_(False)
        ps = list(enc.patch_in.parameters()) + list(enc.out_norm.parameters()) + list(head.parameters())
    else:  # head
        for p in enc.parameters():
            p.requires_grad_(False)
        ps = list(head.parameters())
    return [p for p in ps if p.requires_grad]


def chunk(seq, lab, max_len):
    """Split a whole night into contiguous chunks (keeps order/context, bounds memory)."""
    if max_len <= 0 or seq.shape[0] <= max_len:
        return [(seq, lab)]
    return [(seq[i:i + max_len], lab[i:i + max_len]) for i in range(0, seq.shape[0], max_len)]


def run_fold(ckpt, tr, te, mode, epochs, lr, batch, max_len, device, class_w):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    torch.manual_seed(42)
    enc, head = build(ckpt, device)
    params = trainable(enc, head, mode)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    w = torch.tensor(class_w, dtype=torch.float32, device=device)

    tr_chunks = [c for s, l in tr for c in chunk(s, l, max_len)]
    for ep in range(epochs):
        enc.train(); head.train()
        order = np.random.default_rng(ep).permutation(len(tr_chunks))
        tot, n = 0.0, 0
        for b0 in range(0, len(order), batch):
            idx = order[b0:b0 + batch]
            xs = [tr_chunks[i][0] for i in idx]; ls = [tr_chunks[i][1] for i in idx]
            x, mask = collate_pad(xs)
            tmax = x.shape[1]
            y = np.full((len(idx), tmax), -100, dtype=np.int64)
            for r, l in enumerate(ls):
                y[r, :len(l)] = l
            x = x.to(device); mask = mask.to(device)
            yt = torch.from_numpy(y).to(device)
            logits = head(enc.encode(x, mask))
            loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), yt.reshape(-1),
                                   weight=w, ignore_index=-100)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad()
            tot += float(loss.item()); n += 1
        LOG.info("    epoch %d/%d  loss=%.4f", ep + 1, epochs, tot / max(n, 1))

    enc.eval(); head.eval()
    preds, gts = [], []
    with torch.no_grad():
        for s, l in te:
            for cs, cl in chunk(s, l, max_len):
                x, mask = collate_pad([cs])
                p = head(enc.encode(x.to(device), mask.to(device)))[0, :len(cl)].argmax(-1).cpu().numpy()
                preds.append(p); gts.append(cl)
    p = np.concatenate(preds); g = np.concatenate(gts)
    return (accuracy_score(g, p) * 100,
            f1_score(g, p, average="macro", zero_division=0) * 100,
            cohen_kappa_score(g, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_dir", required=True)
    ap.add_argument("--rand_dir", default=None)
    ap.add_argument("--mode", choices=["full", "io", "head"], default="full")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=400, help="chunk long nights (0=whole night)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out_csv", default="results/phase3/f13/f13_sleep_finetune.csv")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trials, labels = _load_recordings()
    subjects = np.array([t.subject for t in trials])
    uniq = np.array(sorted(set(subjects.tolist())))
    folds = np.array_split(np.random.default_rng(42).permutation(uniq), args.k)  # same split as frozen eval

    rows = []
    for arm, mdir in (("physiofm_pc", args.pc_dir), ("physiofm_rand", args.rand_dir)):
        if mdir is None:
            continue
        mdir = Path(mdir)
        ckpt = torch.load(mdir / "model.pt", map_location=device, weights_only=False)
        mean, std = load_standardizer(mdir / "standardizer.npz")
        seqs = [standardize(t.values, mean, std) for t in trials]
        # class weights from the training pool (sleep is heavily imbalanced toward N2)
        allc = np.concatenate(labels)
        cnt = np.bincount(allc[allc >= 0], minlength=N_CLASSES).astype(np.float64)
        class_w = (cnt.sum() / (N_CLASSES * np.maximum(cnt, 1))).tolist()

        accs, f1s, kaps = [], [], []
        for fi, te_subj in enumerate(folds):
            te_m = np.isin(subjects, te_subj)
            tr = [(seqs[i], labels[i]) for i in range(len(trials)) if not te_m[i]]
            te = [(seqs[i], labels[i]) for i in range(len(trials)) if te_m[i]]
            LOG.info("  %s fold %d/%d (train %d rec, test %d rec)", arm, fi + 1, args.k, len(tr), len(te))
            a, f, kp = run_fold(ckpt, tr, te, args.mode, args.epochs, args.lr,
                                args.batch, args.max_len, device, class_w)
            LOG.info("  %s fold %d -> acc=%.2f f1=%.2f kappa=%.3f", arm, fi + 1, a, f, kp)
            accs.append(a); f1s.append(f); kaps.append(kp)
        LOG.info("RESULT %-14s mode=%s acc=%.2f±%.2f f1=%.2f±%.2f kappa=%.3f±%.3f (%d folds)",
                 arm, args.mode, np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s),
                 np.mean(kaps), np.std(kaps), len(accs))
        rows.append((arm, args.mode, len(accs), np.mean(accs), np.std(accs),
                     np.mean(f1s), np.std(f1s), np.mean(kaps), np.std(kaps)))

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["features", "mode", "folds", "acc_mean", "acc_std",
                        "f1_mean", "f1_std", "kappa_mean", "kappa_std"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.2f}",
                        f"{r[6]:.2f}", f"{r[7]:.3f}", f"{r[8]:.3f}"])
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
