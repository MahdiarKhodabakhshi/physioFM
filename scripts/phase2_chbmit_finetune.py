#!/usr/bin/env python3
"""Fine-tuned seizure evaluation — does the frozen-probe finding replicate on CHB-MIT?

On sleep, end-to-end fine-tuning lifted PC 72.6->75.4 but lifted matched random-init
62.9->73.2, collapsing the pretraining advantage from +9.8 to +2.2 (EXP-0017 §4b). Seizure
is the second per-epoch-label task, so it is the check on whether that is a general property
of the method or a sleep-specific quirk.

Same leave-one-patient-out protocol and imbalance-aware metrics as the frozen eval
(balanced accuracy / sensitivity / specificity / ROC-AUC), with class-weighted BCE since
seizures are ~0.3% of epochs.

    python scripts/phase2_chbmit_finetune.py --pc_dir ... --rand_dir ... --epochs 4
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
from scripts.phase2_chbmit_eval import _load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("chbmit_ft")


def build(ckpt, device):
    import torch.nn as nn

    a = ckpt["args"]
    enc = PhysioFMS(n_cb=ckpt["n_cb"], p_in=a["p_in"], p_out=a["p_out"], variant=a["variant"],
                    hidden=a["hidden"], layers=a["layers"], heads=a["heads"],
                    embedder=a.get("embedder", "linear"))
    enc.load_state_dict(ckpt["state_dict"])
    return enc.to(device), nn.Linear(enc.d, 2).to(device)


def chunk(seq, lab, max_len):
    if max_len <= 0 or seq.shape[0] <= max_len:
        return [(seq, lab)]
    return [(seq[i:i + max_len], lab[i:i + max_len]) for i in range(0, seq.shape[0], max_len)]


def run_fold(ckpt, tr, te, epochs, lr, batch, max_len, device, pos_w):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

    torch.manual_seed(42)
    enc, head = build(ckpt, device)
    params = [p for p in list(enc.parameters()) + list(head.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    w = torch.tensor([1.0, pos_w], dtype=torch.float32, device=device)

    tr_chunks = [c for s, l in tr for c in chunk(s, l, max_len)]
    # keep every seizure-bearing chunk; subsample pure-interictal chunks (0.3% positives
    # otherwise makes almost every step uninformative and the run needlessly long)
    rng = np.random.default_rng(0)
    pos = [c for c in tr_chunks if c[1].max() > 0]
    neg = [c for c in tr_chunks if c[1].max() == 0]
    keep_neg = min(len(neg), max(len(pos) * 3, 200))
    tr_chunks = pos + [neg[i] for i in rng.choice(len(neg), keep_neg, replace=False)]

    for ep in range(epochs):
        enc.train(); head.train()
        order = np.random.default_rng(ep).permutation(len(tr_chunks))
        tot, n = 0.0, 0
        for b0 in range(0, len(order), batch):
            idx = order[b0:b0 + batch]
            xs = [tr_chunks[i][0] for i in idx]; ls = [tr_chunks[i][1] for i in idx]
            x, mask = collate_pad(xs)
            y = np.full((len(idx), x.shape[1]), -100, dtype=np.int64)
            for r, l in enumerate(ls):
                y[r, :len(l)] = l
            logits = head(enc.encode(x.to(device), mask.to(device)))
            loss = F.cross_entropy(logits.reshape(-1, 2),
                                   torch.from_numpy(y).to(device).reshape(-1),
                                   weight=w, ignore_index=-100)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad()
            tot += float(loss.item()); n += 1
        LOG.info("    epoch %d/%d loss=%.4f (%d chunks)", ep + 1, epochs, tot / max(n, 1), len(tr_chunks))

    enc.eval(); head.eval()
    P, G = [], []
    with torch.no_grad():
        for s, l in te:
            for cs, cl in chunk(s, l, max_len):
                x, mask = collate_pad([cs])
                lo = head(enc.encode(x.to(device), mask.to(device)))[0, :len(cl)]
                P.append(torch.softmax(lo, -1)[:, 1].cpu().numpy()); G.append(cl)
    p = np.concatenate(P); g = np.concatenate(G)
    if len(np.unique(g)) < 2:
        return None
    pred = (p >= 0.5).astype(int)
    return (balanced_accuracy_score(g, pred),
            recall_score(g, pred, pos_label=1, zero_division=0),
            recall_score(g, pred, pos_label=0, zero_division=0),
            roc_auc_score(g, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_dir", required=True)
    ap.add_argument("--rand_dir", default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=400)
    ap.add_argument("--out_csv", default="results/phase3/f17/f17_chbmit_finetune.csv")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trials, labels = _load()
    patients = np.array([t.subject for t in trials])
    uniq = sorted(set(patients.tolist()))
    allc = np.concatenate(labels)
    pos_w = float((allc == 0).sum() / max((allc == 1).sum(), 1))
    LOG.info("CHB-MIT: %d recordings, %d patients, positive weight %.1f", len(trials), len(uniq), pos_w)

    rows = []
    for arm, mdir in (("physiofm_pc", args.pc_dir), ("physiofm_rand", args.rand_dir)):
        if mdir is None:
            continue
        mdir = Path(mdir)
        ckpt = torch.load(mdir / "model.pt", map_location=device, weights_only=False)
        mean, std = load_standardizer(mdir / "standardizer.npz")
        seqs = [standardize(t.values, mean, std) for t in trials]
        res = []
        for p in uniq:
            te_m = patients == p
            tr = [(seqs[i], labels[i]) for i in range(len(trials)) if not te_m[i]]
            te = [(seqs[i], labels[i]) for i in range(len(trials)) if te_m[i]]
            LOG.info("  %s patient %s", arm, p)
            r = run_fold(ckpt, tr, te, args.epochs, args.lr, args.batch, args.max_len, device, pos_w)
            if r is not None:
                LOG.info("  %s patient %s -> bal_acc=%.2f sens=%.2f spec=%.2f auc=%.3f",
                         arm, p, r[0] * 100, r[1] * 100, r[2] * 100, r[3])
                res.append(r)
        a = np.array(res)
        LOG.info("RESULT %-14s bal_acc=%.2f±%.2f sens=%.2f spec=%.2f auc=%.3f±%.3f (%d patients)",
                 arm, a[:, 0].mean() * 100, a[:, 0].std() * 100, a[:, 1].mean() * 100,
                 a[:, 2].mean() * 100, a[:, 3].mean(), a[:, 3].std(), len(res))
        rows.append((arm, len(res), a[:, 0].mean() * 100, a[:, 0].std() * 100,
                     a[:, 1].mean() * 100, a[:, 2].mean() * 100, a[:, 3].mean(), a[:, 3].std()))

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["features", "patients", "bal_acc", "bal_acc_std",
                        "sensitivity", "specificity", "auc", "auc_std"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.2f}",
                        f"{r[5]:.2f}", f"{r[6]:.3f}", f"{r[7]:.3f}"])
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
