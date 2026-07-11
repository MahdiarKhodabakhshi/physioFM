#!/usr/bin/env python3
"""E3 — Fine-tuned evaluation of PhysioFM-S (the C2/C3 decider).

For each subject×fold of the PC-SSL subject-dependent split, the pretrained
encoder is re-loaded, a fresh linear classification head is attached, and the
model is fine-tuned on that fold's training segments (limited-label fraction),
then evaluated segment-level on the test segments.

Fine-tune modes:
  * full : encoder + head trainable (default for scratch).
  * io   : only structured input/output blocks (patch_in, out_norm) + head;
           transformer layers frozen (default for timesfm transfer).

Causal context is preserved: whole trial sequences are fed, every window is
classified to its trial label, and the limited-label mask restricts the CE loss.
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
from physiofm.embedding_evaluation import seed_iv_fold_mask, seed_v_fold_mask
from physiofm.physiofm_s import PhysioFMS
from physiofm.structured_data import ARCH, load_standardizer, standardize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("phase2_ft")

FOLD = {"seed_v": seed_v_fold_mask, "seed_iv": seed_iv_fold_mask}
CHANCE = {"seed_v": 20.0, "seed_iv": 25.0}
SEED = 42


class Classifier:
    def __init__(self, ckpt, n_classes, device):
        import torch.nn as nn

        a = ckpt["args"]
        self.enc = PhysioFMS(
            n_cb=ckpt["n_cb"], p_in=a["p_in"], p_out=a["p_out"], variant=a["variant"],
            hidden=a["hidden"], layers=a["layers"], heads=a["heads"],
            embedder=a.get("embedder", "linear"),
        )
        self.enc.load_state_dict(ckpt["state_dict"])
        self.head = nn.Linear(self.enc.d, n_classes)
        self.p_in = a["p_in"]
        self.variant = a["variant"]
        self.device = device
        self.enc.to(device)
        self.head.to(device)

    def trainable_params(self, mode):
        if mode == "full":
            params = list(self.enc.parameters()) + list(self.head.parameters())
        elif mode == "io":
            for p in self.enc.layers.parameters():
                p.requires_grad_(False)
            params = (
                list(self.enc.patch_in.parameters())
                + list(self.enc.out_norm.parameters())
                + list(self.head.parameters())
            )
        else:  # head-only: freeze entire encoder, train classifier head
            for p in self.enc.parameters():
                p.requires_grad_(False)
            params = list(self.head.parameters())
        return [p for p in params if p.requires_grad]

    def logits(self, x):  # x: (B,T,n_cb) -> (B,P,n_cls)
        h = self.enc.encode(x)
        return self.head(h)


def make_batches(seqs, labels, bs, device):
    import torch

    order = list(range(len(seqs)))
    for i in range(0, len(order), bs):
        idx = order[i : i + bs]
        tmax = max(seqs[j].shape[0] for j in idx)
        ncb = seqs[0].shape[1]
        x = np.zeros((len(idx), tmax, ncb), dtype=np.float32)
        y = np.full((len(idx), tmax), -100, dtype=np.int64)
        for r, j in enumerate(idx):
            t = seqs[j].shape[0]
            x[r, :t] = seqs[j]
            y[r, :t] = labels[j]
        yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def run_fold(ckpt, n_classes, tr_seqs, tr_lab, te_seqs, te_lab, mode, epochs, lr, frac, device, rng):
    import torch
    import torch.nn.functional as F

    clf = Classifier(ckpt, n_classes, device)
    params = clf.trainable_params(mode)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    # limited-label mask: keep a fraction of train windows for the CE loss
    masked_lab = []
    for lab_seq in tr_lab:
        m = lab_seq.copy()
        if frac < 1.0:
            keep = rng.random(len(m)) < frac
            m = np.where(keep, m, -100)
        masked_lab.append(m.astype(np.int64))

    clf.enc.train()
    clf.head.train()
    for _ in range(epochs):
        for x, y in make_batches(tr_seqs, masked_lab, 32, device):
            logits = clf.logits(x)  # (B,P,C); p_in=1 -> P=T
            loss = F.cross_entropy(logits.reshape(-1, n_classes), y.reshape(-1), ignore_index=-100)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()

    clf.enc.eval()
    clf.head.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in make_batches(te_seqs, te_lab, 64, device):
            logits = clf.logits(x)
            p = logits.argmax(-1).cpu().numpy()
            yy = y.cpu().numpy()
            valid = yy != -100
            preds.append(p[valid])
            gts.append(yy[valid])
    from sklearn.metrics import accuracy_score, f1_score

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    return accuracy_score(gts, preds), f1_score(gts, preds, average="macro", zero_division=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=["seed_v", "seed_iv"])
    ap.add_argument("--mode", choices=["full", "io", "head"], default=None)
    ap.add_argument("--label_fracs", nargs="+", type=float, default=[1.0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(args.model_dir)
    ckpt = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
    variant = ckpt["args"]["variant"]
    mode = args.mode or ("io" if variant == "timesfm" else "full")
    mean, std = load_standardizer(model_dir / "standardizer.npz")
    LOG.info("fine-tune eval: %s variant=%s mode=%s fracs=%s", model_dir, variant, mode, args.label_fracs)

    from sklearn.preprocessing import LabelEncoder

    results = []
    for ds in args.datasets:
        trials = [t for t in load_de_archive(ARCH[ds]) if t.label is not None]
        le = LabelEncoder().fit([t.label for t in trials])
        n_classes = len(le.classes_)
        seqs = {t.trial: standardize(t.values, mean, std) for t in trials}
        labs = {t.trial: int(le.transform([t.label])[0]) for t in trials}
        tsubj = np.array([t.subject for t in trials])
        ttrial = np.array([t.trial for t in trials])

        for frac in args.label_fracs:
            rng = np.random.default_rng(SEED)
            accs, f1s = [], []
            for s in sorted(set(tsubj.tolist())):
                for f in range(3):
                    tr_m, te_m = FOLD[ds](tsubj, ttrial, s, f)
                    tr_ids = ttrial[tr_m].tolist()
                    te_ids = ttrial[te_m].tolist()
                    if not tr_ids or not te_ids:
                        continue
                    torch.manual_seed(SEED)
                    tr_seqs = [seqs[i] for i in tr_ids]
                    tr_lab = [np.full(seqs[i].shape[0], labs[i], dtype=np.int64) for i in tr_ids]
                    te_seqs = [seqs[i] for i in te_ids]
                    te_lab = [np.full(seqs[i].shape[0], labs[i], dtype=np.int64) for i in te_ids]
                    a, fdf = run_fold(ckpt, n_classes, tr_seqs, tr_lab, te_seqs, te_lab,
                                      mode, args.epochs, args.lr, frac, device, rng)
                    accs.append(a)
                    f1s.append(fdf)
            am, asd = float(np.mean(accs) * 100), float(np.std(accs) * 100)
            fm, fsd = float(np.mean(f1s) * 100), float(np.std(f1s) * 100)
            LOG.info("RESULT %s %s mode=%s frac=%.2f acc=%.2f±%.2f f1=%.2f±%.2f (chance=%.1f, folds=%d)",
                     variant, ds, mode, frac, am, asd, fm, fsd, CHANCE[ds], len(accs))
            results.append((variant, ds, mode, frac, len(accs), am, asd, fm, fsd))

    out_csv = Path(args.out_csv) if args.out_csv else model_dir / "eval_finetuned.csv"
    with out_csv.open("w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["variant", "dataset", "mode", "label_frac", "folds", "acc_mean", "acc_std", "f1_mean", "f1_std"])
        for row in results:
            v, ds, m, fr, n, am, asd, fm, fsd = row
            w.writerow([v, ds, m, f"{fr:.2f}", n, f"{am:.2f}", f"{asd:.2f}", f"{fm:.2f}", f"{fsd:.2f}"])
    LOG.info("wrote %s", out_csv)


if __name__ == "__main__":
    main()
