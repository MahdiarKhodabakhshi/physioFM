#!/usr/bin/env python3
"""F14 — Subject-invariance objective on DE, evaluated on strict inductive LOSO.

EXP-0010 (F10) showed within-subject emotion DE is linearly saturated, so the only
regime with headroom is cross-subject (LOSO). This tests whether *targeting
subject-invariance* — the literature's actual cross-subject lever (DANN adversarial
alignment, CORAL, CLISA) — helps, and whether it helps a learned encoder more than
the fixed raw-DE feature.

Arms (per LOSO fold: train on N-1 subjects, test on the held-out subject):
  raw_de        : balanced logistic regression on raw DE (the linear baseline)
  raw_de_coral  : CORAL-align held-out subject to train covariance, then logreg
  dann_l0       : deep encoder + emotion head, NO invariance term (lambda=0 control)
  dann_adv      : same encoder + gradient-reversal subject classifier (adversarial
                  subject-invariance) — the invariance arm

STRICT INDUCTIVE LOSO: the held-out subject's *labels* are never used in training;
CORAL uses only the held-out *inputs* (flagged transductive-input alignment).

    python scripts/phase2_f14_invariance.py --datasets seed seed_iv seed_v
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
from physiofm.phase2_eval import build_raw_de_segments
from physiofm.structured_data import ARCH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("f14")
SEED = 42


# ----------------------------------------------------------------- CORAL
def coral_align(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Recolor target Xt to the source covariance (CORAL). Inputs already centered/scaled."""
    def msqrt(M, power):
        w, V = np.linalg.eigh(M)
        w = np.clip(w, eps, None)
        return (V * (w ** power)) @ V.T

    ds = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(ds)
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(ds)
    # whiten target, then recolor with source covariance
    return Xt @ msqrt(Ct, -0.5) @ msqrt(Cs, 0.5)


# ----------------------------------------------------------------- DANN
def _grad_reverse(x, lambd):
    import torch

    class _GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.lambd = lambd
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return g.neg() * ctx.lambd

    return _GRL.apply(x)


def train_dann(Xtr, ytr, str_subj, Xte, n_classes, lambda_max, epochs, device, seed=SEED):
    """Train encoder + emotion head (+ optional adversarial subject head). Returns test preds."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    n_subj = int(str_subj.max()) + 1
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    sub_t = torch.tensor(str_subj, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    enc = nn.Sequential(
        nn.Linear(Xtr.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 128), nn.ReLU(),
    ).to(device)
    label_head = nn.Linear(128, n_classes).to(device)
    dom_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_subj)).to(device)

    # balanced class weights from train
    cls, cnt = np.unique(ytr, return_counts=True)
    w = np.ones(n_classes, dtype=np.float32)
    w[cls] = cnt.sum() / (len(cls) * cnt)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    ce_dom = nn.CrossEntropyLoss()

    params = list(enc.parameters()) + list(label_head.parameters()) + list(dom_head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    n = Xtr.shape[0]
    bs = 256
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        p = ep / max(epochs - 1, 1)
        lambd = lambda_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
        enc.train(); label_head.train(); dom_head.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb, yb, sb = Xtr_t[idx], ytr_t[idx], sub_t[idx]
            z = enc(xb)
            loss = ce(label_head(z), yb)
            if lambda_max > 0:
                loss = loss + ce_dom(dom_head(_grad_reverse(z, lambd)), sb)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); label_head.eval()
    with torch.no_grad():
        pred = label_head(enc(Xte_t)).argmax(1).cpu().numpy()
    return pred


# ----------------------------------------------------------------- LOSO driver
def run_dataset(ds: str, arms: list[str], epochs: int, lambda_max: float, max_train: int, device):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    trials = load_de_archive(ARCH[ds])
    feats = build_raw_de_segments(trials)
    y_all = LabelEncoder().fit_transform(feats.label)
    n_classes = len(set(y_all.tolist()))
    subjects = sorted(set(feats.subject.tolist()))
    rng = np.random.default_rng(SEED)
    LOG.info("%s: %d windows, %d subjects, %d classes", ds, feats.X.shape[0], len(subjects), n_classes)

    rows = {a: {"acc": [], "f1": []} for a in arms}
    for s in subjects:
        tr = np.where(feats.subject != s)[0]
        te = np.where(feats.subject == s)[0]
        if tr.size == 0 or te.size == 0:
            continue
        scaler = StandardScaler().fit(feats.X[tr])
        Xtr_full, Xte = scaler.transform(feats.X[tr]), scaler.transform(feats.X[te])
        ytr_full, yte = y_all[tr], y_all[te]

        # subsample train for the sklearn arms (parity with loso_eval); deep arms use full
        sub = tr if tr.size <= max_train else rng.choice(np.arange(tr.size), max_train, replace=False)

        for a in arms:
            if a == "raw_de":
                clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
                clf.fit(Xtr_full[sub], ytr_full[sub]); pred = clf.predict(Xte)
            elif a == "raw_de_coral":
                Xte_c = coral_align(Xtr_full[sub], Xte)
                clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
                clf.fit(Xtr_full[sub], ytr_full[sub]); pred = clf.predict(Xte_c)
            elif a in ("dann_l0", "dann_adv"):
                # remap train subject ids to 0..K-1 for the domain head
                tr_subj = feats.subject[tr]
                remap = {v: i for i, v in enumerate(sorted(set(tr_subj.tolist())))}
                sidx = np.array([remap[v] for v in tr_subj], dtype=np.int64)
                lam = lambda_max if a == "dann_adv" else 0.0
                pred = train_dann(Xtr_full, ytr_full, sidx, Xte, n_classes, lam, epochs, device)
            else:
                continue
            rows[a]["acc"].append(accuracy_score(yte, pred))
            rows[a]["f1"].append(f1_score(yte, pred, average="macro", zero_division=0))

    out = []
    for a in arms:
        acc = np.array(rows[a]["acc"]); f1 = np.array(rows[a]["f1"])
        out.append({
            "dataset": ds, "arm": a, "folds": len(acc),
            "acc_mean": acc.mean() * 100, "acc_std": acc.std() * 100,
            "f1_mean": f1.mean() * 100, "f1_std": f1.std() * 100,
        })
        LOG.info("RESULT %-8s %-13s acc=%.2f±%.2f f1=%.2f±%.2f (%d folds)",
                 ds, a, out[-1]["acc_mean"], out[-1]["acc_std"],
                 out[-1]["f1_mean"], out[-1]["f1_std"], len(acc))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["seed", "seed_iv", "seed_v"])
    ap.add_argument("--arms", nargs="+", default=["raw_de", "raw_de_coral", "dann_l0", "dann_adv"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lambda_max", type=float, default=1.0)
    ap.add_argument("--max_train", type=int, default=8000)
    ap.add_argument("--out_dir", default="results/phase3/f14")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for ds in args.datasets:
        all_rows.extend(run_dataset(ds, args.arms, args.epochs,
                                    args.lambda_max, args.max_train, device))

    csv_path = out_dir / "f14_invariance.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "arm", "folds", "acc_mean", "acc_std", "f1_mean", "f1_std"])
        for r in all_rows:
            w.writerow([r["dataset"], r["arm"], r["folds"],
                        f"{r['acc_mean']:.2f}", f"{r['acc_std']:.2f}",
                        f"{r['f1_mean']:.2f}", f"{r['f1_std']:.2f}"])

    md = ["# F14 — Subject-invariance on DE, strict inductive LOSO (acc % / macro-F1 %)", ""]
    for ds in args.datasets:
        md += [f"## {ds}", "", "| arm | acc % | macro-F1 % |", "| --- | ---: | ---: |"]
        for r in all_rows:
            if r["dataset"] == ds:
                md.append(f"| {r['arm']} | {r['acc_mean']:.2f} ± {r['acc_std']:.2f} "
                          f"| {r['f1_mean']:.2f} ± {r['f1_std']:.2f} |")
        md.append("")
    (out_dir / "f14_invariance.md").write_text("\n".join(md))
    LOG.info("wrote %s and %s", csv_path, out_dir / "f14_invariance.md")


if __name__ == "__main__":
    main()
