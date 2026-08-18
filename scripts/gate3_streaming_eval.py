#!/usr/bin/env python3
"""Gate 3 (docs/NEXT_PHASE_PLAN.md; EXP-0023): streaming / causal evaluation.

Question: where can a CAUSAL decoder legitimately win? When decisions must be made at time t
from data <= t. We fine-tune, on the same subject-disjoint sleep folds and with the same recipe
as phase2_sleep_finetune.py, (i) our causal PhysioFM-S and (ii) its BIDIRECTIONAL twin
(identical stack, full attention), then score every test epoch two ways:

  offline   the whole chunk (W epochs) is visible — the standard batch evaluation.
  online    only epochs <= t are visible when epoch t is scored. For the causal model this is
            the same forward pass (each position only ever attends to its past); the
            bidirectional model must be re-run on the prefix ending at t (sliding context of
            W epochs), i.e. W tokens of compute per decision instead of 1.

Reported: acc / macro-F1 / kappa offline vs online per arm, and tokens-per-decision.
Arms: --arm NAME DIR (checkpoints from phase2_pretrain.py; --causal 0 twins have `_bidir`).

    python scripts/gate3_streaming_eval.py --arm causal_rand <dir> --arm bidir_rand <dir_bidir> ...
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.structured_data import TOKENS_PER_EPOCH, collate_pad, load_standardizer, standardize
import scripts.phase2_f13_sleep as _f13
from scripts.phase2_f13_sleep import _load_recordings
from scripts.phase2_sleep_finetune import N_CLASSES, build, chunk, mask_labels, pool_epochs, trainable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("gate3")


def train_fold(ckpt, tr, mode, epochs, lr, batch, max_len, device, class_w, tpe):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(42)
    enc, head = build(ckpt, device)
    params = trainable(enc, head, mode)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    w = torch.tensor(class_w, dtype=torch.float32, device=device)
    tr_chunks = [c for s, l in tr for c in chunk(s, l, max_len, tpe)]
    for ep in range(epochs):
        enc.train(); head.train()
        order = np.random.default_rng(ep).permutation(len(tr_chunks))
        for b0 in range(0, len(order), batch):
            idx = order[b0:b0 + batch]
            xs = [tr_chunks[i][0] for i in idx]; ls = [tr_chunks[i][1] for i in idx]
            x, mask = collate_pad(xs)
            y = np.full((len(idx), x.shape[1] // tpe), -100, dtype=np.int64)
            for r, l in enumerate(ls):
                y[r, :len(l)] = l
            logits = head(pool_epochs(enc.encode(x.to(device), mask.to(device)), tpe))
            loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), torch.from_numpy(y).to(device).reshape(-1),
                                   weight=w, ignore_index=-100)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad()
    enc.eval(); head.eval()
    return enc, head


def score(preds, gts):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    p = np.concatenate(preds); g = np.concatenate(gts)
    return (accuracy_score(g, p) * 100, f1_score(g, p, average="macro", zero_division=0) * 100, cohen_kappa_score(g, p))


def eval_offline(enc, head, te, max_len, device, tpe):
    import torch

    preds, gts = [], []
    with torch.no_grad():
        for s, l in te:
            for cs, cl in chunk(s, l, max_len, tpe):
                x, mask = collate_pad([cs])
                p = head(pool_epochs(enc.encode(x.to(device), mask.to(device)), tpe))[0, :len(cl)].argmax(-1).cpu().numpy()
                preds.append(p); gts.append(cl)
    return score(preds, gts)


def eval_online(enc, head, te, max_len, device, tpe, batch=64):
    """Score epoch t from the prefix ending at t (sliding context of max_len epochs).
    For a causal encoder this equals eval_offline on the same chunking; it is run anyway as
    the check that the streaming path is implemented correctly."""
    import torch

    preds, gts, n_dec, tok = [], [], 0, 0
    W = max_len if max_len > 0 else 10 ** 9
    with torch.no_grad():
        for s, l in te:
            for cs, cl in chunk(s, l, max_len, tpe):     # same chunk boundaries as offline
                n_ep = len(cl)
                out = np.zeros(n_ep, dtype=np.int64)
                for b0 in range(0, n_ep, batch):
                    ts = list(range(b0, min(n_ep, b0 + batch)))
                    xs = [cs[: (t + 1) * tpe] for t in ts]        # prefix ending at epoch t
                    x, mask = collate_pad(xs)
                    logits = head(pool_epochs(enc.encode(x.to(device), mask.to(device)), tpe))  # (B, P, C)
                    for r, t in enumerate(ts):
                        out[t] = int(logits[r, t].argmax())
                        tok += (t + 1) * tpe
                    n_dec += len(ts)
                preds.append(out); gts.append(cl)
    return score(preds, gts) + (tok / max(n_dec, 1),)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs=2, action="append", required=True, metavar=("NAME", "DIR"))
    ap.add_argument("--arch_key", default="sleep_edf")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--tokens_per_epoch", type=int, default=None)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=400)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="subset of fold indices (default all)")
    ap.add_argument("--out_csv", default="results/phase4/gate3/streaming.csv")
    args = ap.parse_args()
    _f13.ARCH_KEY = args.arch_key
    if args.labels:
        _f13.LABELS_ARCH = args.labels
    tpe = args.tokens_per_epoch or TOKENS_PER_EPOCH.get(args.arch_key, 1)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trials, labels = _load_recordings()
    subjects = np.array([t.subject for t in trials])
    uniq = np.array(sorted(set(subjects.tolist())))
    folds = np.array_split(np.random.default_rng(42).permutation(uniq), args.k)
    fold_ids = args.folds if args.folds else list(range(len(folds)))
    allc = np.concatenate(labels)
    cnt = np.bincount(allc[allc >= 0], minlength=N_CLASSES).astype(np.float64)
    class_w = (cnt.sum() / (N_CLASSES * np.maximum(cnt, 1))).tolist()

    rows = []
    for name, mdir in args.arm:
        mdir = Path(mdir)
        ckpt = torch.load(mdir / "model.pt", map_location=device, weights_only=False)
        mean, std = load_standardizer(mdir / "standardizer.npz")
        seqs = [standardize(t.values, mean, std) for t in trials]
        off, on = [], []
        for fi in fold_ids:
            te_m = np.isin(subjects, folds[fi])
            tr = [(seqs[i], labels[i]) for i in range(len(trials)) if not te_m[i]]
            te = [(seqs[i], labels[i]) for i in range(len(trials)) if te_m[i]]
            t0 = time.time()
            enc, head = train_fold(ckpt, tr, args.mode, args.epochs, args.lr, args.batch, args.max_len, device, class_w, tpe)
            o = eval_offline(enc, head, te, args.max_len, device, tpe)
            t1 = time.time()
            n = eval_online(enc, head, te, args.max_len, device, tpe)
            t2 = time.time()
            LOG.info("  %s fold %d causal=%s offline acc=%.2f k=%.3f | online acc=%.2f k=%.3f tok/decision=%.0f (%.0fs/%.0fs)",
                     name, fi, bool(ckpt["args"].get("causal", 1)), o[0], o[2], n[0], n[2], n[3], t1 - t0, t2 - t1)
            off.append(o); on.append(n)
        off, on = np.array(off), np.array(on)
        LOG.info("RESULT %-14s causal=%s offline acc=%.2f±%.2f f1=%.2f kappa=%.3f | online acc=%.2f±%.2f f1=%.2f kappa=%.3f tok/decision=%.0f",
                 name, bool(ckpt["args"].get("causal", 1)), off[:, 0].mean(), off[:, 0].std(), off[:, 1].mean(), off[:, 2].mean(),
                 on[:, 0].mean(), on[:, 0].std(), on[:, 1].mean(), on[:, 2].mean(), on[:, 3].mean())
        rows.append((name, bool(ckpt["args"].get("causal", 1)), len(fold_ids), off[:, 0].mean(), off[:, 0].std(), off[:, 1].mean(), off[:, 2].mean(),
                     on[:, 0].mean(), on[:, 0].std(), on[:, 1].mean(), on[:, 2].mean(), on[:, 3].mean()))
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["arm", "causal", "folds", "off_acc", "off_acc_std", "off_f1", "off_kappa",
                        "on_acc", "on_acc_std", "on_f1", "on_kappa", "tokens_per_decision"])
        for r in rows:
            w.writerow([r[0], r[1], r[2]] + [f"{v:.3f}" for v in r[3:]])
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
