#!/usr/bin/env python3
"""HMC external validation — fixed-split end-to-end fine-tuning.

The tf64 + causal-decoder + PC-pretraining recipe, applied unchanged to HMC
(docs/SLEEP_DATASET_CANDIDATES.md). Differences from phase2_sleep_finetune.py are
protocol-only:

  * FIXED published split (NeuroLM protocol, used by REVE/CSBrain/LaBraM/CBraMod
    reruns): subjects SN001-SN100 train, SN101-SN125 val (reported, not used for
    selection — we train a fixed number of epochs), SN126-SN151 test.
  * Metrics of that ladder: balanced accuracy / Cohen's kappa / weighted-F1
    (plus plain acc and macro-F1 for our own cross-dataset table).
  * --ft_seed actually seeds the fine-tuning run (torch + label shuffling).

Training loop, chunking, class weighting and hyperparameters are imported from /
identical to phase2_sleep_finetune.py so the recipe stays the same.

    python scripts/phase2_hmc_finetune.py --pc_dir <...>/pc --rand_dir <...>/rand \
        --epochs 8 --out_csv results/phase4/hmc/finetune.csv --tag _seed42
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
from physiofm.hmc import split_masks
from physiofm.sleep_edf import load_sleep_labels
from physiofm.structured_data import ARCH, load_standardizer, standardize
from scripts.phase2_sleep_finetune import (
    N_CLASSES, apply_head, build, chunk, mask_labels, trainable,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("hmc_ft")


def encode_batch(enc, xs, device):
    from physiofm.structured_data import collate_pad
    x, mask = collate_pad(xs)
    return enc.encode(x.to(device), mask.to(device))   # (B, P, d); tpe == 1 for tf64


def _eval_pairs(enc, head, pairs, max_len, device):
    import torch
    import numpy as np_
    preds, gts = [], []
    with torch.no_grad():
        for s, l in pairs:
            for cs, cl in chunk(s, l, max_len):
                p = apply_head(head, encode_batch(enc, [cs], device))[0, :len(cl)].argmax(-1).cpu().numpy()
                m = cl >= 0
                preds.append(p[m]); gts.append(cl[m])
    return np_.concatenate(preds), np_.concatenate(gts)


def run_split(ckpt, tr, ev_sets, mode, epochs, lr, batch, max_len, device, class_w, seed,
              collect=(), head_kind="linear", lookahead=None):
    """Train on ``tr``; select the best epoch by VALIDATION Cohen kappa (the published
    ladder's monitor score — NeuroLM App. D.2); report all metrics for every ev set at
    that best epoch (plus the epoch index under key ``_best_epoch``)."""
    import copy
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 cohen_kappa_score, f1_score)

    torch.manual_seed(seed)
    enc, head = build(ckpt, device, head_kind, lookahead)
    params = trainable(enc, head, mode)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    w = torch.tensor(class_w, dtype=torch.float32, device=device)

    tr_chunks = [c for s, l in tr for c in chunk(s, l, max_len)]
    best = (-2.0, -1, None, None)     # (val kappa, epoch, enc state, head state)
    for ep in range(epochs):
        enc.train(); head.train()
        order = np.random.default_rng(seed * 1000 + ep).permutation(len(tr_chunks))
        for b0 in range(0, len(order), batch):
            idx = order[b0:b0 + batch]
            xs = [tr_chunks[i][0] for i in idx]; ls = [tr_chunks[i][1] for i in idx]
            feats = encode_batch(enc, xs, device)
            y = np.full((len(idx), feats.shape[1]), -100, dtype=np.int64)
            for r, l in enumerate(ls):
                y[r, :len(l)] = l
            loss = F.cross_entropy(apply_head(head, feats, [len(l) for l in ls]).reshape(-1, N_CLASSES),
                                   torch.from_numpy(y).to(device).reshape(-1),
                                   weight=w, ignore_index=-100)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad()
        enc.eval(); head.eval()
        vp, vg = _eval_pairs(enc, head, ev_sets["val"], max_len, device)
        vk = cohen_kappa_score(vg, vp)
        LOG.info("  ep %d/%d val kappa %.4f", ep + 1, epochs, vk)
        if vk > best[0]:
            best = (vk, ep,
                    copy.deepcopy({k: v.detach().cpu() for k, v in enc.state_dict().items()}),
                    copy.deepcopy({k: v.detach().cpu() for k, v in head.state_dict().items()}))

    enc.load_state_dict(best[2]); head.load_state_dict(best[3])
    enc.to(device); head.to(device)
    enc.eval(); head.eval()
    out = {"_best_epoch": best[1] + 1, "_pred": {}}
    for name, pairs in ev_sets.items():
            p, g = _eval_pairs(enc, head, pairs, max_len, device)
            if name in collect:
                out["_pred"][name] = (p, g)
            out[name] = dict(
                acc=accuracy_score(g, p) * 100,
                bac=balanced_accuracy_score(g, p) * 100,
                kappa=cohen_kappa_score(g, p),
                mf1=f1_score(g, p, average="macro", zero_division=0) * 100,
                wf1=f1_score(g, p, average="weighted", zero_division=0) * 100,
                n=len(g),
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_dir", default=None)
    ap.add_argument("--rand_dir", default=None)
    ap.add_argument("--arm", nargs=2, action="append", default=[], metavar=("NAME", "DIR"))
    ap.add_argument("--mode", choices=["full", "io", "head"], default="full")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=400)
    ap.add_argument("--label_fracs", type=float, nargs="+", default=[1.0])
    ap.add_argument("--arch_key", default="hmc_tf64")
    ap.add_argument("--labels", default="data/physiofm/tf_features/hmc_labels.npz")
    ap.add_argument("--ft_seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--head", choices=["linear", "context"], default="linear")
    ap.add_argument("--class_weight", choices=["balanced", "none"], default="balanced",
                    help="'none' = plain CE (the NeuroLM-ladder convention favours kappa/wF1)")
    ap.add_argument("--lookahead", type=int, default=-1)
    ap.add_argument("--out_csv", default="results/phase4/hmc/finetune.csv")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trials = load_de_archive(ARCH[args.arch_key])
    labels, subj, night, key = load_sleep_labels(args.labels)
    assert len(trials) == len(labels)
    tr_m, va_m, te_m = split_masks(subj)
    LOG.info("split: %d train / %d val / %d test recordings", tr_m.sum(), va_m.sum(), te_m.sum())

    rows = []
    arms = [("physiofm_pc", args.pc_dir), ("physiofm_rand", args.rand_dir)]
    arms += [(n, d) for n, d in args.arm]
    for arm, mdir in arms:
        if mdir is None:
            continue
        ckpt = torch.load(Path(mdir) / "model.pt", map_location=device, weights_only=False)
        mean, std = load_standardizer(Path(mdir) / "standardizer.npz")
        seqs = [standardize(t.values, mean, std) for t in trials]

        tr = [(seqs[i], labels[i]) for i in range(len(trials)) if tr_m[i]]
        ev = {"val":  [(seqs[i], labels[i]) for i in range(len(trials)) if va_m[i]],
              "test": [(seqs[i], labels[i]) for i in range(len(trials)) if te_m[i]]}

        # class weights from the TRAIN pool only
        allc = np.concatenate([labels[i] for i in range(len(trials)) if tr_m[i]])
        cnt = np.bincount(allc[allc >= 0], minlength=N_CLASSES).astype(np.float64)
        class_w = (cnt.sum() / (N_CLASSES * np.maximum(cnt, 1))).tolist()
        if args.class_weight == "none":
            class_w = [1.0] * N_CLASSES

        for frac in sorted(args.label_fracs):
            tr_f = mask_labels([(s, l) for s, l in tr], frac, seed=args.ft_seed) if frac < 1.0 else tr
            res = run_split(ckpt, tr_f, ev, args.mode, args.epochs, args.lr, args.batch,
                            args.max_len, device, class_w, args.ft_seed,
                            head_kind=args.head,
                            lookahead=None if args.lookahead < 0 else args.lookahead)
            best_ep = res.pop("_best_epoch")
            res.pop("_pred", None)
            LOG.info("best epoch by val kappa: %d", best_ep)
            for split_name, m in res.items():
                LOG.info("RESULT %-13s %s frac=%.2f %-4s acc=%.2f bac=%.2f kappa=%.3f mf1=%.2f wf1=%.2f (n=%d)",
                         arm, args.mode, frac, split_name, m["acc"], m["bac"], m["kappa"],
                         m["mf1"], m["wf1"], m["n"])
                rows.append([arm, f"{args.mode}_frac{frac:g}{args.tag}", split_name,
                             f"{m['acc']:.2f}", f"{m['bac']:.2f}", f"{m['kappa']:.4f}",
                             f"{m['mf1']:.2f}", f"{m['wf1']:.2f}", m["n"]])

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["features", "mode", "split", "acc", "bal_acc", "kappa", "macro_f1", "weighted_f1", "n_epochs"])
        w.writerows(rows)
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
