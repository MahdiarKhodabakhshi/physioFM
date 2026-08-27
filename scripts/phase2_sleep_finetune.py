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
from physiofm.structured_data import TOKENS_PER_EPOCH, collate_pad, load_standardizer, standardize
import scripts.phase2_f13_sleep as _f13
from scripts.phase2_f13_sleep import _load_recordings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("sleep_ft")

N_CLASSES = 5


def build(ckpt, device, head_kind="linear", lookahead=None, window=None):
    from physiofm.context_head import build_head

    a = ckpt["args"]
    enc = PhysioFMS(n_cb=ckpt["n_cb"], p_in=a["p_in"], p_out=a["p_out"], variant=a["variant"],
                    hidden=a["hidden"], layers=a["layers"], heads=a["heads"],
                    embedder=a.get("embedder", "linear"), causal=bool(a.get("causal", 1)))
    enc.load_state_dict(ckpt["state_dict"])
    head = build_head(head_kind, enc.d, N_CLASSES, lookahead=lookahead, window=window)
    return enc.to(device), head.to(device)


def apply_head(head, feats, lengths=None):
    import torch.nn as nn
    return head(feats) if isinstance(head, nn.Linear) else head(feats, lengths)


def pool_epochs(h, tpe):
    """(B, P, d) token states -> (B, P//tpe, d) per-epoch means (identity when tpe == 1)."""
    if tpe == 1:
        return h
    b, p, d = h.shape
    n = p // tpe
    return h[:, : n * tpe].reshape(b, n, tpe, d).mean(2)


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


def chunk(seq, lab, max_len, tpe=1):
    """Split a whole night into contiguous chunks of ``max_len`` EPOCHS (keeps order/context,
    bounds memory). ``seq`` has ``tpe`` tokens per labelled epoch."""
    n_ep = min(seq.shape[0] // tpe, lab.shape[0])
    seq = seq[: n_ep * tpe]; lab = lab[:n_ep]
    if max_len <= 0 or n_ep <= max_len:
        return [(seq, lab)]
    return [(seq[i * tpe:(i + max_len) * tpe], lab[i:i + max_len]) for i in range(0, n_ep, max_len)]


def mask_labels(chunks, frac, seed):
    """Keep only `frac` of the TRAINING labels (stratified by class); the rest become -100
    so the CE loss ignores them. Sequences stay intact, so the encoder still sees full
    temporal context — only supervision is reduced. This is the fine-tuned analogue of the
    frozen probe's label-fraction sweep."""
    if frac >= 1.0:
        return chunks
    rng = np.random.default_rng(seed)
    flat = np.concatenate([c[1] for c in chunks])
    keep = np.zeros(len(flat), dtype=bool)
    for c in np.unique(flat[flat >= 0]):
        idx = np.where(flat == c)[0]
        n = max(1, int(round(frac * len(idx))))
        keep[rng.choice(idx, size=min(n, len(idx)), replace=False)] = True
    out, off = [], 0
    for x, l in chunks:
        m = keep[off:off + len(l)]
        off += len(l)
        l2 = np.where(m, l, -100).astype(np.int64)
        out.append((x, l2))
    return out


def group_members(pairs, m):
    """[(seq, lab)] -> [((seq_1..seq_m), lab)] for per-electrode groups (identity when m == 1)."""
    if m == 1:
        return [((s,), l) for s, l in pairs]
    assert len(pairs) % m == 0
    out = []
    for i in range(0, len(pairs), m):
        labs = [pairs[i + j][1] for j in range(m)]
        assert all(np.array_equal(labs[0], l) for l in labs)
        out.append((tuple(pairs[i + j][0] for j in range(m)), labs[0]))
    return out


def chunk_group(seqs, lab, max_len, tpe):
    """chunk() applied identically to every member sequence of a group."""
    per = [chunk(s, lab, max_len, tpe) for s in seqs]
    return [(tuple(per[j][k][0] for j in range(len(seqs))), per[0][k][1]) for k in range(len(per[0]))]


def encode_group(enc, xs_groups, device, tpe):
    """xs_groups: list of tuples of member seqs -> per-epoch features (B, P, d) averaged over members."""
    m = len(xs_groups[0])
    flat = [s for g in xs_groups for s in g]
    x, mask = collate_pad(flat)
    h = pool_epochs(enc.encode(x.to(device), mask.to(device)), tpe)     # (B*m, P, d)
    b = len(xs_groups)
    return h.reshape(b, m, h.shape[1], h.shape[2]).mean(1)


def run_fold(ckpt, tr, te, mode, epochs, lr, batch, max_len, device, class_w, label_frac=1.0, tpe=1, merge=1, seed=42,
             head_kind="linear", lookahead=None, window=None):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    torch.manual_seed(seed)
    enc, head = build(ckpt, device, head_kind, lookahead, window)
    params = trainable(enc, head, mode)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    w = torch.tensor(class_w, dtype=torch.float32, device=device)

    tr_chunks = [c for g, l in group_members(tr, merge) for c in chunk_group(g, l, max_len, tpe)]
    tr_chunks = mask_labels(tr_chunks, label_frac, seed=seed)
    for ep in range(epochs):
        enc.train(); head.train()
        order = np.random.default_rng(seed * 1000 + ep).permutation(len(tr_chunks))
        tot, n = 0.0, 0
        for b0 in range(0, len(order), batch):
            idx = order[b0:b0 + batch]
            xs = [tr_chunks[i][0] for i in idx]; ls = [tr_chunks[i][1] for i in idx]
            feats = encode_group(enc, xs, device, tpe)                     # (B, P, d)
            tmax = feats.shape[1]
            y = np.full((len(idx), tmax), -100, dtype=np.int64)
            for r, l in enumerate(ls):
                y[r, :len(l)] = l
            yt = torch.from_numpy(y).to(device)
            logits = apply_head(head, feats, [len(l) for l in ls])
            loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), yt.reshape(-1),
                                   weight=w, ignore_index=-100)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad()
            tot += float(loss.item()); n += 1
        pass  # per-epoch loss suppressed (many fractions x folds)

    enc.eval(); head.eval()
    preds, gts = [], []
    with torch.no_grad():
        for g, l in group_members(te, merge):
            for cs, cl in chunk_group(g, l, max_len, tpe):
                p = apply_head(head, encode_group(enc, [cs], device, tpe))[0, :len(cl)].argmax(-1).cpu().numpy()
                preds.append(p); gts.append(cl)
    p = np.concatenate(preds); g = np.concatenate(gts)
    return (accuracy_score(g, p) * 100,
            f1_score(g, p, average="macro", zero_division=0) * 100,
            cohen_kappa_score(g, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_dir", default=None)
    ap.add_argument("--rand_dir", default=None)
    ap.add_argument("--mode", choices=["full", "io", "head"], default="full")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=400, help="chunk long nights (0=whole night)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--label_fracs", type=float, nargs="+", default=[1.0])
    ap.add_argument("--out_csv", default="results/phase3/f13/f13_sleep_finetune.csv")
    # ---- next-phase plan ----
    ap.add_argument("--arch_key", default="sleep_edf")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--tokens_per_epoch", type=int, default=None)
    ap.add_argument("--latent_dir", default=None, help="latent-objective model dir (arm physiofm_latent)")
    ap.add_argument("--arm", nargs=2, action="append", default=[], metavar=("NAME", "DIR"))
    ap.add_argument("--tag", default="", help="suffix appended to the mode column (e.g. _seed1)")
    ap.add_argument("--merge_every", type=int, default=1, help="per-electrode ablation: group m channel-sequences per recording")
    ap.add_argument("--ft_seed", type=int, default=42)
    ap.add_argument("--head", choices=["linear", "context"], default="linear")
    ap.add_argument("--lookahead", type=int, default=-1, help="context head: -1 = unrestricted bidirectional; k>=0 = bounded future window")
    ap.add_argument("--ctx_window", type=int, default=-1, help="context head: symmetric band |i-j|<=w (ladder-matched context); -1 = unrestricted")
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
    folds = np.array_split(np.random.default_rng(42).permutation(uniq), args.k)  # same split as frozen eval

    rows = []
    arms = [("physiofm_pc", args.pc_dir), ("physiofm_latent", args.latent_dir), ("physiofm_rand", args.rand_dir)]
    arms += [(n, d) for n, d in args.arm]
    for arm, mdir in arms:
        if mdir is None:
            continue
        mdir = Path(mdir)
        ckpt = torch.load(mdir / "model.pt", map_location=device, weights_only=False)
        mean, std = load_standardizer(mdir / "standardizer.npz")
        seqs = [standardize(t.values, mean, std) for t in trials]
        # class weights from the WHOLE corpus incl. test folds (pre-existing SEDF choice,
        # identical across arms; HMC/P2018 use train-only — noted in EXP-0027 review)
        allc = np.concatenate(labels)
        cnt = np.bincount(allc[allc >= 0], minlength=N_CLASSES).astype(np.float64)
        class_w = (cnt.sum() / (N_CLASSES * np.maximum(cnt, 1))).tolist()

        for frac in sorted(args.label_fracs):
            accs, f1s, kaps = [], [], []
            for fi, te_subj in enumerate(folds):
                te_m = np.isin(subjects, te_subj)
                tr = [(seqs[i], labels[i]) for i in range(len(trials)) if not te_m[i]]
                te = [(seqs[i], labels[i]) for i in range(len(trials)) if te_m[i]]
                a, f, kp = run_fold(ckpt, tr, te, args.mode, args.epochs, args.lr,
                                    args.batch, args.max_len, device, class_w, frac, tpe, args.merge_every,
                                    seed=args.ft_seed, head_kind=args.head,
                                    lookahead=None if args.lookahead < 0 else args.lookahead,
                                    window=None if args.ctx_window < 0 else args.ctx_window)
                accs.append(a); f1s.append(f); kaps.append(kp)
            LOG.info("RESULT %-14s mode=%s frac=%.2f acc=%.2f±%.2f f1=%.2f±%.2f kappa=%.3f±%.3f (%d folds)",
                     arm, args.mode, frac, np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s),
                     np.mean(kaps), np.std(kaps), len(accs))
            rows.append((arm, f"{args.mode}_frac{frac:g}{args.tag}", len(accs), np.mean(accs), np.std(accs),
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
