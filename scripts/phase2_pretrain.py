#!/usr/bin/env python3
"""E1 — Self-supervised predictive-coding pretraining of PhysioFM-S.

Pretrains on the combined SEED-IV + SEED-V + SEED DE corpus (labels unused).
Objective: at each patch position, predict the next ``p_out`` DE windows (MSE),
generalizing PC-SSL's single-step t+1 to multi-horizon.

Two co-primary variants: --variant scratch | timesfm.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.physiofm_s import PhysioFMS
from physiofm.structured_data import (
    SequenceDataset,
    collate_pad,
    fit_standardizer,
    load_corpus,
    save_standardizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("phase2_pretrain")


def build_targets(x, mask, p_in, p_out):
    import torch

    b, t, n_cb = x.shape
    p = t // p_in
    base = (torch.arange(p, device=x.device) + 1) * p_in  # first predicted window per patch
    tgt_idx = base[:, None] + torch.arange(p_out, device=x.device)[None, :]  # (P, p_out)
    in_range = tgt_idx < t
    clamped = tgt_idx.clamp(max=t - 1)
    flat = clamped.reshape(-1)
    target = x[:, flat].reshape(b, p, p_out, n_cb)
    tgt_mask = mask[:, flat].reshape(b, p, p_out)
    src_valid = mask[:, base.clamp(max=t - 1) - 1]  # last input window of each patch valid
    valid = in_range[None] & (tgt_mask > 0.5) & (src_valid[:, :, None] > 0.5)
    return target, valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["scratch", "timesfm", "timesfm_rand"], default="scratch")
    ap.add_argument("--datasets", nargs="+", default=["seed_v", "seed_iv", "seed"])
    ap.add_argument("--p_in", type=int, default=1)
    ap.add_argument("--p_out", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--embedder", choices=["linear", "attn"], default="linear")
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="freeze transformer layers (default for timesfm transfer test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/phase2/pretrain")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag = f"{args.variant}_pin{args.p_in}_pout{args.p_out}_{args.embedder}"
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    trials = load_corpus(args.datasets)
    mean, std = fit_standardizer(trials)
    save_standardizer(out / "standardizer.npz", mean, std, args.datasets)
    ds = SequenceDataset(trials, mean, std, min_len=args.p_in + 1)
    LOG.info("corpus: %d trials -> %d usable sequences (datasets=%s)", len(trials), len(ds), args.datasets)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_pad, drop_last=False)

    model = PhysioFMS(
        n_cb=mean.shape[0], p_in=args.p_in, p_out=args.p_out, variant=args.variant,
        hidden=args.hidden, layers=args.layers, heads=args.heads, embedder=args.embedder,
    ).to(device)

    freeze = args.freeze_backbone or args.variant in ("timesfm", "timesfm_rand")
    if freeze:
        model.freeze_backbone()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    LOG.info("variant=%s params: trainable=%.2fM / total=%.2fM (freeze_backbone=%s)",
             args.variant, n_train / 1e6, n_total / 1e6, freeze)

    def save():
        torch.save(
            {"state_dict": model.state_dict(), "args": vars(args), "n_cb": int(mean.shape[0])},
            out / "model.pt",
        )

    if args.epochs == 0:  # no-pretrain control: save random-init encoder
        save()
        LOG.info("DONE %s NO-PRETRAIN random-init saved=%s", args.variant, out / "model.pt")
        return

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    best = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, cnt = 0.0, 0
        for x, mask in loader:
            x = x.to(device)
            mask = mask.to(device)
            pred = model(x, mask)
            target, valid = build_targets(x, mask, args.p_in, args.p_out)
            if valid.sum() == 0:
                continue
            diff = (pred - target) ** 2
            loss = diff[valid].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
            tot += float(loss.item())
            cnt += 1
        avg = tot / max(cnt, 1)
        LOG.info("epoch %d/%d pc_mse=%.5f (%.0fs)", epoch, args.epochs, avg, time.time() - t0)
        if avg < best:
            best = avg
            save()
    LOG.info("DONE %s best_pc_mse=%.5f saved=%s", args.variant, best, out / "model.pt")


if __name__ == "__main__":
    main()
