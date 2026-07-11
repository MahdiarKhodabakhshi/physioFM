#!/usr/bin/env python3
"""F9 — Masked-DE reconstruction pretraining (a static-structure SSL objective).

Alternative to forecasting PC: per DE window, mask a fraction of the 310-d (C×B)
entries and reconstruct them from the unmasked entries + causal context. The
inductive bias is spatial-spectral, matching the static structure the Stage-2
analysis found — so this forks "the temporal objective was wrong" from
"pretraining can't help at all" (see docs/experiments/EXP-0011).

Reuses PhysioFMS with p_in=1, p_out=1 so ``model.forward`` reconstructs the
current window (head: d -> n_cb). The checkpoint is saved in the exact format
``phase2_extract_eval.py`` expects, so the frozen probe is identical to every
other ladder entry.
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
LOG = logging.getLogger("phase2_pretrain_masked")


def make_mask(x, pad_mask, ratio, mode, n_ch, n_band, gen):
    """Boolean mask (B,T,n_cb): True = entry hidden from input & scored in loss.

    ``random`` hides individual (channel,band) entries; ``channel`` hides whole
    channels (all bands); ``band`` hides whole bands (all channels).
    """
    import torch

    b, t, n_cb = x.shape
    if mode == "random":
        m = torch.rand(b, t, n_cb, generator=gen, device=x.device) < ratio
    elif mode == "channel":
        cm = torch.rand(b, t, n_ch, generator=gen, device=x.device) < ratio
        m = cm[..., None].expand(b, t, n_ch, n_band).reshape(b, t, n_cb)
    elif mode == "band":
        bm = torch.rand(b, t, n_band, generator=gen, device=x.device) < ratio
        m = bm[:, :, None, :].expand(b, t, n_ch, n_band).reshape(b, t, n_cb)
    else:
        raise ValueError(f"unknown mask mode {mode!r}")
    return m & (pad_mask.bool()[..., None])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["seed_v", "seed_iv", "seed"])
    ap.add_argument("--mask_ratio", type=float, default=0.5)
    ap.add_argument("--mask_mode", choices=["random", "channel", "band"], default="random")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--embedder", choices=["linear", "attn"], default="linear")
    ap.add_argument("--n_ch", type=int, default=62)
    ap.add_argument("--n_band", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/phase2/followup/f9")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag = f"masked_{args.mask_mode}{int(args.mask_ratio * 100)}_pin1_pout1_{args.embedder}"
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    trials = load_corpus(args.datasets)
    mean, std = fit_standardizer(trials)
    save_standardizer(out / "standardizer.npz", mean, std, args.datasets)
    ds = SequenceDataset(trials, mean, std, min_len=1)
    LOG.info("corpus: %d trials -> %d sequences (datasets=%s)", len(trials), len(ds), args.datasets)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_pad, drop_last=False)

    n_cb = int(mean.shape[0])
    n_ch = args.n_ch if n_cb == args.n_ch * args.n_band else n_cb // args.n_band
    model = PhysioFMS(
        n_cb=n_cb, p_in=1, p_out=1, variant="scratch",
        hidden=args.hidden, layers=args.layers, heads=args.heads,
        embedder=args.embedder, n_ch=n_ch, n_band=args.n_band,
    ).to(device)
    LOG.info("masked-recon n_cb=%d (n_ch=%d n_band=%d) ratio=%.2f mode=%s params=%.2fM",
             n_cb, n_ch, args.n_band, args.mask_ratio, args.mask_mode,
             sum(p.numel() for p in model.parameters()) / 1e6)

    def save():
        # store as p_out=1 scratch so phase2_extract_eval reconstructs the encoder identically
        a = {"variant": "scratch", "p_in": 1, "p_out": 1, "hidden": args.hidden,
             "layers": args.layers, "heads": args.heads, "embedder": args.embedder,
             "objective": "masked_recon", "mask_ratio": args.mask_ratio, "mask_mode": args.mask_mode}
        torch.save({"state_dict": model.state_dict(), "args": a, "n_cb": n_cb}, out / "model.pt")

    if args.epochs == 0:
        save()
        LOG.info("DONE masked NO-PRETRAIN random-init saved=%s", out / "model.pt")
        return

    gen = torch.Generator(device=device).manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, cnt = 0.0, 0
        for x, mask in loader:
            x = x.to(device)
            mask = mask.to(device)
            m = make_mask(x, mask, args.mask_ratio, args.mask_mode, n_ch, args.n_band, gen)
            if m.sum() == 0:
                continue
            x_in = x.masked_fill(m, 0.0)  # 0 == per-(C,B) mean in standardized space
            pred = model(x_in, mask)[:, :, 0, :]  # (B,P=T,n_cb)
            diff = (pred - x) ** 2
            loss = diff[m].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            tot += float(loss.item())
            cnt += 1
        avg = tot / max(cnt, 1)
        LOG.info("epoch %d/%d recon_mse=%.5f (%.0fs)", epoch, args.epochs, avg, time.time() - t0)
        if avg < best:
            best = avg
            save()
    LOG.info("DONE masked best_recon_mse=%.5f saved=%s", best, out / "model.pt")


if __name__ == "__main__":
    main()
