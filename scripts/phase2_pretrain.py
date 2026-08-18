#!/usr/bin/env python3
"""E1 — Self-supervised predictive-coding pretraining of PhysioFM-S.

Pretrains on the combined SEED-IV + SEED-V + SEED DE corpus (labels unused).
Objective: at each patch position, predict the next ``p_out`` DE windows (MSE),
generalizing PC-SSL's single-step t+1 to multi-horizon.

Two co-primary variants: --variant scratch | timesfm.

Next-phase plan (docs/NEXT_PHASE_PLAN.md, Gate 1) adds a second objective:

  --objective input   (default) input-space predictive coding: predict the next
                      ``p_out`` raw token vectors, masked MSE. Unchanged from Phase 2.
  --objective latent  latent-target predictive coding (JEPA / BYOL-style): the causal
                      decoder at patch j predicts the *embedding* of patches j+1..j+p_out
                      as produced by an EMA target copy of the same encoder (stop-grad).
                      The prediction target is learned, so the model is free to represent
                      what matters rather than the smooth, autocorrelated input component
                      that input-space MSE rewards (EXP-0017 §4e anti-correlation).
                      Collapse guards (data2vec-style): the targets are instance-normalised
                      OVER TIME within each sequence (--target_norm instance, default), so a
                      time-constant embedding cannot fit them; EMA + predictor asymmetry;
                      optional within-sequence variance term (--var_reg); and per-epoch
                      monitoring of within-sequence std / effective rank / a
                      persistence-in-latent-space baseline (predict z_{j+k} := z_j).
                      Loss: --latent_loss mse (default) or cos.

The checkpoint format is unchanged (encoder ``state_dict`` + ``args`` + ``n_cb``); the
latent predictor is stored under ``latent_predictor`` so downstream frozen / fine-tuned
evaluators load either objective identically. ``--epochs 0`` still saves the matched
random-init control.
"""
from __future__ import annotations

import argparse
import copy
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


def build_latent_targets(z, patch_valid, p_out):
    """Latent-space analogue of build_targets, in PATCH units.

    z: (B, P, d) target-encoder embeddings; patch_valid: (B, P) in {0,1}.
    Returns target (B, P, p_out, d) = z[:, j+1 .. j+p_out] and valid (B, P, p_out).
    """
    import torch

    b, p, d = z.shape
    src = torch.arange(p, device=z.device)
    tgt_idx = src[:, None] + 1 + torch.arange(p_out, device=z.device)[None, :]  # (P, p_out)
    in_range = tgt_idx < p
    flat = tgt_idx.clamp(max=p - 1).reshape(-1)
    target = z[:, flat].reshape(b, p, p_out, d)
    tgt_valid = patch_valid[:, flat].reshape(b, p, p_out) > 0.5
    valid = in_range[None] & tgt_valid & (patch_valid[:, :, None] > 0.5)
    return target, valid


class LatentPredictor:
    """Small MLP predictor h_j -> [ẑ_{j+1}, ..., ẑ_{j+p_out}] (BYOL-style asymmetry)."""

    @staticmethod
    def build(d: int, p_out: int, hidden_mult: int = 2):
        import torch.nn as nn

        net = nn.Sequential(
            nn.Linear(d, hidden_mult * d), nn.GELU(), nn.Linear(hidden_mult * d, p_out * d)
        )
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        return net


def _instance_norm_time(z, patch_valid, eps: float = 1e-5):
    """Normalise each sequence's embeddings over its VALID time steps, per feature.
    (data2vec-style target normalisation: a time-constant embedding cannot fit these targets.)"""
    m = patch_valid.unsqueeze(-1)                                    # (B, P, 1)
    n = m.sum(1, keepdim=True).clamp_min(1.0)                        # (B, 1, 1)
    mu = (z * m).sum(1, keepdim=True) / n
    var = (((z - mu) ** 2) * m).sum(1, keepdim=True) / n
    return (z - mu) / (var + eps).sqrt() * m


def _within_seq_std(h, patch_valid, eps: float = 1e-5):
    """Per-sequence, per-feature std over valid time steps -> (B, d)."""
    m = patch_valid.unsqueeze(-1)
    n = m.sum(1, keepdim=True).clamp_min(1.0)
    mu = (h * m).sum(1, keepdim=True) / n
    var = (((h - mu) ** 2) * m).sum(1, keepdim=True) / n
    return (var.squeeze(1) + eps).sqrt()


def _effective_rank(h: "torch.Tensor", n: int = 4096) -> float:
    """exp(entropy of normalized singular values) of a token sample — 1 == collapsed."""
    import torch

    if h.shape[0] > n:
        idx = torch.randperm(h.shape[0], device=h.device)[:n]
        h = h[idx]
    h = h - h.mean(0, keepdim=True)
    s = torch.linalg.svdvals(h.float())
    p = s / s.sum().clamp_min(1e-12)
    ent = -(p * (p + 1e-12).log()).sum()
    return float(ent.exp())


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
    # ---- next-phase plan ----------------------------------------------------------
    ap.add_argument("--objective", choices=["input", "latent"], default="input",
                    help="input-space MSE (Phase 2) or latent-target prediction (Gate 1)")
    ap.add_argument("--ema", type=float, default=0.996, help="latent: EMA momentum of the target encoder")
    ap.add_argument("--ema_final", type=float, default=1.0,
                    help="latent: cosine-anneal EMA momentum from --ema to this value")
    ap.add_argument("--var_reg", type=float, default=0.0,
                    help="latent: within-sequence variance penalty weight on online embeddings (0=off)")
    ap.add_argument("--target_norm", choices=["none", "instance"], default="instance",
                    help="latent: instance-normalise target embeddings over time within each sequence")
    ap.add_argument("--latent_loss", choices=["mse", "cos"], default="mse")
    ap.add_argument("--target_mode", choices=["absolute", "delta"], default="absolute",
                    help="latent: predict z_{j+k} (absolute) or z_{j+k} - z_j (delta; persistence == predicting 0)")
    ap.add_argument("--max_len", type=int, default=0,
                    help="split sequences longer than this many tokens into contiguous chunks")
    ap.add_argument("--tag", default=None, help="override the output sub-directory name")
    ap.add_argument("--causal", type=int, default=1, help="1=causal decoder (default), 0=bidirectional twin")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag = args.tag or (
        f"{args.variant}_pin{args.p_in}_pout{args.p_out}_{args.embedder}"
        + ("_latent" if args.objective == "latent" else "")
        + ("" if args.causal else "_bidir")
    )
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    trials = load_corpus(args.datasets)
    mean, std = fit_standardizer(trials)
    save_standardizer(out / "standardizer.npz", mean, std, args.datasets)
    ds = SequenceDataset(trials, mean, std, min_len=args.p_in + 1, max_len=args.max_len)
    LOG.info("corpus: %d trials -> %d usable sequences (datasets=%s, max_len=%d)",
             len(trials), len(ds), args.datasets, args.max_len)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_pad, drop_last=False)

    model = PhysioFMS(
        n_cb=mean.shape[0], p_in=args.p_in, p_out=args.p_out, variant=args.variant,
        hidden=args.hidden, layers=args.layers, heads=args.heads, embedder=args.embedder,
        causal=bool(args.causal),
    ).to(device)

    freeze = args.freeze_backbone or args.variant in ("timesfm", "timesfm_rand")
    if freeze:
        model.freeze_backbone()

    predictor = None
    target = None
    if args.objective == "latent":
        predictor = LatentPredictor.build(model.d, args.p_out).to(device)
        target = copy.deepcopy(model).to(device)
        for p in target.parameters():
            p.requires_grad_(False)
        target.eval()

    params = [p for p in model.parameters() if p.requires_grad]
    if predictor is not None:
        params += list(predictor.parameters())
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    LOG.info("variant=%s objective=%s params: trainable=%.2fM / encoder total=%.2fM (freeze_backbone=%s)",
             args.variant, args.objective, n_train / 1e6, n_total / 1e6, freeze)

    def save(extra=None):
        ck = {"state_dict": model.state_dict(), "args": vars(args), "n_cb": int(mean.shape[0])}
        if predictor is not None:
            ck["latent_predictor"] = predictor.state_dict()
            ck["target_state_dict"] = target.state_dict()   # EMA target encoder (for the pretext diagnostic)
        if extra:
            ck.update(extra)
        torch.save(ck, out / "model.pt")

    if args.epochs == 0:  # no-pretrain control: save random-init encoder
        save()
        (out / "DONE").write_text("epochs=0 random-init\n")
        LOG.info("DONE %s NO-PRETRAIN random-init saved=%s", args.variant, out / "model.pt")
        return

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    best = float("inf")
    t0 = time.time()
    total_steps = args.epochs * max(len(loader), 1)
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, cnt = 0.0, 0
        # collapse / pretext monitors (latent objective)
        mon_std, mon_rank, mon_persist, mon_pred, mon_const, mon_n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for x, mask in loader:
            x = x.to(device)
            mask = mask.to(device)
            if args.objective == "input":
                pred = model(x, mask)
                tgt, valid = build_targets(x, mask, args.p_in, args.p_out)
                if valid.sum() == 0:
                    continue
                loss = ((pred - tgt) ** 2)[valid].mean()
            else:
                h = model.encode(x, mask)                       # (B, P, d) online
                b, p, d = h.shape
                pv = mask[:, : p * args.p_in].reshape(b, p, args.p_in)
                patch_valid = (pv.sum(-1) > 0).float()
                with torch.no_grad():
                    z = target.encode(x, mask)                  # (B, P, d) EMA target, stop-grad
                    zn = _instance_norm_time(z, patch_valid) if args.target_norm == "instance" else z
                    tgt, valid = build_latent_targets(zn, patch_valid, args.p_out)
                    if args.target_mode == "delta":
                        tgt = tgt - zn.unsqueeze(2)          # predict the CHANGE of the latent
                if valid.sum() == 0:
                    continue
                zhat = predictor(h).reshape(b, p, args.p_out, d)
                if args.latent_loss == "cos":
                    per = 1.0 - F.cosine_similarity(zhat, tgt, dim=-1)          # (B, P, p_out)
                else:
                    per = ((zhat - tgt) ** 2).mean(-1)                          # (B, P, p_out)
                loss = per[valid].mean()
                if args.var_reg > 0:
                    hstd = _within_seq_std(h, patch_valid)                      # (B, d)
                    loss = loss + args.var_reg * F.relu(1.0 - hstd).mean()
                with torch.no_grad():
                    # persistence-in-latent baseline: predict z_{j+k} := z_j (copy the current
                    # normalised target embedding). The predictor must beat this to be
                    # learning dynamics rather than smoothness.
                    zc = zn.unsqueeze(2).expand(-1, -1, args.p_out, -1)
                    if args.target_mode == "delta":
                        zc = torch.zeros_like(zc)            # persistence == zero change
                    if args.latent_loss == "cos":
                        pers = (1.0 - F.cosine_similarity(zc, tgt, dim=-1))[valid].mean()
                    else:
                        pers = ((zc - tgt) ** 2).mean(-1)[valid].mean()
                    mon_std += float(_within_seq_std(z, patch_valid).mean()); mon_persist += float(pers)
                    mon_pred += float(per[valid].mean()); mon_n += 1
                    mon_const += float((tgt ** 2).mean(-1)[valid].mean()) if args.latent_loss == "mse" else 1.0
                    if mon_n % 20 == 1:
                        mon_rank += _effective_rank(z[patch_valid > 0.5])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if target is not None:
                # EMA update of the target encoder (cosine schedule ema -> ema_final)
                m = args.ema_final - (args.ema_final - args.ema) * (
                    0.5 * (1.0 + float(np.cos(np.pi * step / max(total_steps, 1)))))
                with torch.no_grad():
                    for pt, po in zip(target.parameters(), model.parameters()):
                        pt.mul_(m).add_(po.detach(), alpha=1.0 - m)
            tot += float(loss.item())
            cnt += 1
        avg = tot / max(cnt, 1)
        if args.objective == "input":
            LOG.info("epoch %d/%d pc_mse=%.5f (%.0fs)", epoch, args.epochs, avg, time.time() - t0)
        else:
            nr = max(mon_n // 20 + (1 if mon_n % 20 else 0), 1)
            LOG.info("epoch %d/%d latent_loss=%.4f pred_err=%.4f persist_err=%.4f const_err=%.4f "
                     "z_within_seq_std=%.3f eff_rank=%.1f (%.0fs)",
                     epoch, args.epochs, avg, mon_pred / max(mon_n, 1), mon_persist / max(mon_n, 1),
                     mon_const / max(mon_n, 1), mon_std / max(mon_n, 1), mon_rank / nr, time.time() - t0)
        if avg < best:
            best = avg
            save({"epoch": epoch, "best_loss": best})
    (out / "DONE").write_text(f"epochs={args.epochs} best_loss={best:.6f}\n")  # completion sentinel for the drivers
    LOG.info("DONE %s objective=%s best_loss=%.5f saved=%s", args.variant, args.objective, best, out / "model.pt")


if __name__ == "__main__":
    main()
