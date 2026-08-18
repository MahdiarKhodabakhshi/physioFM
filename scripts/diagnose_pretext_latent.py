#!/usr/bin/env python3
"""Gate 1 mechanism check (EXP-0021 P2): pretext skill in LATENT space vs downstream gain.

EXP-0017 §4e measured, for the input-space objective, that pretext skill (model / persistence
MSE) ANTI-correlates with downstream gain across datasets. This is the same measurement for the
latent-target objective: for a latent-PC checkpoint, compute on the (standardised) corpus

    constant    — predict z_{j+k} := 0            (the per-sequence mean; MSE ~ 1 by construction
                                                   because targets are unit-variance over time)
    persistence — predict z_{j+k} := z_j          (copy the current target embedding)
    shrinkage   — predict z_{j+k} := rho_k * z_j  (best scalar AR map per lag, fit on the corpus;
                                                   the latent analogue of the input-space ridge)
    model       — predictor(h_j) -> z_{j+1..j+p_out}

under the identical masked-MSE metric in the EMA target's instance-normalised latent space
(exactly the training loss). skill = 1 - model / best_trivial, where best_trivial =
min(constant, persistence, shrinkage); skill <= 0 means the predictor has learned nothing a
zero-parameter / one-parameter-per-lag rule does not already give. Together with the
downstream gains this gives the sign of the pretext-vs-transfer correlation for the latent
objective (EXP-0021 P2).

    python scripts/diagnose_pretext_latent.py --dataset sleep_edf --model_dir results/phase4/gate1/sleep_edf/seed42/latent
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
from physiofm.physiofm_s import PhysioFMS
from physiofm.structured_data import ARCH, collate_pad, load_standardizer, standardize
from scripts.phase2_pretrain import LatentPredictor, _instance_norm_time, build_latent_targets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("pretext_latent")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_seqs", type=int, default=120)
    ap.add_argument("--max_len", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out_csv", default="results/phase4/gate1/diagnose_pretext_latent.csv")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdir = Path(args.model_dir)
    ck = torch.load(mdir / "model.pt", map_location=device, weights_only=False)
    a = ck["args"]
    assert a.get("objective") == "latent", "needs a latent-objective checkpoint"
    kw = dict(n_cb=ck["n_cb"], p_in=a["p_in"], p_out=a["p_out"], variant=a["variant"], hidden=a["hidden"],
              layers=a["layers"], heads=a["heads"], embedder=a.get("embedder", "linear"), causal=bool(a.get("causal", 1)))
    online = PhysioFMS(**kw).to(device); online.load_state_dict(ck["state_dict"]); online.eval()
    target = PhysioFMS(**kw).to(device)
    target.load_state_dict(ck.get("target_state_dict", ck["state_dict"])); target.eval()
    used_target = "ema" if "target_state_dict" in ck else "online(no ema saved)"
    pred = LatentPredictor.build(online.d, a["p_out"]).to(device); pred.load_state_dict(ck["latent_predictor"]); pred.eval()
    mean, std = load_standardizer(mdir / "standardizer.npz")

    trials = load_de_archive(ARCH[args.dataset])
    seqs = [standardize(t.values, mean, std) for t in trials]
    if args.max_len:
        seqs = [s[:args.max_len] for s in seqs]
    rng = np.random.default_rng(0)
    if len(seqs) > args.max_seqs:
        seqs = [seqs[i] for i in rng.choice(len(seqs), args.max_seqs, replace=False)]
    p_out, p_in = a["p_out"], a["p_in"]
    is_cos = a.get("latent_loss", "mse") == "cos"
    tot_m, tot_p, tot_c, tot_n = 0.0, 0.0, 0.0, 0
    # AR-shrinkage statistics per lag (feature-averaged): rho_k = <z_j z_{j+k}> / <z_j^2>
    s_xy = torch.zeros(p_out, dtype=torch.float64); s_xx = torch.zeros(p_out, dtype=torch.float64)
    s_yy = torch.zeros(p_out, dtype=torch.float64); n_k = torch.zeros(p_out, dtype=torch.float64)
    with torch.no_grad():
        for b0 in range(0, len(seqs), args.batch):
            x, mask = collate_pad(seqs[b0:b0 + args.batch]); x = x.to(device); mask = mask.to(device)
            h = online.encode(x, mask); z = target.encode(x, mask)
            b, p, d = h.shape
            pv = mask[:, : p * p_in].reshape(b, p, p_in); patch_valid = (pv.sum(-1) > 0).float()
            zn = _instance_norm_time(z, patch_valid) if a.get("target_norm", "instance") == "instance" else z
            tgt, valid = build_latent_targets(zn, patch_valid, p_out)
            zc = zn.unsqueeze(2).expand(-1, -1, p_out, -1)
            if a.get("target_mode", "absolute") == "delta":
                tgt = tgt - zn.unsqueeze(2); zc = torch.zeros_like(zc)
            zhat = pred(h).reshape(b, p, p_out, d)
            if is_cos:
                em = (1 - F.cosine_similarity(zhat, tgt, dim=-1))[valid]
                ep = (1 - F.cosine_similarity(zc, tgt, dim=-1))[valid]
                ec = torch.ones_like(ep)  # cosine to a constant vector is uninformative; treated as 1
            else:
                em = ((zhat - tgt) ** 2).mean(-1)[valid]; ep = ((zc - tgt) ** 2).mean(-1)[valid]
                ec = (tgt ** 2).mean(-1)[valid]
            tot_m += float(em.sum()); tot_p += float(ep.sum()); tot_c += float(ec.sum()); tot_n += int(valid.sum())
            vm = valid.float()                                              # (B, P, p_out)
            s_xy += ((zc * tgt).mean(-1) * vm).sum((0, 1)).double().cpu()
            s_xx += ((zc ** 2).mean(-1) * vm).sum((0, 1)).double().cpu()
            s_yy += ((tgt ** 2).mean(-1) * vm).sum((0, 1)).double().cpu()
            n_k += vm.sum((0, 1)).double().cpu()
    m, pers, const = tot_m / tot_n, tot_p / tot_n, tot_c / tot_n
    rho = s_xy / s_xx.clamp_min(1e-12)
    shrink = float(((s_yy - rho * s_xy) / n_k.clamp_min(1)).mean()) if not is_cos else float("nan")
    best = min(pers, const, shrink) if not is_cos else min(pers, const)
    skill = 1.0 - m / best
    LOG.info("RESULT %-14s latent model=%.4f persistence=%.4f constant=%.4f shrinkage=%.4f -> skill=%.3f "
             "(model/persist=%.3f, target=%s, %d seqs)", args.dataset, m, pers, const, shrink, skill, m / pers,
             used_target, len(seqs))
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["dataset", "model_dir", "model_err", "persistence_err", "constant_err", "shrinkage_err",
                        "skill_vs_best_trivial", "model_over_persistence", "target_used", "n_seqs"])
        w.writerow([args.dataset, str(mdir), f"{m:.5f}", f"{pers:.5f}", f"{const:.5f}", f"{shrink:.5f}",
                    f"{skill:.4f}", f"{m / pers:.4f}", used_target, len(seqs)])


if __name__ == "__main__":
    main()
