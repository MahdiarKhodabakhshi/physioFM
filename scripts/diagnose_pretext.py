#!/usr/bin/env python3
"""Is the PRETEXT TASK actually being learned? (the untested root-cause hypothesis)

Everything so far measured what the pretrained encoder is worth *downstream*. This asks a
prior question: does the model succeed at the job we trained it to do — predicting the next
p_out DE windows?

Three predictors, identical masked-MSE metric, same standardized feature space:
  * persistence — predict every future window = the current window (zero parameters).
  * ridge       — one linear map from the current window to each future window.
  * model       — our pretrained PhysioFM-S.

Interpretation:
  model < ridge < persistence  -> the pretext is being learned; representations are earned.
  model ~ persistence          -> the objective is (near-)degenerate: DE is so autocorrelated
                                  that "copy the last window" is nearly optimal, so the model
                                  learns an identity map and nothing transferable.
  model > persistence          -> the model is FAILING its own pretext — it cannot beat a
                                  zero-parameter baseline, so whatever it learned is not
                                  predictive structure.

    python scripts/diagnose_pretext.py --dataset sleep_edf --model_dir <pc_dir>
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
from physiofm.structured_data import ARCH, collate_pad, load_standardizer, standardize
from scripts.phase2_extract_eval import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("pretext")


def targets(seq, p_out):
    """(positions, p_out, n_cb) future windows + validity mask, matching build_targets()."""
    T = seq.shape[0]
    idx = np.arange(T)[:, None] + 1 + np.arange(p_out)[None, :]
    valid = idx < T
    tgt = seq[np.clip(idx, 0, T - 1)]
    return tgt, valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_seqs", type=int, default=120)
    ap.add_argument("--out_csv", default="results/phase3/diagnose_pretext.csv")
    args = ap.parse_args()

    import torch
    from sklearn.linear_model import Ridge

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdir = Path(args.model_dir)
    model, margs = load_model(mdir, device)
    p_out = margs["p_out"]
    mean, std = load_standardizer(mdir / "standardizer.npz")

    trials = load_de_archive(ARCH[args.dataset])
    seqs = [standardize(t.values, mean, std) for t in trials if t.values.shape[0] > p_out + 1]
    rng = np.random.default_rng(0)
    if len(seqs) > args.max_seqs:
        seqs = [seqs[i] for i in rng.choice(len(seqs), args.max_seqs, replace=False)]
    LOG.info("%s: %d sequences, p_out=%d, n_cb=%d", args.dataset, len(seqs), p_out, seqs[0].shape[1])

    # ---- fit the ridge reference on a held-out-free split of the same data -------------
    Xs, Ys = [], []
    for s in seqs[: max(1, len(seqs) // 2)]:
        tgt, val = targets(s, p_out)
        keep = val.all(axis=1)
        Xs.append(s[keep]); Ys.append(tgt[keep].reshape(keep.sum(), -1))
    Xr = np.concatenate(Xs); Yr = np.concatenate(Ys)
    if Xr.shape[0] > 20000:
        i = rng.choice(Xr.shape[0], 20000, replace=False); Xr, Yr = Xr[i], Yr[i]
    ridge = Ridge(alpha=1.0).fit(Xr, Yr)

    se = {"persistence": 0.0, "ridge": 0.0, "model": 0.0}
    n = 0
    with torch.no_grad():
        for s in seqs:
            tgt, val = targets(s, p_out)                       # (T, p_out, n_cb)
            m = val[..., None]
            # persistence: every future window = the current one
            pers = np.repeat(s[:, None, :], p_out, axis=1)
            # ridge: linear map current -> next p_out windows
            rid = ridge.predict(s).reshape(s.shape[0], p_out, -1)
            # model
            x, mask = collate_pad([s])
            pred = model(x.to(device), mask.to(device))[0].cpu().numpy()  # (T, p_out, n_cb)
            for k, p in (("persistence", pers), ("ridge", rid), ("model", pred)):
                se[k] += float((((p - tgt) ** 2) * m).sum())
            n += int(m.sum())

    mse = {k: v / n for k, v in se.items()}
    LOG.info("PRETEXT %-12s persistence=%.5f  ridge=%.5f  model=%.5f", args.dataset,
             mse["persistence"], mse["ridge"], mse["model"])
    LOG.info("        model/persistence=%.2fx   model/ridge=%.2fx  (<1 is better than baseline)",
             mse["model"] / mse["persistence"], mse["model"] / mse["ridge"])

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["dataset", "p_out", "persistence_mse", "ridge_mse", "model_mse",
                        "model_over_persistence", "model_over_ridge"])
        w.writerow([args.dataset, p_out, f"{mse['persistence']:.6f}", f"{mse['ridge']:.6f}",
                    f"{mse['model']:.6f}", f"{mse['model']/mse['persistence']:.4f}",
                    f"{mse['model']/mse['ridge']:.4f}"])
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
