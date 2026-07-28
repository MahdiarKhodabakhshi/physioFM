#!/usr/bin/env python3
"""Compute per-dataset TEMPORAL STRUCTURE metrics (the dose-response x-axis).

The 4-task claim is really a dose-response: predictive-coding pretraining should help
in proportion to how much learnable temporal structure the features carry. This script
computes that quantity from the RAW features alone — before any pretraining — so it can
be plotted against the measured PC-vs-random gain.

Two metrics (both cheap, CPU-only, no GPU / no model needed):

  1. within_var_frac — fraction of each (channel,band) feature's variance that is
     WITHIN-sequence rather than between-sequence. The F1 quantity: LDS-smoothed
     emotion DE ~0.1% (a static per-trial level), un-smoothed ~17.2%.

  2. tau_k — the k-step PREDICTABILITY GAP:
         tau = 1 - MSE(ridge: x_t -> x_{t+k}) / MSE(persistence: x_{t+k} = x_t)
     i.e. how much a cheap learned predictor beats last-value carry-forward, held out.
     tau ~ 0 => nothing to forecast (the SSL pretext is degenerate); tau > 0 => the
     future is learnably predictable from the past, which is exactly what the
     predictive-coding objective trains on.

    python scripts/compute_temporal_structure.py --datasets seed_iv seed_iv_raw sleep_edf chbmit bci_iv_2a
    python scripts/compute_temporal_structure.py --datasets seed_iv --shuffle   # control: tau -> 0
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
from physiofm.structured_data import ARCH, fit_standardizer, standardize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("tau")


def within_variance_fraction(seqs: list[np.ndarray]) -> float:
    """Mean over features of within-sequence variance / total variance."""
    n_f = seqs[0].shape[1]
    tot_sum = np.zeros(n_f); tot_sq = np.zeros(n_f); n = 0
    within_ss = np.zeros(n_f)
    for s in seqs:
        if s.shape[0] < 2:
            continue
        m = s.mean(0)
        within_ss += ((s - m) ** 2).sum(0)
        tot_sum += s.sum(0); tot_sq += (s ** 2).sum(0); n += s.shape[0]
    total_var = np.maximum(tot_sq / n - (tot_sum / n) ** 2, 1e-12)
    within_var = within_ss / n
    return float(np.mean(within_var / total_var))


def predictability_gap(seqs: list[np.ndarray], ks=(1, 2, 4, 8), max_pairs=20000,
                       seed=42) -> dict[int, float]:
    """tau_k = 1 - MSE(ridge x_t->x_{t+k}) / MSE(persistence), on a held-out split."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        Xs, Ys = [], []
        for s in seqs:
            if s.shape[0] <= k:
                continue
            Xs.append(s[:-k]); Ys.append(s[k:])
        if not Xs:
            out[k] = float("nan"); continue
        X = np.concatenate(Xs); Y = np.concatenate(Ys)
        if X.shape[0] > max_pairs:  # subsample for speed; unbiased
            idx = rng.choice(X.shape[0], max_pairs, replace=False)
            X, Y = X[idx], Y[idx]
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=seed)
        mse_pers = float(np.mean((Yte - Xte) ** 2))          # last-value carry-forward
        model = Ridge(alpha=1.0).fit(Xtr, Ytr)
        mse_model = float(np.mean((Yte - model.predict(Xte)) ** 2))
        out[k] = 1.0 - mse_model / max(mse_pers, 1e-12)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["seed_iv", "seed_iv_raw", "seed_v", "seed", "sleep_edf", "chbmit", "bci_iv_2a"])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--shuffle", action="store_true",
                    help="control: shuffle each sequence's window order (tau should collapse to ~0)")
    ap.add_argument("--out_csv", default="results/phase3/temporal_structure.csv")
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        if ds not in ARCH:
            LOG.warning("skip %s (not in ARCH)", ds); continue
        path = Path(ARCH[ds])
        if not path.exists():
            LOG.warning("skip %s (archive missing: %s)", ds, path); continue
        trials = load_de_archive(ARCH[ds])
        mean, std = fit_standardizer(trials)          # same corpus standardization the model sees
        seqs = [standardize(t.values, mean, std) for t in trials if t.values.shape[0] >= 2]
        if args.shuffle:
            rng = np.random.default_rng(0)
            seqs = [s[rng.permutation(s.shape[0])] for s in seqs]

        wvf = within_variance_fraction(seqs)
        taus = predictability_gap(seqs, tuple(args.ks))
        tau_mean = float(np.nanmean(list(taus.values())))
        lens = np.array([s.shape[0] for s in seqs])
        LOG.info("%-12s%s n_seq=%4d med_len=%5d  within_var_frac=%.4f  tau_1=%.4f  tau_mean=%.4f",
                 ds, " (SHUF)" if args.shuffle else "", len(seqs), int(np.median(lens)),
                 wvf, taus.get(1, float("nan")), tau_mean)
        rows.append((ds, args.shuffle, len(seqs), int(np.median(lens)), wvf, tau_mean, taus))

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["dataset", "shuffled", "n_sequences", "median_len",
                        "within_var_frac", "tau_mean"] + [f"tau_k{k}" for k in args.ks])
        for ds, shuf, n, ml, wvf, tm, taus in rows:
            w.writerow([ds, int(shuf), n, ml, f"{wvf:.6f}", f"{tm:.6f}"]
                       + [f"{taus.get(k, float('nan')):.6f}" for k in args.ks])
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
