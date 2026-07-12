#!/usr/bin/env python3
"""Build the CHB-MIT seizure DE archives (F17).

    python scripts/build_chbmit_dataset.py                 # all downloaded patients
    python scripts/build_chbmit_dataset.py --patients chb01 chb02 --seizure_files_only

Reads datasets/CHB-MIT/chb*/, computes per-2s-epoch DE + per-epoch binary
seizure labels, and writes chbmit_de.npz (label-less, for pretraining) +
chbmit_labels.npz (per-epoch labels + patient/file, for the evaluator).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.chbmit import build_chbmit_corpus, save_chbmit_archives
from physiofm.de import summarize_trials
from physiofm.chbmit import recordings_to_detrials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/CHB-MIT")
    ap.add_argument("--patients", nargs="*", default=None, help="e.g. chb01 chb02 (default: all)")
    ap.add_argument("--seizure_files_only", action="store_true",
                    help="keep only files containing >=1 seizure (smaller, less imbalanced)")
    ap.add_argument("--de_out", default="data/physiofm/de_features/chbmit_de.npz")
    ap.add_argument("--labels_out", default="data/physiofm/de_features/chbmit_labels.npz")
    args = ap.parse_args()

    recs = build_chbmit_corpus(args.root, patients=args.patients,
                               seizure_files_only=args.seizure_files_only)
    if not recs:
        raise SystemExit(f"no recordings built from {args.root} (downloaded yet?)")
    summ = summarize_trials(recordings_to_detrials(recs))
    n_epochs = sum(r.values.shape[0] for r in recs)
    n_seiz = sum(int((r.labels == 1).sum()) for r in recs)
    print("CHB-MIT DE:", summ)
    print(f"  recordings: {len(recs)} | patients: {sorted(set(r.patient for r in recs))}")
    print(f"  epochs: {n_epochs} | seizure epochs: {n_seiz} ({100*n_seiz/max(n_epochs,1):.2f}%)")
    save_chbmit_archives(recs, args.de_out, args.labels_out)
    print(f"wrote {args.de_out} + {args.labels_out}")


if __name__ == "__main__":
    main()
