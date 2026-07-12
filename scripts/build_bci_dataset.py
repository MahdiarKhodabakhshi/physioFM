#!/usr/bin/env python3
"""Build the BCI-IV-2a (motor imagery) DE archive.

    python scripts/build_bci_dataset.py            # -> data/physiofm/de_features/bci_iv_2a_de.npz

Reads datasets/BCI-IV-2a/A0{1..9}{T,E}.mat, extracts per-trial DE sequences over
the motor-imagery window, and writes the canonical DE archive (variable-length
trials, per-trial 4-way labels, session 1=T / 2=E for the holdout eval).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.bci_iv_2a import load_bci_trials
from physiofm.de import save_de_archive, summarize_trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/BCI-IV-2a")
    ap.add_argument("--out", default="data/physiofm/de_features/bci_iv_2a_de.npz")
    args = ap.parse_args()

    trials = load_bci_trials(args.root)
    summ = summarize_trials(trials)
    print("BCI-IV-2a DE:", summ)
    labels = np.array([t.label for t in trials])
    sess = np.array([t.session for t in trials])
    print(f"  class balance: {np.bincount(labels)}  (0..3 = left/right/feet/tongue)")
    print(f"  session split: T={int((sess==1).sum())}  E={int((sess==2).sum())}")
    save_de_archive(trials, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
