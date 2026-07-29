#!/usr/bin/env python3
"""Build SEIZURE PREDICTION labels (pre-ictal vs interictal) from the existing CHB-MIT DE.

Why this task. Every failure we diagnosed came from the pretext (forecast the future) being
misaligned with the downstream task (classify the present) — see EXP-0017. Seizure PREDICTION
is the one EEG task where the downstream task *is* forecasting: decide, from EEG now, whether
a seizure is coming. Predictability and discriminability coincide by construction, so our
analysis makes a falsifiable prediction that PC pretraining should help here — even
fine-tuned, where it collapsed on detection.

Standard clinical protocol:
  * SPH (seizure prediction horizon) — the last `sph_min` before onset is EXCLUDED. A warning
    that arrives 30 s ahead is useless; the model must predict with lead time.
  * SOP (seizure occurrence period) — the `sop_min` before the SPH is PRE-ICTAL (label 1).
  * Ictal + `postictal_min` after seizure end are EXCLUDED (not the prediction problem).
  * INTERICTAL (label 0) is taken only from recordings containing NO seizure at all.

Reuses `chbmit_de.npz` unchanged — only the labels differ — so no DE recomputation.

    python scripts/build_chbmit_prediction.py --sop_min 30 --sph_min 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.chbmit import EPOCH_SEC, parse_summary
from physiofm.de import load_de_archive
from physiofm.chbmit import load_chbmit_labels

EXCLUDE = -1


def preictal_labels(intervals, n_epochs, sop_min, sph_min, postictal_min):
    """Per-epoch labels: 1=pre-ictal, 0=candidate interictal, -1=excluded."""
    lab = np.zeros(n_epochs, dtype=np.int64)
    if not intervals:
        return lab  # seizure-free recording: all candidate interictal
    sop = sop_min * 60.0 / EPOCH_SEC
    sph = sph_min * 60.0 / EPOCH_SEC
    post = postictal_min * 60.0 / EPOCH_SEC
    for start, end in intervals:
        s = start / EPOCH_SEC
        e = end / EPOCH_SEC
        lo = int(max(0, s - sph - sop))
        hi = int(max(0, s - sph))
        lab[lo:hi] = 1                                   # pre-ictal
        lab[int(max(0, s - sph)):int(min(n_epochs, e + post))] = EXCLUDE  # SPH + ictal + post
    # any remaining 0s inside a seizure-bearing recording are ambiguous -> exclude
    lab[lab == 0] = EXCLUDE
    return lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/CHB-MIT")
    ap.add_argument("--de", default="data/physiofm/de_features/chbmit_de.npz")
    ap.add_argument("--det_labels", default="data/physiofm/de_features/chbmit_labels.npz")
    ap.add_argument("--out", default="data/physiofm/de_features/chbmit_pred_labels.npz")
    ap.add_argument("--sop_min", type=float, default=30.0, help="pre-ictal window length (min)")
    ap.add_argument("--sph_min", type=float, default=5.0, help="excluded lead-time gap (min)")
    ap.add_argument("--postictal_min", type=float, default=30.0)
    args = ap.parse_args()

    trials = load_de_archive(args.de)
    _, patient, file_idx, keys = load_chbmit_labels(args.det_labels)
    if len(trials) != len(keys):
        raise SystemExit(f"archive/label misalignment: {len(trials)} vs {len(keys)}")

    ann = {}
    for d in sorted(Path(args.root).glob("chb*")):
        s = d / f"{d.name}-summary.txt"
        if s.exists():
            ann.update(parse_summary(s))

    out, n_pre, n_int, n_exc = [], 0, 0, 0
    for i, t in enumerate(trials):
        key = str(keys[i])
        intervals = ann.get(key + ".edf", [])
        lab = preictal_labels(intervals, t.values.shape[0], args.sop_min,
                              args.sph_min, args.postictal_min)
        out.append(lab)
        n_pre += int((lab == 1).sum()); n_int += int((lab == 0).sum()); n_exc += int((lab == EXCLUDE).sum())

    tot = n_pre + n_int
    print(f"CHB-MIT seizure PREDICTION (SOP={args.sop_min}min, SPH={args.sph_min}min)")
    print(f"  recordings: {len(out)}  patients: {len(set(patient.tolist()))}")
    print(f"  pre-ictal epochs : {n_pre:>8}  ({100*n_pre/max(tot,1):.1f}% of usable)")
    print(f"  interictal epochs: {n_int:>8}  ({100*n_int/max(tot,1):.1f}%)")
    print(f"  excluded         : {n_exc:>8}")
    pat_pre = {}
    for i, lab in enumerate(out):
        if (lab == 1).any():
            pat_pre[int(patient[i])] = pat_pre.get(int(patient[i]), 0) + 1
    print(f"  patients with pre-ictal data: {len(pat_pre)}")

    arr = np.empty(len(out), dtype=object)
    for i, l in enumerate(out):
        arr[i] = l
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, labels=arr, patient=patient, file_idx=file_idx, key=keys)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
