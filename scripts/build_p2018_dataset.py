#!/usr/bin/env python3
"""Build the Physio2018 tf64 archives (third sleep corpus; docs/SLEEP_DATASET_CANDIDATES.md).

Products:
  data/physiofm/tf_features/p2018_tf64.npz    all labeled training records (6 ch x 64)
  data/physiofm/tf_features/p2018_labels.npz  per-epoch label companion

    python scripts/build_p2018_dataset.py [--root datasets/P2018] [--workers 8]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.spectral import compute_log_spectrogram, DEFAULT_N_BINS
from physiofm.physio2018 import load_recording, list_records
from physiofm.sleep_edf import save_sleep_archives, recordings_to_detrials
from physiofm.de import summarize_trials


def tf_feature_fn(n_bins: int):
    def fn(eeg, sfreq, window_sec, step_sec):
        return compute_log_spectrogram(eeg, sfreq, window_sec, step_sec, n_bins=n_bins)
    return fn


def _one(args):
    rec_path, n_bins = args
    try:
        return load_recording(rec_path, feature_fn=tf_feature_fn(n_bins))
    except Exception as e:              # surface the record name with the failure
        raise RuntimeError(f"{rec_path}: {e}") from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/P2018")
    ap.add_argument("--n_bins", type=int, default=DEFAULT_N_BINS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out_dir", default="data/physiofm/tf_features")
    args = ap.parse_args()

    t0 = time.time()
    recs_paths = list_records(args.root)
    print(f"{len(recs_paths)} records under {args.root}")
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            recs = pool.map(_one, [(str(p), args.n_bins) for p in recs_paths], chunksize=4)
    else:
        recs = [_one((str(p), args.n_bins)) for p in recs_paths]
    recs = [r for r in recs if r is not None]
    recs.sort(key=lambda r: r.key)
    print(f"built {len(recs)} recordings in {time.time() - t0:.0f}s")

    shapes = {r.values.shape[1:] for r in recs}
    assert len(shapes) == 1, f"inconsistent token shapes: {shapes}"
    n_lab = sum(int((r.labels >= 0).sum()) for r in recs)
    per_class = np.bincount(np.concatenate([r.labels[r.labels >= 0] for r in recs]), minlength=5)
    print(f"token shape {shapes.pop()}, {n_lab} labelled epochs, W/N1/N2/N3/R = {per_class.tolist()}")
    # Protocol check: SleePyCo Table 2 (arXiv:2209.09452) counts from the same 994 records:
    # W 157,945 / N1 136,978 / N2 377,870 / N3 102,592 / REM 116,877 = 892,262.
    published = [157945, 136978, 377870, 102592, 116877]
    if per_class.tolist() != published:
        diff = [int(a - b) for a, b in zip(per_class.tolist(), published)]
        print(f"WARNING: per-class counts differ from SleePyCo Table 2 by {diff} "
              f"(total {n_lab} vs 892262) — investigate before comparing rows")
    else:
        print("per-class epoch counts MATCH SleePyCo Table 2 exactly (892,262)")
    print(summarize_trials(recordings_to_detrials(recs)))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    save_sleep_archives(recs, out / "p2018_tf64.npz", out / "p2018_labels.npz")
    print(f"wrote {out}/p2018_tf64.npz + p2018_labels.npz ({len(recs)} recs)")


if __name__ == "__main__":
    main()
