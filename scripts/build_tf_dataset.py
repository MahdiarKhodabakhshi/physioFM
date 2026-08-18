#!/usr/bin/env python3
"""Gate 0 (docs/NEXT_PHASE_PLAN.md): build rich time–frequency archives.

Same recordings, same epoching, same wake-trimming / channel picks / recording order as
the DE archives — only the per-epoch feature changes: 64 log-spaced log-power bins per
channel (physiofm/spectral.py) instead of the 5 DE bands. The script asserts that the
per-epoch labels it derives are IDENTICAL to the existing DE label companion, so the two
archives are interchangeable in every evaluator (`--arch_key sleep_edf_tf64`).

    python scripts/build_tf_dataset.py --task sleep
    python scripts/build_tf_dataset.py --task chbmit [--workers 6]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.spectral import compute_log_spectrogram, DEFAULT_N_BINS


def tf_feature_fn(n_bins: int):
    def fn(eeg, sfreq, window_sec, step_sec):
        return compute_log_spectrogram(eeg, sfreq, window_sec, step_sec, n_bins=n_bins)
    return fn


def _check_alignment(new_labels, ref_labels_path, ref_key_field, new_keys):
    with np.load(ref_labels_path, allow_pickle=True) as z:
        ref = [np.asarray(a, dtype=np.int64) for a in z["labels"]]
        ref_keys = list(z[ref_key_field])
    if len(ref) != len(new_labels):
        raise SystemExit(f"ALIGNMENT FAILED: {len(new_labels)} recordings vs {len(ref)} in {ref_labels_path}")
    bad = [i for i, (a, b) in enumerate(zip(ref, new_labels)) if a.shape != b.shape or not np.array_equal(a, b)]
    if bad or list(ref_keys) != list(new_keys):
        raise SystemExit(f"ALIGNMENT FAILED on {len(bad)} recordings (first: {bad[:5]}); keys equal={list(ref_keys) == list(new_keys)}")
    print(f"alignment OK: {len(ref)} recordings, labels identical to {ref_labels_path}")


def build_sleep(args):
    from physiofm.sleep_edf import build_sleep_corpus, save_sleep_archives, recordings_to_detrials
    from physiofm.de import summarize_trials

    t0 = time.time()
    recs = build_sleep_corpus(args.root, feature_fn=tf_feature_fn(args.n_bins))
    print(f"built {len(recs)} recordings in {time.time() - t0:.0f}s")
    print(summarize_trials(recordings_to_detrials(recs)))
    if args.check_against:
        _check_alignment([r.labels for r in recs], args.check_against, "key", [r.key for r in recs])
    save_sleep_archives(recs, args.out, args.labels_out)
    print("wrote", args.out, args.labels_out)


def _chbmit_one(pdir_str, n_bins, channels):
    """Worker: all EDFs of one patient dir -> list of SeizureRecording (picklable)."""
    from physiofm.chbmit import parse_summary, load_recording

    pdir = Path(pdir_str)
    summ = pdir / f"{pdir.name}-summary.txt"
    if not summ.exists():
        return []
    ann = parse_summary(summ)
    out = []
    for edf in sorted(pdir.glob("*.edf")):
        rec = load_recording(edf, ann.get(edf.name, []), channels=channels, feature_fn=tf_feature_fn(n_bins))
        if rec is not None:
            out.append(rec)
    return out


def build_chbmit(args):
    from concurrent.futures import ProcessPoolExecutor
    from physiofm.chbmit import CORE_CHANNELS, save_chbmit_archives, recordings_to_detrials
    from physiofm.de import summarize_trials

    root = Path(args.root)
    pdirs = sorted(str(p) for p in root.glob("chb*") if p.is_dir())
    t0 = time.time()
    recs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, part in enumerate(ex.map(_chbmit_one, pdirs, [args.n_bins] * len(pdirs), [CORE_CHANNELS] * len(pdirs))):
            recs.extend(part)
            print(f"  {Path(pdirs[i]).name}: {len(part)} recordings (total {len(recs)}, {time.time() - t0:.0f}s)", flush=True)
    print(f"built {len(recs)} recordings in {time.time() - t0:.0f}s")
    print(summarize_trials(recordings_to_detrials(recs)))
    if args.check_against:
        _check_alignment([r.labels for r in recs], args.check_against, "key", [r.key for r in recs])
    save_chbmit_archives(recs, args.out, args.labels_out)
    print("wrote", args.out, args.labels_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sleep", "chbmit"], required=True)
    ap.add_argument("--n_bins", type=int, default=DEFAULT_N_BINS)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels_out", default=None)
    ap.add_argument("--check_against", default=None, help="DE label companion to assert alignment with")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.task == "sleep":
        args.root = args.root or "physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
        args.out = args.out or f"data/physiofm/tf_features/sleep_edf_tf{args.n_bins}.npz"
        args.labels_out = args.labels_out or f"data/physiofm/tf_features/sleep_edf_tf{args.n_bins}_labels.npz"
        if args.check_against is None:
            args.check_against = "data/physiofm/de_features/sleep_edf_labels.npz"
        build_sleep(args)
    else:
        args.root = args.root or "datasets/CHB-MIT"
        args.out = args.out or f"data/physiofm/tf_features/chbmit_tf{args.n_bins}.npz"
        args.labels_out = args.labels_out or f"data/physiofm/tf_features/chbmit_tf{args.n_bins}_labels.npz"
        if args.check_against is None:
            args.check_against = "data/physiofm/de_features/chbmit_labels.npz"
        build_chbmit(args)


if __name__ == "__main__":
    main()
