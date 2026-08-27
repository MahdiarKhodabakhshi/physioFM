#!/usr/bin/env python3
"""Build the HMC tf64 archives (external sleep validation; docs/SLEEP_DATASET_CANDIDATES.md).

Products (mirrors the sleep_edf tf64 layout):
  data/physiofm/tf_features/hmc_tf64.npz           all 151 recordings (evaluators)
  data/physiofm/tf_features/hmc_labels.npz         per-epoch label companion
  data/physiofm/tf_features/hmc_tf64_pretrain.npz  subjects SN001–SN125 ONLY
                                                   (train+val; the pretraining corpus —
                                                    test subjects never seen in pretraining
                                                    or in the standardizer)

    python scripts/build_hmc_dataset.py [--root datasets/HMC] [--workers 6]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.spectral import compute_log_spectrogram, DEFAULT_N_BINS
from physiofm.hmc import (build_corpus, load_recording, pair_recordings,
                          split_masks, PRETRAIN_MAX_SUBJECT)
from physiofm.sleep_edf import save_sleep_archives, recordings_to_detrials
from physiofm.de import save_de_archive, summarize_trials


def tf_feature_fn(n_bins: int):
    def fn(eeg, sfreq, window_sec, step_sec):
        return compute_log_spectrogram(eeg, sfreq, window_sec, step_sec, n_bins=n_bins)
    return fn


def _one(args):
    psg, hyp, n_bins = args
    return load_recording(psg, hyp, feature_fn=tf_feature_fn(n_bins))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/HMC")
    ap.add_argument("--n_bins", type=int, default=DEFAULT_N_BINS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out_dir", default="data/physiofm/tf_features")
    args = ap.parse_args()

    t0 = time.time()
    pairs = pair_recordings(args.root)
    print(f"{len(pairs)} PSG/scoring pairs under {args.root}")
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            recs = pool.map(_one, [(p, h, args.n_bins) for p, h in pairs])
    else:
        recs = [_one((p, h, args.n_bins)) for p, h in pairs]
    recs = [r for r in recs if r is not None]
    recs.sort(key=lambda r: r.subject)
    print(f"built {len(recs)} recordings in {time.time() - t0:.0f}s")

    shapes = {r.values.shape[1:] for r in recs}
    assert len(shapes) == 1, f"inconsistent token shapes: {shapes}"
    n_lab = sum(int((r.labels >= 0).sum()) for r in recs)
    per_class = np.bincount(
        np.concatenate([r.labels[r.labels >= 0] for r in recs]), minlength=5)
    print(f"token shape {shapes.pop()}, {n_lab} labelled epochs, "
          f"W/N1/N2/N3/R = {per_class.tolist()}")
    # Hard protocol check: the NeuroLM ladder's split sizes (NeuroLM Table 1, verified
    # against the local corpus by the pre-launch review). All scored epochs, no trim.
    import numpy as _np
    subj = _np.array([r.subject for r in recs])
    tr_m, va_m, te_m = split_masks(subj)
    counts = tuple(int(sum((r.labels >= 0).sum() for r, m in zip(recs, mm) if m))
                   for mm in (tr_m, va_m, te_m))
    print(f"split recordings {tr_m.sum()}/{va_m.sum()}/{te_m.sum()}, epochs {counts}")
    assert (tr_m.sum(), va_m.sum(), te_m.sum()) == (100, 25, 26), "recording split != 100/25/26"
    assert counts == (91248, 22124, 23871), f"epoch counts {counts} != published (91248, 22124, 23871)"
    print(summarize_trials(recordings_to_detrials(recs)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_sleep_archives(recs, out / "hmc_tf64.npz", out / "hmc_labels.npz")

    pre = [r for r in recs if r.subject <= PRETRAIN_MAX_SUBJECT]
    save_de_archive(recordings_to_detrials(pre), out / "hmc_tf64_pretrain.npz")
    print(f"wrote {out}/hmc_tf64.npz ({len(recs)} recs), "
          f"hmc_labels.npz, hmc_tf64_pretrain.npz ({len(pre)} recs, subj<= {PRETRAIN_MAX_SUBJECT})")


if __name__ == "__main__":
    main()
