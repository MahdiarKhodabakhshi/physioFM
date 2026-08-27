#!/usr/bin/env python3
"""Per-fold pretraining corpora + single-channel variant for Physio2018.

For each SleePyCo fold k: pretraining archive = all recordings EXCEPT the test block
(train 745 + val 50), so pretraining and the standardizer never see test subjects —
same discipline as HMC. Also writes C3-M2-only slices (channel index 2 of the 6-ch
archive) for the strictly-ladder-comparable single-channel arm.

    python scripts/prepare_p2018_folds.py
Products (data/physiofm/tf_features/):
  p2018_tf64_c3.npz                    all recs, 1x64 tokens (slice, no recompute)
  p2018_pretrain_fold{1..5}.npz        6x64, non-test recs of fold k
  p2018_c3_pretrain_fold{1..5}.npz     1x64 variant
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import DETrial, load_de_archive, save_de_archive
from physiofm.physio2018 import load_folds
from physiofm.structured_data import ARCH

C3_IDX = 2  # DEFAULT_EEG_CHANNELS = (F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1)


def slice_c3(t: DETrial) -> DETrial:
    return DETrial(dataset=t.dataset, subject=t.subject, session=t.session,
                   trial=t.trial, label=t.label,
                   values=np.ascontiguousarray(t.values[:, C3_IDX:C3_IDX + 1, :]),
                   source=t.source)


def main() -> None:
    out = Path("data/physiofm/tf_features")
    trials = load_de_archive(ARCH["p2018_tf64"])
    assert len(trials) == 994, f"expected 994 recordings, got {len(trials)}"
    folds = load_folds()

    c3 = [slice_c3(t) for t in trials]
    save_de_archive(c3, out / "p2018_tf64_c3.npz")
    print(f"wrote p2018_tf64_c3.npz ({len(c3)} recs, 1x64)")

    for k, f in enumerate(folds, start=1):
        test = set(f["test"].tolist())
        keep = [i for i in range(len(trials)) if i not in test]
        save_de_archive([trials[i] for i in keep], out / f"p2018_pretrain_fold{k}.npz")
        save_de_archive([c3[i] for i in keep], out / f"p2018_c3_pretrain_fold{k}.npz")
        print(f"fold{k}: pretrain corpus {len(keep)} recs (test excluded: {len(test)})")


if __name__ == "__main__":
    main()
