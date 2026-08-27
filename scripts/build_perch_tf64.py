#!/usr/bin/env python3
"""Per-electrode tf64 archives — sliced from the structured archives (no recompute).

Each structured recording (T, C, 64) becomes C consecutive sequences (T, 1, 64); the
label companion repeats the recording's labels C times (keys "KEY#chc"), matching the
sleep_edf_raw_perch convention so `--merge_every C` regroups them at evaluation.
Used by the cross-corpus transfer experiment (EXP-0026): 1x64 tokens are
channel-count-agnostic, so full pretrained weights move between corpora.

    python scripts/build_perch_tf64.py --base sleep_edf_tf64 --labels sleep_edf_labels.npz
    python scripts/build_perch_tf64.py --base hmc_tf64 --labels hmc_labels.npz
    python scripts/build_perch_tf64.py --base p2018_tf64 --labels p2018_labels.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import DETrial, load_de_archive, save_de_archive
from physiofm.sleep_edf import load_sleep_labels
from physiofm.structured_data import ARCH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="ARCH key of the structured tf64 archive")
    ap.add_argument("--labels", required=True, help="label companion filename (in tf_features)")
    ap.add_argument("--out_dir", default="data/physiofm/tf_features")
    args = ap.parse_args()

    out = Path(args.out_dir)
    trials = load_de_archive(ARCH[args.base])
    labels_path = Path(args.labels) if "/" in args.labels else out / args.labels
    labels, subj, night, key = load_sleep_labels(labels_path)
    assert len(trials) == len(labels)

    p_trials, la, ps, pn, pk = [], [], [], [], []
    C = trials[0].values.shape[1]
    for i, (t, l) in enumerate(zip(trials, labels)):
        assert t.values.shape[1] == C, f"channel count varies at {i}"
        for c in range(C):
            p_trials.append(DETrial(f"{args.base}_perch", t.subject, t.session,
                                    i * C + c, None,
                                    np.ascontiguousarray(t.values[:, c:c + 1, :]),
                                    f"{t.source}#ch{c}"))
            la.append(l.astype(np.int64)); ps.append(int(subj[i])); pn.append(int(night[i]))
            pk.append(f"{key[i]}#ch{c}")

    save_de_archive(p_trials, out / f"{args.base}_perch.npz")
    arr = np.empty(len(la), dtype=object)
    for i, l in enumerate(la):
        arr[i] = l
    np.savez_compressed(out / f"{args.base}_perch_labels.npz", labels=arr,
                        subject=np.array(ps, dtype=np.int64),
                        night=np.array(pn, dtype=np.int64), key=np.array(pk))
    print(f"{args.base}: {len(trials)} recs x {C} ch -> {len(p_trials)} sequences; "
          f"wrote {args.base}_perch.npz + labels")


if __name__ == "__main__":
    main()
