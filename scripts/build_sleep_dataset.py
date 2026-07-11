#!/usr/bin/env python3
"""Build the Sleep-EDF DE corpus for F13 (Phase 3).

Reads Sleep-EDF Cassette PSG/Hypnogram pairs from ``datasets/SLEEP-EDF/`` and
writes the canonical DE archive + per-epoch label companion under
``data/physiofm/de_features/``:

    sleep_edf_de.npz       (one DETrial per recording, label=None -> pretraining)
    sleep_edf_labels.npz   (per-epoch W/N1/N2/N3/REM labels + subject/night)

Usage:
    python scripts/build_sleep_dataset.py
    python scripts/build_sleep_dataset.py --limit 4 --no_trim     # quick smoke
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.sleep_edf import (
    DEFAULT_EEG_CHANNELS,
    LABEL_NAMES,
    build_sleep_corpus,
    save_sleep_archives,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/SLEEP-EDF")
    ap.add_argument("--out_dir", default="data/physiofm/de_features")
    ap.add_argument("--channels", nargs="+", default=list(DEFAULT_EEG_CHANNELS))
    ap.add_argument("--trim_wake_min", type=float, default=30.0,
                    help="keep this many minutes of wake around sleep (0 -> keep all)")
    ap.add_argument("--no_trim", action="store_true", help="keep all wake epochs")
    ap.add_argument("--limit", type=int, default=None, help="cap #recordings (smoke test)")
    args = ap.parse_args()

    trim = None if args.no_trim else (None if args.trim_wake_min <= 0 else args.trim_wake_min)
    recs = build_sleep_corpus(
        args.root, channels=tuple(args.channels), trim_wake_min=trim, limit=args.limit
    )
    if not recs:
        raise SystemExit(f"No recordings built from {args.root} (PSG/Hypnogram pairs found?)")

    n_epochs = sum(r.values.shape[0] for r in recs)
    n_ch, n_band = recs[0].values.shape[1], recs[0].values.shape[2]
    subjects = sorted({r.subject for r in recs})
    dist = Counter()
    for r in recs:
        dist.update(r.labels.tolist())

    out_dir = Path(args.out_dir)
    de_path = out_dir / "sleep_edf_de.npz"
    labels_path = out_dir / "sleep_edf_labels.npz"
    save_sleep_archives(recs, de_path, labels_path)

    print(f"recordings: {len(recs)}  subjects: {len(subjects)}  channels: {args.channels}")
    print(f"epochs: {n_epochs}  n_cb: {n_ch * n_band} ({n_ch}ch x {n_band}band)")
    print("stage distribution:", {LABEL_NAMES[k]: dist[k] for k in range(len(LABEL_NAMES))})
    print(f"wrote {de_path}")
    print(f"wrote {labels_path}")


if __name__ == "__main__":
    main()
