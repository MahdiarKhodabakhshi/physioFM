#!/usr/bin/env python3
"""Gate 2 (docs/NEXT_PHASE_PLAN.md): build RAW-EEG structured-token archives.

Same recordings / epoching / wake-trimming / order as the DE archives; the per-epoch feature is
the raw signal itself, re-cut into structured tokens = all channels x 200 ms.

    python scripts/build_raw_dataset.py --task sleep                 # (tokens, 2, 20), 150 tok/epoch
    python scripts/build_raw_dataset.py --task sleep --per_channel   # per-electrode ablation
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import DETrial, save_de_archive
from physiofm.raw_eeg import epochs_to_channel_tokens, epochs_to_tokens, raw_token_feature_fn


def _check_alignment(new_labels, ref_labels_path, new_keys):
    with np.load(ref_labels_path, allow_pickle=True) as z:
        ref = [np.asarray(a, dtype=np.int64) for a in z["labels"]]
        ref_keys = list(z["key"])
    ok = len(ref) == len(new_labels) and all(np.array_equal(a, b) for a, b in zip(ref, new_labels)) \
        and list(ref_keys) == list(new_keys)
    if not ok:
        raise SystemExit(f"ALIGNMENT FAILED vs {ref_labels_path}")
    print(f"alignment OK: {len(ref)} recordings, labels identical to {ref_labels_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sleep"], default="sleep")
    ap.add_argument("--token_sec", type=float, default=0.2)
    ap.add_argument("--per_channel", action="store_true")
    ap.add_argument("--root", default="physionet.org/files/sleep-edfx/1.0.0/sleep-cassette")
    ap.add_argument("--out_dir", default="data/physiofm/raw_tokens")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from physiofm.sleep_edf import build_sleep_corpus

    t0 = time.time()
    recs = build_sleep_corpus(args.root, feature_fn=raw_token_feature_fn(args.token_sec), limit=args.limit)
    print(f"loaded {len(recs)} recordings in {time.time() - t0:.0f}s; epoch shape {recs[0].values.shape[1:]}")
    _check_alignment([r.labels for r in recs], "data/physiofm/de_features/sleep_edf_labels.npz", [r.key for r in recs])
    sfreq_tok = int(round(args.token_sec * 100))  # Sleep-EDF EEG is 100 Hz
    trials, label_arrays, subj, night, key = [], [], [], [], []
    for i, r in enumerate(recs):
        if args.per_channel:
            for c, tok in enumerate(epochs_to_channel_tokens(r.values, sfreq_tok)):
                trials.append(DETrial("sleep_edf_raw_perch", r.subject, r.night, i * r.values.shape[1] + c, None, tok, f"{r.source}#ch{c}"))
                label_arrays.append(r.labels); subj.append(r.subject); night.append(r.night); key.append(f"{r.key}#ch{c}")
        else:
            trials.append(DETrial("sleep_edf_raw", r.subject, r.night, i, None, epochs_to_tokens(r.values, sfreq_tok), r.source))
            label_arrays.append(r.labels); subj.append(r.subject); night.append(r.night); key.append(r.key)
    n_tok = sum(t.values.shape[0] for t in trials)
    print(f"{len(trials)} sequences, {n_tok} tokens, token shape {trials[0].values.shape[1:]}, "
          f"tokens/epoch={trials[0].values.shape[0] // len(label_arrays[0])}")
    suffix = "_perch" if args.per_channel else ""
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    save_de_archive(trials, out / f"sleep_edf_raw{int(args.token_sec * 1000)}ms{suffix}.npz", dtype=np.float16)
    la = np.empty(len(label_arrays), dtype=object)
    for i, l in enumerate(label_arrays):
        la[i] = l.astype(np.int64)
    np.savez_compressed(out / f"sleep_edf_raw{int(args.token_sec * 1000)}ms{suffix}_labels.npz", labels=la,
                        subject=np.array(subj, dtype=np.int64), night=np.array(night, dtype=np.int64), key=np.array(key))
    print("wrote", out, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
