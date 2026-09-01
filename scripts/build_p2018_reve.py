#!/usr/bin/env python3
"""Physio2018 epochs -> frozen REVE-base embeddings (EXP-0028 extension).

Mirrors physiofm/physio2018.py labels exactly; signal path per REVE conventions:
bandpass 0.1-75 Hz + 50 Hz notch at the native 200 Hz (no resampling needed),
microvolts/10 clip ±100, REVE forward, per-channel patch-mean -> (6, 512).
Positions: monopolar names (F3 F4 C3 C4 O1 O2), reference stripped (their HMC
convention for referenced derivations).

    python scripts/build_p2018_reve.py
Products: p2018_reve.npz, p2018_reve_labels.npz (data/physiofm/reve_features/)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.physio2018 import (list_records, _epoch_labels, DEFAULT_EEG_CHANNELS,
                                 STAGE_NAMES)
from physiofm.sleep_edf import SleepRecording, EPOCH_SEC, save_sleep_archives, recordings_to_detrials
from physiofm.de import DETrial, save_de_archive

SCALE, CLIP = 10.0, 100.0
POSITION_NAMES = [c.split("-")[0] for c in DEFAULT_EEG_CHANNELS]   # F3 F4 C3 C4 O1 O2


def one_record(rec_path, model, pos, device, batch=96):
    import mne
    import torch
    import wfdb

    rec = wfdb.rdrecord(str(rec_path), channel_names=list(DEFAULT_EEG_CHANNELS))
    order = [rec.sig_name.index(c) for c in DEFAULT_EEG_CHANNELS]
    eeg = np.ascontiguousarray(rec.p_signal[:, order].T)           # (6, T) physical (uV)
    sfreq = float(rec.fs)
    assert abs(sfreq - 200.0) < 1e-6
    eeg = mne.filter.filter_data(eeg.astype(np.float64), sfreq, 0.1, 75.0, verbose="ERROR")
    eeg = mne.filter.notch_filter(eeg, sfreq, 50.0, verbose="ERROR")
    eeg = np.clip(eeg / SCALE, -CLIP, CLIP).astype(np.float32)

    spe = int(EPOCH_SEC * sfreq)
    n = eeg.shape[1] // spe
    if n == 0:
        return None
    x = eeg[:, : n * spe].reshape(6, n, spe).transpose(1, 0, 2)
    outs = []
    with torch.no_grad():
        for b0 in range(0, n, batch):
            xb = torch.from_numpy(x[b0:b0 + batch]).to(device)
            pb = pos.unsqueeze(0).expand(xb.shape[0], -1, -1)
            outs.append(model(xb, pb).mean(dim=2).float().cpu().numpy())
    values = np.concatenate(outs, 0)

    ann = wfdb.rdann(str(rec_path), "arousal")
    labels = _epoch_labels(ann, n, sfreq)
    keep = labels >= 0
    if keep.sum() == 0:
        return None
    key = Path(str(rec_path)).name
    return SleepRecording(subject=int(key.replace("tr", "").replace("-", "")), night=1,
                         key=key, values=values[keep].astype(np.float16),
                         labels=labels[keep].astype(np.int64), source=str(rec_path))


def main() -> None:
    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval().to(device)
    with torch.no_grad():
        pos = pos_bank(POSITION_NAMES).to(device)
    assert pos.shape == (6, 3)

    t0 = time.time()
    recs = []
    paths = list_records("datasets/P2018")
    for i, rp in enumerate(paths):
        r = one_record(rp, model, pos, device)
        if r is not None:
            recs.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
    recs.sort(key=lambda r: r.key)
    n_lab = sum(int((r.labels >= 0).sum()) for r in recs)
    print(f"extracted {len(recs)} records, {n_lab} labelled epochs in {time.time()-t0:.0f}s")
    assert len(recs) == 994 and n_lab == 892200, f"corpus mismatch: {len(recs)} recs / {n_lab} epochs"

    out = Path("data/physiofm/reve_features"); out.mkdir(parents=True, exist_ok=True)
    save_sleep_archives(recs, out / "p2018_reve.npz", out / "p2018_reve_labels.npz")
    print("wrote p2018_reve archives")


if __name__ == "__main__":
    main()
