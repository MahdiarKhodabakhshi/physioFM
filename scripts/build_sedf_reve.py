#!/usr/bin/env python3
"""Sleep-EDF-78 epochs -> frozen REVE-base embeddings (EXP-0028 extension).

Reuses the canonical SEDF corpus builder (same recordings, ±30-min wake trim, label
alignment) with a REVE feature_fn: per 30-s epoch — 0.1 Hz high-pass (the 75 Hz
low-pass edge of REVE's band is a no-op at 100 Hz native; 50 Hz notch skipped, it
sits exactly at Nyquist), upsample 100->200 Hz (resample_poly), microvolts/10 clip
±100 (REVE's sleep convention), REVE forward, per-channel patch-mean -> (2, 512).
Positions: bipolar derivations use the electrode-midpoint convention from REVE's own
position_utils ((Fpz+Cz)/2, (Pz+Oz)/2).

    python scripts/build_sedf_reve.py
Products (data/physiofm/reve_features/): sleep_edf_reve.npz, sleep_edf_reve_labels.npz
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.sleep_edf import build_sleep_corpus, save_sleep_archives, recordings_to_detrials
from physiofm.de import summarize_trials

ROOT = "physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
SCALE, CLIP = 10.0, 100.0
ELECTRODE_PAIRS = (("Fpz", "Cz"), ("Pz", "Oz"))   # EEG Fpz-Cz, EEG Pz-Oz


def make_feature_fn(model, pos, device, batch=96):
    import torch
    from scipy.signal import resample_poly

    def fn(eeg, sfreq, window_sec, step_sec):
        assert abs(sfreq - 100.0) < 1e-6, f"expected 100 Hz SEDF, got {sfreq}"
        import mne
        eeg = mne.filter.filter_data(eeg.astype(np.float64), sfreq, 0.1, None, verbose="ERROR")
        eeg = resample_poly(eeg, 2, 1, axis=-1)                    # -> 200 Hz
        eeg = np.clip(eeg / SCALE, -CLIP, CLIP).astype(np.float32)
        spe = int(window_sec * 200)
        n = eeg.shape[1] // spe
        if n == 0:
            return np.zeros((0, eeg.shape[0], 512), dtype=np.float32)
        x = eeg[:, : n * spe].reshape(eeg.shape[0], n, spe).transpose(1, 0, 2)
        outs = []
        with torch.no_grad():
            for b0 in range(0, n, batch):
                xb = torch.from_numpy(x[b0:b0 + batch]).to(device)
                pb = pos.unsqueeze(0).expand(xb.shape[0], -1, -1)
                outs.append(model(xb, pb).mean(dim=2).float().cpu().numpy())
        return np.concatenate(outs, 0)                              # (n, C, 512)
    return fn


def main() -> None:
    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval().to(device)
    with torch.no_grad():
        mids = [pos_bank(list(pair)).mean(0) for pair in ELECTRODE_PAIRS]
    pos = torch.stack(mids).to(device)                              # (2, 3)
    print("positions", tuple(pos.shape))

    t0 = time.time()
    recs = build_sleep_corpus(ROOT, feature_fn=make_feature_fn(model, pos, device))
    print(f"built {len(recs)} recordings in {time.time()-t0:.0f}s; token {recs[0].values.shape[1:]}")
    print(summarize_trials(recordings_to_detrials(recs)))

    # label-alignment check against the canonical DE companion (same builder, same trim)
    ref = np.load("data/physiofm/de_features/sleep_edf_labels.npz", allow_pickle=True)
    ref_labels = [np.asarray(a) for a in ref["labels"]]
    assert len(ref_labels) == len(recs)
    bad = [i for i, (r, a) in enumerate(zip(recs, ref_labels)) if not np.array_equal(r.labels, a)]
    assert not bad, f"label misalignment on {len(bad)} recordings: {bad[:5]}"
    print("labels identical to canonical sleep_edf_labels.npz")

    out = Path("data/physiofm/reve_features"); out.mkdir(parents=True, exist_ok=True)
    save_sleep_archives(recs, out / "sleep_edf_reve.npz", out / "sleep_edf_reve_labels.npz")
    print("wrote sleep_edf_reve archives")


if __name__ == "__main__":
    main()
