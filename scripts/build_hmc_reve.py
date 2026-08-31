#!/usr/bin/env python3
"""HMC epochs -> frozen REVE-base embeddings (EXP-0028: stacking our sequence model
on the strongest EEG foundation model).

Per 30-s epoch: preprocess exactly as REVE's own HMC pipeline (their repo,
preprocessing/preprocessing_hmc.py — bandpass 0.1-75 Hz, 50 Hz notch, resample
200 Hz, microvolts / 100, monopolar position names F4/C4/O2/C3) -> REVE-base
forward -> per-channel mean over the ~33 one-second patches -> (4, 512) token.
Stored in the standard DETrial container so the ENTIRE existing ladder
(phase2_pretrain / phase2_hmc_finetune) runs unchanged with arch_key hmc_reve.

Label pipeline identical to build_hmc_dataset.py (all scored epochs, no trim);
the build asserts the split epoch counts still equal NeuroLM Table 1 exactly.

    HF_HOME=... python scripts/build_hmc_reve.py [--model brain-bzh/reve-base]
Products (data/physiofm/reve_features/):
  hmc_reve.npz, hmc_reve_labels.npz, hmc_reve_pretrain.npz (subjects <= SN127)
(--model brain-bzh/reve-large writes hmc_revelarge*.npz; token (4, 1216))
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.hmc import (_epoch_labels, pair_recordings, split_masks,
                          DEFAULT_EEG_CHANNELS, PRETRAIN_MAX_SUBJECT)
from physiofm.sleep_edf import SleepRecording, EPOCH_SEC, save_sleep_archives, recordings_to_detrials
from physiofm.de import save_de_archive

# REVE conventions (their preprocessing_hmc.py + dataloaders.py)
REVE_SFREQ = 200.0
BANDPASS = (0.1, 75.0)
NOTCH = 50.0
SCALE = 10.0                       # microvolts / 10  (their hmc.yaml: scale_factor 10)
CLIP = 100.0                       # post-scale clip (their dataloader: clip 100)
POSITION_NAMES = [c.split(" ")[-1].split("-")[0] for c in DEFAULT_EEG_CHANNELS]  # F4 C4 O2 C3


def load_reve(model_name: str, device: str):
    import torch
    from transformers import AutoModel

    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval().to(device)
    with torch.no_grad():
        pos = pos_bank(POSITION_NAMES)            # (4, 3)
    return model, pos.to(device)


def one_recording(psg_path, hyp_path, model, pos, device, batch=96):
    import mne
    import torch

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    raw.pick(list(DEFAULT_EEG_CHANNELS))
    raw.filter(*BANDPASS, verbose="ERROR")
    raw.notch_filter(NOTCH, verbose="ERROR")
    raw.resample(REVE_SFREQ, verbose="ERROR")
    eeg = np.clip(raw.get_data(units="uV") / SCALE, -CLIP, CLIP)  # (4, T) @ 200 Hz

    spe = int(EPOCH_SEC * REVE_SFREQ)             # 6000
    n_epochs = eeg.shape[1] // spe
    if n_epochs == 0:
        return None
    x = eeg[:, : n_epochs * spe].reshape(len(DEFAULT_EEG_CHANNELS), n_epochs, spe)
    x = np.transpose(x, (1, 0, 2)).astype(np.float32)   # (n_epochs, 4, 6000)

    feats = []
    with torch.no_grad():
        for b0 in range(0, n_epochs, batch):
            xb = torch.from_numpy(x[b0:b0 + batch]).to(device)
            pb = pos.unsqueeze(0).expand(xb.shape[0], -1, -1)
            out = model(xb, pb)                    # (B, 4, num_patches, 512)
            feats.append(out.mean(dim=2).float().cpu().numpy())  # (B, 4, 512)
    values = np.concatenate(feats, 0)              # (n_epochs, 4, 512)

    import mne as _m
    ann = _m.read_annotations(hyp_path)
    labels = _epoch_labels(ann, n_epochs)
    keep = labels >= 0                             # all scored epochs, no trim
    if keep.sum() == 0:
        return None
    key = Path(psg_path).stem
    return SleepRecording(subject=int(key[2:]), night=1, key=key,
                          values=values[keep].astype(np.float16),
                          labels=labels[keep].astype(np.int64), source=str(psg_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/HMC")
    ap.add_argument("--model", default="brain-bzh/reve-base")
    ap.add_argument("--out_dir", default="data/physiofm/reve_features")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--stem", default=None, help="archive basename (default: hmc_reve / hmc_revelarge)")
    args = ap.parse_args()
    stem = args.stem or ("hmc_revelarge" if "large" in args.model else "hmc_reve")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, pos = load_reve(args.model, device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"loaded {args.model}: {n_par/1e6:.1f}M params, positions {pos.shape}")

    t0 = time.time()
    recs = []
    pairs = pair_recordings(args.root)
    for i, (psg, hyp) in enumerate(pairs):
        r = one_recording(psg, hyp, model, pos, device, args.batch)
        if r is not None:
            recs.append(r)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pairs)} recordings ({time.time()-t0:.0f}s)")
    recs.sort(key=lambda r: r.subject)
    print(f"extracted {len(recs)} recordings in {time.time()-t0:.0f}s; token {recs[0].values.shape[1:]}")

    subj = np.array([r.subject for r in recs])
    tr_m, va_m, te_m = split_masks(subj)
    counts = tuple(int(sum((r.labels >= 0).sum() for r, m in zip(recs, mm) if m))
                   for mm in (tr_m, va_m, te_m))
    print(f"split recordings {tr_m.sum()}/{va_m.sum()}/{te_m.sum()}, epochs {counts}")
    assert (tr_m.sum(), va_m.sum(), te_m.sum()) == (100, 25, 26)
    assert counts == (91248, 22124, 23871), f"epoch counts {counts} != NeuroLM Table 1"

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    save_sleep_archives(recs, out / f"{stem}.npz", out / f"{stem}_labels.npz")
    pre = [r for r in recs if r.subject <= PRETRAIN_MAX_SUBJECT]
    save_de_archive(recordings_to_detrials(pre), out / f"{stem}_pretrain.npz")
    print(f"wrote {stem}.npz ({len(recs)}), labels, pretrain ({len(pre)})")


if __name__ == "__main__":
    main()
