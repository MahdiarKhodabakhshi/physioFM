"""Structured-patch data pipeline for PhysioFM-S.

A trial (T, C, B) becomes a sequence of T tokens, each the flattened (C*B)=310-d
DE window. Normalization is a fixed per-(channel,band) standardization fit on the
pretraining corpus (NOT per-series-over-time instance norm), so absolute
band-power structure is preserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .de import DETrial, load_de_archive

ARCH = {
    "seed_v": "data/physiofm/de_features/seed_v_de_LDS.npz",
    "seed_iv": "data/physiofm/de_features/seed_iv_de_LDS.npz",
    "seed": "data/physiofm/de_features/seed_de_LDS.npz",
    # F1 (follow-up): un-smoothed per-window DE (moving-average, pre-LDS). Only
    # SEED-IV exposes an un-smoothed key in its published features; SEED-V/SEED
    # ship LDS-only, so no un-smoothed corpus exists for them without raw EEG.
    "seed_iv_raw": "data/physiofm/de_features/seed_iv_de_movingAve.npz",
    # F13 (Phase 3): Sleep-EDF per-epoch DE (un-smoothed). One DETrial per
    # recording with label=None so PC pretraining reuses it directly; per-epoch
    # sleep-stage labels live in the companion sleep_edf_labels.npz read by the
    # sleep evaluator (scripts/phase2_f13_sleep.py).
    "sleep_edf": "data/physiofm/de_features/sleep_edf_de.npz",
    # F16 (Phase 3): BCI-IV-2a motor imagery, 2nd dynamic task. Per-trial DE
    # sequence (13 windows x 22 ch x 5 bands); per-trial 4-way label; session
    # 1=T / 2=E for the canonical leakage-free session-holdout eval.
    "bci_iv_2a": "data/physiofm/de_features/bci_iv_2a_de.npz",
    # F17 (Phase 3): CHB-MIT scalp-EEG seizure detection — the truer 2nd dynamic
    # task (genuine sequence-level dynamics). Per-epoch DE (2 s epochs x 18 ch x 5
    # bands); per-epoch binary label (seizure/interictal) in the companion
    # chbmit_labels.npz, read by the seizure evaluator.
    "chbmit": "data/physiofm/de_features/chbmit_de.npz",
    # ---- Next-phase plan (docs/NEXT_PHASE_PLAN.md) ------------------------------------
    # Gate 0: rich time-frequency tokens — the same per-epoch pipeline as DE but 64
    # log-spaced log-power bins per channel instead of 5 bands (physiofm/spectral.py).
    # Same recordings, same order, same label companions as the DE archives.
    "sleep_edf_tf64": "data/physiofm/tf_features/sleep_edf_tf64.npz",
    "chbmit_tf64": "data/physiofm/tf_features/chbmit_tf64.npz",
    # Gate 2: RAW-EEG structured tokens. One token = all channels x 200 ms of raw signal
    # (sleep: 2 ch x 20 samples @100 Hz = 40-d; 150 tokens per 30 s epoch). Stored in the
    # DE-archive container as (tokens, channels, samples) so every consumer sees a
    # generic (T, C, S) sequence (physiofm/raw_eeg.py). *_perch = the BrainGPT-style
    # per-electrode decomposition ablation (one sequence per channel, token = 1 x 20).
    "sleep_edf_raw": "data/physiofm/raw_tokens/sleep_edf_raw200ms.npz",
    "sleep_edf_raw_perch": "data/physiofm/raw_tokens/sleep_edf_raw200ms_perch.npz",
    "chbmit_raw": "data/physiofm/raw_tokens/chbmit_raw200ms.npz",
    # External sleep validation (docs/SLEEP_DATASET_CANDIDATES.md): HMC, 151 clinical
    # PSGs, 4 EEG @ 256 Hz, tf64 tokens (4 x 64). *_pretrain = subjects SN001-SN125
    # only (fixed-split train+val) so pretraining/standardizer never see test subjects.
    "hmc_tf64": "data/physiofm/tf_features/hmc_tf64.npz",
    "hmc_tf64_pretrain": "data/physiofm/tf_features/hmc_tf64_pretrain.npz",
    # Physio2018 (CinC training set): 994 labeled records, 6 EEG @ 200 Hz, tf64 (6 x 64).
    # 5-fold subject-disjoint protocol; per-fold pretrain corpora are built on the fly.
    "p2018_tf64": "data/physiofm/tf_features/p2018_tf64.npz",
    # EXP-0026 cross-corpus transfer: per-electrode 1x64 slices of the tf64 archives
    # (channel-count-agnostic tokens; --merge_every C regroups at evaluation).
    "sleep_edf_tf64_perch": "data/physiofm/tf_features/sleep_edf_tf64_perch.npz",
    "hmc_tf64_perch": "data/physiofm/tf_features/hmc_tf64_perch.npz",
    "p2018_tf64_perch": "data/physiofm/tf_features/p2018_tf64_perch.npz",
    # EXP-0028: frozen REVE-base (69M-param EEG foundation model) epoch embeddings —
    # token = (4 ch x 512), per-channel mean over 1-s patches. *_pretrain = SN001-127.
    "hmc_reve": "data/physiofm/reve_features/hmc_reve.npz",
    "hmc_reve_pretrain": "data/physiofm/reve_features/hmc_reve_pretrain.npz",
    "hmc_revelarge": "data/physiofm/reve_features/hmc_revelarge.npz",
    "hmc_revelarge_pretrain": "data/physiofm/reve_features/hmc_revelarge_pretrain.npz",
    "sleep_edf_reve": "data/physiofm/reve_features/sleep_edf_reve.npz",
    "p2018_reve": "data/physiofm/reve_features/p2018_reve.npz",
}

# Tokens per labelled epoch for each archive (1 = one token per epoch, the DE default).
# Raw archives carry many tokens per 30 s / 2 s epoch; per-epoch consumers pool them.
TOKENS_PER_EPOCH = {
    "sleep_edf_raw": 150,
    "sleep_edf_raw_perch": 150,
    "chbmit_raw": 10,
}


def load_corpus(datasets: list[str]) -> list[DETrial]:
    trials: list[DETrial] = []
    for ds in datasets:
        # Unknown keys are treated as archive paths (ad-hoc corpora, e.g. the
        # per-fold Physio2018 pretraining archives).
        path = ARCH.get(ds, ds)
        if ds not in ARCH and not Path(path).exists():
            raise KeyError(f"{ds}: not an ARCH key and not an existing archive path")
        trials.extend(load_de_archive(path))
    return trials


def fit_standardizer(trials: list[DETrial], eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Per-(channel,band) mean/std over all windows in the corpus. Returns flat 310-d."""
    sums = None
    sqsums = None
    count = 0
    for t in trials:
        v = np.asarray(t.values, dtype=np.float64).reshape(t.values.shape[0], -1)
        if sums is None:
            sums = np.zeros(v.shape[1])
            sqsums = np.zeros(v.shape[1])
        sums += v.sum(0)
        sqsums += (v ** 2).sum(0)
        count += v.shape[0]
    mean = sums / count
    var = np.maximum(sqsums / count - mean ** 2, eps)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(values.shape[0], -1)
    return ((flat - mean) / std).astype(np.float32)


class SequenceDataset:
    """Yields standardized trial sequences (T, n_cb) for PC pretraining.

    ``max_len`` (next-phase plan): sequences longer than this are split into contiguous
    chunks so whole-night / whole-file / raw-token sequences fit GPU memory. Chunking is
    a pure memory bound (each chunk keeps its internal order and full context up to
    ``max_len``); leftover chunks shorter than ``min_len`` are dropped.
    """

    def __init__(self, trials: list[DETrial], mean: np.ndarray, std: np.ndarray, min_len: int = 2,
                 max_len: int = 0) -> None:
        self.seqs = []
        for t in trials:
            s = standardize(t.values, mean, std)
            if max_len and s.shape[0] > max_len:
                for i in range(0, s.shape[0], max_len):
                    c = s[i:i + max_len]
                    if c.shape[0] >= min_len:
                        self.seqs.append(c)
            elif s.shape[0] >= min_len:
                self.seqs.append(s)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.seqs[idx]


def collate_pad(batch: list[np.ndarray]) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Pad variable-length (T, n_cb) sequences to (B, Tmax, n_cb) + pad_mask (B, Tmax)."""
    import torch

    tmax = max(s.shape[0] for s in batch)
    n_cb = batch[0].shape[1]
    x = np.zeros((len(batch), tmax, n_cb), dtype=np.float32)
    mask = np.zeros((len(batch), tmax), dtype=np.float32)
    for i, s in enumerate(batch):
        x[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = 1.0
    return torch.from_numpy(x), torch.from_numpy(mask)


def save_standardizer(path: str | Path, mean: np.ndarray, std: np.ndarray, datasets: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, std=std, datasets=np.array(datasets))


def load_standardizer(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return z["mean"].astype(np.float32), z["std"].astype(np.float32)
