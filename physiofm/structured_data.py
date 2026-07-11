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
}


def load_corpus(datasets: list[str]) -> list[DETrial]:
    trials: list[DETrial] = []
    for ds in datasets:
        trials.extend(load_de_archive(ARCH[ds]))
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
    """Yields standardized trial sequences (T, n_cb) for PC pretraining."""

    def __init__(self, trials: list[DETrial], mean: np.ndarray, std: np.ndarray, min_len: int = 2) -> None:
        self.seqs = [
            standardize(t.values, mean, std) for t in trials if t.values.shape[0] >= min_len
        ]

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
