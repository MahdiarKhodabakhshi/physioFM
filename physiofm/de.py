"""Differential entropy feature loading and extraction utilities.

Canonical DE trial shape throughout this package is:

    time x channels x bands

TimesFM Phase 1 then treats each channel-band trace as one univariate series.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 14.0),
    ("beta", 14.0, 31.0),
    ("gamma", 31.0, 51.0),
)


SEED_IV_LABELS: dict[int, list[int]] = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}


@dataclass(frozen=True)
class DETrial:
    dataset: str
    subject: int
    session: int
    trial: int
    label: int | None
    values: np.ndarray
    source: str


def compute_differential_entropy(
    eeg: np.ndarray,
    sfreq: float,
    window_sec: float = 4.0,
    step_sec: float = 1.0,
    bands: Iterable[tuple[str, float, float]] = DEFAULT_BANDS,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute band-wise Gaussian differential entropy from raw EEG.

    Args:
        eeg: Array shaped channels x samples.
        sfreq: Sampling rate in Hz.
        window_sec: Window length in seconds.
        step_sec: Step length in seconds.
        bands: Frequency bands as (name, low_hz, high_hz).
        eps: Minimum variance for numerical stability.

    Returns:
        Array shaped windows x channels x bands.
    """
    from scipy import signal

    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2:
        raise ValueError(f"Expected channels x samples EEG, got shape {eeg.shape}")

    window = int(round(window_sec * sfreq))
    step = int(round(step_sec * sfreq))
    if window <= 1 or step <= 0:
        raise ValueError("window_sec and step_sec must produce positive sample counts")
    if eeg.shape[1] < window:
        return np.empty((0, eeg.shape[0], len(tuple(bands))), dtype=np.float32)

    band_features: list[np.ndarray] = []
    nyq = sfreq / 2.0
    for _, low, high in bands:
        high = min(high, nyq * 0.98)
        if low <= 0 or high <= low:
            raise ValueError(f"Invalid band ({low}, {high}) for sfreq={sfreq}")
        sos = signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
        filtered = signal.sosfiltfilt(sos, eeg, axis=-1)

        windows = []
        for start in range(0, filtered.shape[1] - window + 1, step):
            chunk = filtered[:, start : start + window]
            var = np.var(chunk, axis=-1, ddof=1)
            windows.append(0.5 * np.log(2.0 * np.pi * np.e * np.maximum(var, eps)))
        band_features.append(np.stack(windows, axis=0))

    return np.stack(band_features, axis=-1).astype(np.float32)


def load_seed_v_npz(root: str | Path) -> list[DETrial]:
    """Load SEED-V precomputed DE `.npz` files."""
    import pickle

    root = Path(root)
    trials: list[DETrial] = []
    for npz_path in sorted(root.glob("*.npz")):
        subject = int(npz_path.stem.split("_")[0])
        with np.load(npz_path, allow_pickle=True) as npz:
            raw_data = pickle.loads(npz["data"].tobytes())
            raw_labels = pickle.loads(npz["label"].tobytes())

        for trial_id in sorted(raw_data):
            flat = np.asarray(raw_data[trial_id], dtype=np.float32)
            values = flat.reshape(flat.shape[0], 62, 5)
            labels = np.asarray(raw_labels[trial_id]).reshape(-1)
            label = int(labels[0]) if labels.size else None
            trials.append(
                DETrial(
                    dataset="seed_v",
                    subject=subject,
                    session=(int(trial_id) // 15) + 1,
                    trial=int(trial_id),
                    label=label,
                    values=values,
                    source=str(npz_path),
                )
            )
    if not trials:
        raise FileNotFoundError(f"No SEED-V .npz files found under {root}")
    return trials


def load_seed_iv_mat(root: str | Path, feature_key: str = "de_LDS") -> list[DETrial]:
    """Load SEED-IV `eeg_feature_smooth` MATLAB files."""
    from scipy.io import loadmat

    root = Path(root)
    trials: list[DETrial] = []
    for mat_path in sorted(root.glob("*/*.mat")):
        session = int(mat_path.parent.name)
        subject = int(mat_path.stem.split("_")[0])
        mat = loadmat(mat_path)
        for trial_in_session, label in enumerate(SEED_IV_LABELS[session], start=1):
            key = f"{feature_key}{trial_in_session}"
            if key not in mat:
                raise KeyError(f"{key} not found in {mat_path}")
            values = np.asarray(mat[key], dtype=np.float32).transpose(1, 0, 2)
            trials.append(
                DETrial(
                    dataset="seed_iv",
                    subject=subject,
                    session=session,
                    trial=(session - 1) * 24 + (trial_in_session - 1),
                    label=int(label),
                    values=values,
                    source=str(mat_path),
                )
            )
    if not trials:
        raise FileNotFoundError(f"No SEED-IV .mat files found under {root}")
    return trials


def load_seed_mat(root: str | Path, feature_key: str = "de_LDS") -> list[DETrial]:
    """Load original SEED `ExtractedFeatures_4s` MATLAB files."""
    from scipy.io import loadmat

    root = Path(root)
    label_path = root / "label.mat"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing SEED label file: {label_path}")
    labels = np.asarray(loadmat(label_path, squeeze_me=True)["label"]).reshape(-1)

    by_subject: dict[int, list[Path]] = {}
    for mat_path in sorted(root.glob("*.mat")):
        if mat_path.name == "label.mat":
            continue
        subject = int(mat_path.stem.split("_")[0])
        by_subject.setdefault(subject, []).append(mat_path)

    trials: list[DETrial] = []
    for subject, paths in sorted(by_subject.items()):
        for session, mat_path in enumerate(sorted(paths), start=1):
            mat = loadmat(mat_path)
            for trial_in_session, raw_label in enumerate(labels, start=1):
                key = f"{feature_key}{trial_in_session}"
                if key not in mat:
                    raise KeyError(f"{key} not found in {mat_path}")
                values = np.asarray(mat[key], dtype=np.float32).transpose(1, 0, 2)
                trials.append(
                    DETrial(
                        dataset="seed",
                        subject=subject,
                        session=session,
                        trial=(session - 1) * len(labels) + (trial_in_session - 1),
                        label=int(raw_label),
                        values=values,
                        source=str(mat_path),
                    )
                )
    if not trials:
        raise FileNotFoundError(f"No SEED .mat files found under {root}")
    return trials


def load_trials(dataset: str, root: str | Path, feature_key: str = "de_LDS") -> list[DETrial]:
    dataset = dataset.lower().replace("-", "_")
    if dataset == "seed_v":
        return load_seed_v_npz(root)
    if dataset == "seed_iv":
        return load_seed_iv_mat(root, feature_key=feature_key)
    if dataset == "seed":
        return load_seed_mat(root, feature_key=feature_key)
    raise ValueError(f"Unsupported dataset: {dataset}")


def save_de_archive(trials: list[DETrial], output_path: str | Path, dtype=np.float32) -> None:
    """Save variable-length DE trials to a compressed local archive.

    ``dtype`` (next-phase plan): raw-token archives are stored as float16 to halve disk/RAM;
    ``load_de_archive`` always returns float32.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.empty(len(trials), dtype=object)
    for i, trial in enumerate(trials):
        values[i] = np.asarray(trial.values, dtype=dtype)

    labels = np.array([trial.label if trial.label is not None else -999 for trial in trials], dtype=np.int64)
    label_missing = np.array([trial.label is None for trial in trials], dtype=bool)
    np.savez_compressed(
        output_path,
        values=values,
        dataset=np.array([trial.dataset for trial in trials]),
        subject=np.array([trial.subject for trial in trials], dtype=np.int64),
        session=np.array([trial.session for trial in trials], dtype=np.int64),
        trial=np.array([trial.trial for trial in trials], dtype=np.int64),
        label=labels,
        label_missing=label_missing,
        source=np.array([trial.source for trial in trials]),
    )


def load_de_archive(path: str | Path) -> list[DETrial]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as archive:
        values = archive["values"]
        datasets = archive["dataset"]
        subjects = archive["subject"]
        sessions = archive["session"]
        trial_ids = archive["trial"]
        labels = archive["label"]
        missing = archive["label_missing"]
        sources = archive["source"]
        trials = []
        for i in range(len(values)):
            trials.append(
                DETrial(
                    dataset=str(datasets[i]),
                    subject=int(subjects[i]),
                    session=int(sessions[i]),
                    trial=int(trial_ids[i]),
                    label=None if bool(missing[i]) else int(labels[i]),
                    values=np.asarray(values[i], dtype=np.float32),
                    source=str(sources[i]),
                )
            )
    return trials


def trial_to_univariate_series(values: np.ndarray) -> np.ndarray:
    """Convert one trial from time x channels x bands to series x time."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected time x channels x bands, got {values.shape}")
    return values.transpose(1, 2, 0).reshape(values.shape[1] * values.shape[2], values.shape[0])


def summarize_trials(trials: list[DETrial]) -> dict[str, object]:
    lengths = np.array([trial.values.shape[0] for trial in trials], dtype=np.int64)
    channels = sorted({int(trial.values.shape[1]) for trial in trials})
    bands = sorted({int(trial.values.shape[2]) for trial in trials})
    labels = sorted({trial.label for trial in trials if trial.label is not None})
    return {
        "trials": len(trials),
        "subjects": len({trial.subject for trial in trials}),
        "sessions": len({(trial.subject, trial.session) for trial in trials}),
        "labels": labels,
        "channels": channels,
        "bands": bands,
        "min_len": int(lengths.min()) if len(lengths) else 0,
        "mean_len": float(lengths.mean()) if len(lengths) else 0.0,
        "max_len": int(lengths.max()) if len(lengths) else 0,
        "univariate_series": int(sum(trial.values.shape[1] * trial.values.shape[2] for trial in trials)),
    }
