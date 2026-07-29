"""VitalDB anaesthesia-depth pipeline (F19).

Why this task. Our diagnosis (EXP-0017) says predictive-coding pretraining should help when
(a) labels are per-epoch and (b) what is predictable is also what is discriminative.
Anaesthesia depth fits both better than anything we have run: BIS is a *continuously evolving
state* driven by drug concentration, sampled per second over hours, so the temporal trajectory
IS the label. It is also structurally identical to sleep — 2 EEG channels, long recordings,
5 ordered classes — which makes it a clean comparison.

Data: VitalDB (Seoul National University Hospital), 5,867 surgical cases with raw BIS-monitor
EEG (`BIS/EEG1_WAV`, `BIS/EEG2_WAV` @128 Hz) plus the BIS index and its signal-quality index.

Labels: the standard 5 depth bands, matching clinical convention —
    0 burst-suppression (BIS 0-20) · 1 deep hypnosis (20-40) · 2 general anaesthesia (40-60)
    3 moderate sedation (60-80)     · 4 awake / light (80-100)

CAUTION worth stating in any write-up: BIS is itself a proprietary index computed *from* the
EEG spectrum, so predicting BIS from spectral features is partly reverse-engineering a
spectral formula. That is exactly why the linear-saturation check must be run FIRST — if a
linear model on raw DE already recovers BIS, there is no headroom for representation learning
and the task cannot discriminate our hypotheses (the same trap as smoothed emotion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .de import DEFAULT_BANDS, DETrial, compute_differential_entropy

SFREQ = 128.0
EPOCH_SEC = 10.0
TRACKS = ["BIS/EEG1_WAV", "BIS/EEG2_WAV", "BIS/BIS", "BIS/SQI"]
LABEL_NAMES = ["burst_supp", "deep", "general", "moderate", "awake"]
DROP_LABEL = -1
MIN_SQI = 50.0          # BIS-monitor signal-quality index; below this the index is unreliable
EEG_ABS_MAX = 500.0     # uV; epochs exceeding this are artefact (cautery, movement)


@dataclass
class AnesthesiaRecording:
    case_id: int
    values: np.ndarray   # (n_epochs, 2, 5) per-epoch DE
    labels: np.ndarray   # (n_epochs,) 0..4, or -1 = drop
    source: str


def bis_to_class(bis: np.ndarray) -> np.ndarray:
    """BIS 0-100 -> 5 ordered depth classes (20-wide bands)."""
    return np.clip((bis // 20).astype(np.int64), 0, 4)


def load_case(case_id: int, bands=DEFAULT_BANDS) -> AnesthesiaRecording | None:
    """Download one case and reduce it to per-epoch DE + depth label."""
    import vitaldb

    try:
        d = vitaldb.load_case(case_id, TRACKS, 1.0 / SFREQ)
    except Exception:
        return None
    if d is None or d.shape[0] < SFREQ * EPOCH_SEC * 10:
        return None

    eeg = d[:, :2].T                      # (2, samples)
    bis, sqi = d[:, 2], d[:, 3]
    n_per = int(SFREQ * EPOCH_SEC)
    n_ep = eeg.shape[1] // n_per
    if n_ep < 10:
        return None
    eeg = eeg[:, : n_ep * n_per]

    # per-epoch label = median BIS over the epoch, gated on signal quality
    lab = np.full(n_ep, DROP_LABEL, dtype=np.int64)
    for i in range(n_ep):
        sl = slice(i * n_per, (i + 1) * n_per)
        b, q = bis[sl], sqi[sl]
        b = b[~np.isnan(b)]; q = q[~np.isnan(q)]
        if b.size == 0 or q.size == 0 or np.median(q) < MIN_SQI:
            continue
        lab[i] = bis_to_class(np.array([np.median(b)]))[0]

    # interpolate short NaN runs in the EEG, then drop artefact epochs
    for c in range(eeg.shape[0]):
        x = eeg[c]
        bad = np.isnan(x)
        if bad.all():
            return None
        if bad.any():
            x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    de = compute_differential_entropy(eeg, SFREQ, EPOCH_SEC, EPOCH_SEC, bands)  # (n_ep,2,5)
    n = min(de.shape[0], lab.shape[0])
    de, lab = de[:n], lab[:n]

    amp = np.abs(eeg[:, : n * n_per]).reshape(2, n, n_per).max(axis=(0, 2))
    lab[amp > EEG_ABS_MAX] = DROP_LABEL
    if (lab >= 0).sum() < 30:
        return None
    return AnesthesiaRecording(case_id, de.astype(np.float32), lab, f"vitaldb:{case_id}")


def recordings_to_detrials(recs) -> list[DETrial]:
    return [DETrial(dataset="vitaldb", subject=r.case_id, session=0, trial=i,
                    label=None, values=r.values, source=r.source)
            for i, r in enumerate(recs)]


def save_archives(recs, de_path, labels_path) -> None:
    from pathlib import Path

    from .de import save_de_archive

    save_de_archive(recordings_to_detrials(recs), de_path)
    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty(len(recs), dtype=object)
    for i, r in enumerate(recs):
        arr[i] = r.labels.astype(np.int64)
    np.savez_compressed(labels_path, labels=arr,
                        case_id=np.array([r.case_id for r in recs], dtype=np.int64))


def load_labels(path):
    with np.load(path, allow_pickle=True) as z:
        return [np.asarray(a, dtype=np.int64) for a in z["labels"]], z["case_id"].astype(np.int64)
