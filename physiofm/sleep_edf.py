"""Sleep-EDF Expanded data pipeline for the F13 temporal-PC test (Phase 3).

Turns Sleep-EDF Cassette PSG recordings into the project's canonical DE currency
so the *same* PhysioFM-S machinery (structured patches, PC pretraining, frozen
encode) can be applied to a genuinely **dynamic** biosignal task — sleep staging —
where the pre-registered prediction is pretrained > random-init (unlike static
emotion DE; see docs/experiments/EXP-0009).

Key differences from the SEED emotion pipeline:
  * One token = one **30 s epoch**; DE is computed per epoch per (channel, band)
    over the full 30 s window (no LDS smoothing — keep the dynamics).
  * Labels are **per-epoch** (W / N1 / N2 / N3 / REM), not trial-constant.
  * A "recording" (one night for one subject) plays the role of a trial-sequence
    for PC pretraining; evaluation is **subject-disjoint** k-fold.

Each recording is stored two ways so the rest of the codebase is reused unchanged:
  * ``sleep_edf_de.npz``     — standard DE archive (one ``DETrial`` per recording,
    ``label=None``) so ``phase2_pretrain.py --datasets sleep_edf`` works as-is.
  * ``sleep_edf_labels.npz`` — companion per-epoch labels + subject/night, aligned
    to the same recording order, read only by the sleep evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .de import DETrial, compute_differential_entropy, DEFAULT_BANDS

# AASM 5-class mapping. Sleep-EDF uses R&K stages; merge stage 4 into N3, and
# drop MOVEMENT / UNKNOWN ("?") epochs (label -1, excluded from train/eval).
STAGE_TO_LABEL = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # R&K stage 4 -> N3
    "Sleep stage R": 4,
}
LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]
DROP_LABEL = -1

EPOCH_SEC = 30.0
DEFAULT_EEG_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")


@dataclass
class SleepRecording:
    subject: int
    night: int
    key: str                 # e.g. "SC4001"
    values: np.ndarray       # (n_epochs, n_ch, n_band) per-epoch DE
    labels: np.ndarray       # (n_epochs,) int in 0..4
    source: str


# --------------------------------------------------------------------------- IO
def _pair_recordings(root: str | Path) -> list[tuple[Path, Path]]:
    """Match each ``*-PSG.edf`` with its ``*-Hypnogram.edf`` by the SC4ssN prefix."""
    root = Path(root)
    psgs = {p.name[:6]: p for p in sorted(root.glob("*-PSG.edf"))}
    hyps = {p.name[:6]: p for p in sorted(root.glob("*-Hypnogram.edf"))}
    pairs = []
    for key in sorted(psgs):
        if key in hyps:
            pairs.append((psgs[key], hyps[key]))
    return pairs


def _epoch_labels_from_annotations(annotations, n_epochs: int) -> np.ndarray:
    """Expand mne annotations (onset, duration, stage) into per-30s-epoch labels."""
    labels = np.full(n_epochs, DROP_LABEL, dtype=np.int64)
    for onset, duration, desc in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        lab = STAGE_TO_LABEL.get(desc, DROP_LABEL)
        start = int(round(onset / EPOCH_SEC))
        count = int(round(duration / EPOCH_SEC))
        end = min(start + count, n_epochs)
        if start < n_epochs:
            labels[start:end] = lab
    return labels


def _trim_wake(labels: np.ndarray, keep_min: float | None) -> np.ndarray:
    """Boolean keep-mask trimming long W stretches to ``keep_min`` around sleep.

    Sleep-EDF Cassette recordings contain hours of lights-on wake; the standard
    protocol keeps only ~30 min of wake before the first and after the last
    non-wake epoch. ``keep_min=None`` keeps everything.
    """
    keep = labels != DROP_LABEL
    if keep_min is None:
        return keep
    sleep_idx = np.where((labels != 0) & (labels != DROP_LABEL))[0]
    if sleep_idx.size == 0:
        return keep
    pad = int(round(keep_min * 60.0 / EPOCH_SEC))
    lo = max(0, sleep_idx[0] - pad)
    hi = min(labels.size, sleep_idx[-1] + pad + 1)
    window = np.zeros_like(keep)
    window[lo:hi] = True
    return keep & window


def load_recording(
    psg_path: str | Path,
    hyp_path: str | Path,
    channels: tuple[str, ...] = DEFAULT_EEG_CHANNELS,
    bands=DEFAULT_BANDS,
    trim_wake_min: float | None = 30.0,
    feature_fn=None,
) -> SleepRecording | None:
    """Read one PSG + hypnogram into per-epoch DE + per-epoch labels.

    ``feature_fn(eeg, sfreq, window_sec, step_sec) -> (n_epochs, n_ch, n_feat)`` replaces
    the DE computation (next-phase plan: rich time-frequency tokens, raw tokens) while
    keeping recording order, epoching and wake-trimming identical, so the per-epoch label
    companion of the DE archive stays aligned.
    """
    import mne

    psg_path, hyp_path = Path(psg_path), Path(hyp_path)
    key = psg_path.name[:6]
    subject = int(psg_path.name[3:5])
    night = int(psg_path.name[5])

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    have = [c for c in channels if c in raw.ch_names]
    if not have:
        return None
    raw.pick(have)
    sfreq = float(raw.info["sfreq"])
    eeg = raw.get_data() * 1e6  # Volts -> microvolts (DE scale invariant, but keep sane units)

    # Per-30s-epoch DE: window == step == 30 s gives consecutive epoch tokens.
    if feature_fn is None:
        values = compute_differential_entropy(
            eeg, sfreq, window_sec=EPOCH_SEC, step_sec=EPOCH_SEC, bands=bands
        )  # (n_epochs, n_ch, n_band)
    else:
        values = feature_fn(eeg, sfreq, EPOCH_SEC, EPOCH_SEC)
    n_epochs = values.shape[0]
    if n_epochs == 0:
        return None

    ann = mne.read_annotations(hyp_path)
    labels = _epoch_labels_from_annotations(ann, n_epochs)

    keep = _trim_wake(labels, trim_wake_min)
    if keep.sum() == 0:
        return None
    values = values[keep].astype(np.float32)
    labels = labels[keep].astype(np.int64)
    return SleepRecording(
        subject=subject, night=night, key=key,
        values=values, labels=labels, source=str(psg_path),
    )


def build_sleep_corpus(
    root: str | Path,
    channels: tuple[str, ...] = DEFAULT_EEG_CHANNELS,
    trim_wake_min: float | None = 30.0,
    limit: int | None = None,
    feature_fn=None,
) -> list[SleepRecording]:
    pairs = _pair_recordings(root)
    if limit is not None:
        pairs = pairs[:limit]
    recs: list[SleepRecording] = []
    for psg, hyp in pairs:
        rec = load_recording(psg, hyp, channels=channels, trim_wake_min=trim_wake_min,
                             feature_fn=feature_fn)
        if rec is not None:
            recs.append(rec)
    return recs


# ----------------------------------------------------------------- archive save
def recordings_to_detrials(recs: list[SleepRecording]) -> list[DETrial]:
    """Wrap recordings as label-less DETrials so PC pretraining reuses them."""
    return [
        DETrial(
            dataset="sleep_edf", subject=r.subject, session=r.night, trial=i,
            label=None, values=r.values, source=r.source,
        )
        for i, r in enumerate(recs)
    ]


def save_sleep_archives(recs: list[SleepRecording], de_path: str | Path, labels_path: str | Path) -> None:
    """Write the DE archive (for pretraining) + the per-epoch label companion."""
    from .de import save_de_archive

    de_path, labels_path = Path(de_path), Path(labels_path)
    save_de_archive(recordings_to_detrials(recs), de_path)

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    label_arrays = np.empty(len(recs), dtype=object)
    for i, r in enumerate(recs):
        label_arrays[i] = r.labels.astype(np.int64)
    np.savez_compressed(
        labels_path,
        labels=label_arrays,
        subject=np.array([r.subject for r in recs], dtype=np.int64),
        night=np.array([r.night for r in recs], dtype=np.int64),
        key=np.array([r.key for r in recs]),
    )


def load_sleep_labels(labels_path: str | Path):
    """Return (list[label_array], subject_array, night_array, key_array)."""
    with np.load(labels_path, allow_pickle=True) as z:
        labels = [np.asarray(a, dtype=np.int64) for a in z["labels"]]
        return labels, z["subject"].astype(np.int64), z["night"].astype(np.int64), z["key"]
