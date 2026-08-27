"""PhysioNet/CinC Challenge 2018 ("You Snooze You Win") sleep-staging pipeline.

Third sleep corpus (docs/SLEEP_DATASET_CANDIDATES.md): 994 labeled training records
(MGH sleep lab), 6 EEG channels @ 200 Hz, AASM 30-s stages delivered as WFDB
``.arousal`` annotations (aux_note W/N1/N2/N3/R/undefined at 30-s-aligned samples;
a stage persists until the next stage mark; samples before the first mark and
'undefined' stretches are unscored -> dropped). Only the labeled training half of the
challenge is used — test-set stage labels were never released.

Mirrors physiofm/hmc.py: per-30-s ``feature_fn`` tokens into the SleepRecording
container so every downstream consumer is unchanged. No wake trimming — all scored
epochs are kept (the XSleepNet/SleePyCo ladder convention).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .sleep_edf import SleepRecording, EPOCH_SEC, DROP_LABEL

STAGE_TO_LABEL = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "undefined": DROP_LABEL}
STAGE_NAMES = set(STAGE_TO_LABEL)

# The 6 EEG derivations, in header order (verified on tr03-0005).
DEFAULT_EEG_CHANNELS = ("F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1")


def _epoch_labels(ann, n_epochs: int, sfreq: float) -> np.ndarray:
    """Expand stage marks (sample-indexed, persist-until-next) into per-epoch labels."""
    spe = int(round(EPOCH_SEC * sfreq))
    labels = np.full(n_epochs, DROP_LABEL, dtype=np.int64)
    marks = [(int(s), STAGE_TO_LABEL[a]) for s, a in zip(ann.sample, ann.aux_note)
             if a in STAGE_NAMES]
    marks.sort()
    for i, (start_sample, lab) in enumerate(marks):
        if start_sample % spe != 0:      # all stage marks are 30-s aligned (verified);
            raise ValueError(f"stage mark at non-epoch-aligned sample {start_sample}")
        start = start_sample // spe
        end = marks[i + 1][0] // spe if i + 1 < len(marks) else n_epochs
        if start >= n_epochs:
            continue
        labels[start:min(end, n_epochs)] = lab
    return labels


def load_recording(
    record_path: str | Path,
    channels: tuple[str, ...] = DEFAULT_EEG_CHANNELS,
    feature_fn=None,
) -> SleepRecording | None:
    """One WFDB record (path without extension) -> per-epoch features + labels."""
    import wfdb

    record_path = str(record_path)
    key = Path(record_path).name                       # "tr03-0005"
    subject = int(key.replace("tr", "").replace("-", ""))

    rec = wfdb.rdrecord(record_path, channel_names=list(channels))
    if rec.p_signal is None or rec.p_signal.shape[1] != len(channels):
        raise ValueError(f"{key}: missing EEG channels (got {rec.sig_name})")
    order = [rec.sig_name.index(c) for c in channels]  # enforce fixed order
    eeg = np.ascontiguousarray(rec.p_signal[:, order].T)  # (n_ch, n_samples), physical units
    sfreq = float(rec.fs)

    if feature_fn is None:
        raise ValueError("physio2018 is tf64-only; pass feature_fn")
    values = feature_fn(eeg, sfreq, EPOCH_SEC, EPOCH_SEC)  # (n_epochs, n_ch, n_feat)
    n_epochs = values.shape[0]
    if n_epochs == 0:
        return None

    ann = wfdb.rdann(record_path, "arousal")
    labels = _epoch_labels(ann, n_epochs, sfreq)

    keep = labels != DROP_LABEL                        # scored epochs only, no trim
    if keep.sum() == 0:
        return None
    return SleepRecording(
        subject=subject, night=1, key=key,
        values=values[keep].astype(np.float32),
        labels=labels[keep].astype(np.int64),
        source=record_path,
    )


def list_records(root: str | Path) -> list[Path]:
    """All record basepaths (no extension) under <root>/training, sorted."""
    tr = Path(root) / "training"
    recs = sorted(p.with_suffix("") for p in tr.glob("tr*/tr*.hea"))
    return recs


SPLIT_FILE = "data/physiofm/splits/p2018_sleepyco_folds.npy"


def load_folds(split_file: str | Path = SPLIT_FILE):
    """SleePyCo's published 5-fold assignment (split_idx/idx_Physio2018.npy, MIT).

    Returns a list of 5 dicts of integer index arrays {train, val, test}; indices
    refer to the position of each record in the lexicographically sorted record
    list — identical to our archive order (build sorts by key). Verified: test
    folds partition 0..993 (199/199/199/199/198), val = 50, train = remainder.
    """
    folds = np.load(split_file, allow_pickle=True)
    assert len(folds) == 5
    return [{k: np.asarray(sorted(int(i) for i in f[k]), dtype=np.int64)
             for k in ("train", "val", "test")} for f in folds]
