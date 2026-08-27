"""HMC (Haaglanden Medisch Centrum) sleep-staging pipeline — external validation corpus.

Second sleep dataset for the tf64 + causal-decoder + PC-pretraining recipe validated on
Sleep-EDF-78 (docs/SLEEP_DATASET_CANDIDATES.md). 151 clinical PSGs, one per subject,
4 EEG derivations @ 256 Hz, AASM 5-class 30-s scoring (PhysioNet
hmc-sleep-staging v1.1, CC-BY). The pipeline mirrors ``physiofm/sleep_edf.py`` exactly
(same epoching, ±30-min wake trim, ``feature_fn`` injection, SleepRecording container)
so every downstream consumer — archives, pretraining, fine-tuning — is reused unchanged.

Protocol: the fixed published split ("NeuroLM protocol", also used by REVE/CSBrain/
LaBraM reruns) — subjects SN001–SN100 train, SN101–SN125 val, SN126–SN151 test —
reported as balanced accuracy / Cohen's kappa / weighted-F1, full fine-tune.
Pretraining must only ever see train+val subjects (<= PRETRAIN_MAX_SUBJECT).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .sleep_edf import SleepRecording, EPOCH_SEC, DROP_LABEL, _trim_wake

# AASM v2.4 scoring, annotation vocabulary verified on SN001_sleepscoring.edf
# (2026-08-27): exactly these five stage strings plus "Lights off@@…"/"Lights on@@…"
# events (duration 0 -> no epochs). Unknown descriptions map to DROP_LABEL.
STAGE_TO_LABEL = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}

# All four EEG derivations shipped with every recording (channel names verified on
# SN001.edf). Order is fixed; the build asserts every recording has all four so the
# structured token is always (4 x n_feat).
DEFAULT_EEG_CHANNELS = ("EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2")

# Fixed split boundaries. The published split (NeuroLM prepare_HMC.py, confirmed
# verbatim by CSBrain App. D and REVE App. C) is POSITIONAL: sort the 151 SNxxx.edf
# lexicographically, first 100 train / next 25 val / last 26 test. On the v1.1 roster
# (SN001..SN154 with SN014, SN064, SN135 absent) that is exactly subject-number
# thresholds 102 / 127: train SN001-SN102 (100 recs, 91,248 epochs), val SN103-SN127
# (25 recs, 22,124), test SN128-SN154 (26 recs, 23,871); total 137,243 = NeuroLM Table 1.
TRAIN_MAX_SUBJECT = 102      # first 100 recordings
VAL_MAX_SUBJECT = 127        # next 25 recordings
PRETRAIN_MAX_SUBJECT = VAL_MAX_SUBJECT  # pretraining corpus = train + val, never test


def _epoch_labels(annotations, n_epochs: int) -> np.ndarray:
    """Expand mne annotations into per-30-s labels (HMC stage vocabulary)."""
    labels = np.full(n_epochs, DROP_LABEL, dtype=np.int64)
    for onset, duration, desc in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        lab = STAGE_TO_LABEL.get(desc, DROP_LABEL)
        if lab == DROP_LABEL:
            continue  # lights on/off etc. — never overwrite a scored epoch
        start = int(round(onset / EPOCH_SEC))
        count = int(round(duration / EPOCH_SEC))
        if start >= n_epochs:
            continue
        labels[start : min(start + count, n_epochs)] = lab
    return labels


def load_recording(
    psg_path: str | Path,
    hyp_path: str | Path,
    channels: tuple[str, ...] = DEFAULT_EEG_CHANNELS,
    trim_wake_min: float | None = None,
    feature_fn=None,
) -> SleepRecording | None:
    """One PSG + scoring EDF -> per-epoch features + labels (mirrors sleep_edf).

    Default ``trim_wake_min=None``: the published HMC protocol keeps ALL scored
    epochs (pre-lights-off wake included) — a ±30-min trim would drop ~1.2k scored
    epochs and make the test set incomparable to the NeuroLM-ladder rows."""
    import mne

    psg_path, hyp_path = Path(psg_path), Path(hyp_path)
    key = psg_path.stem                      # "SN001"
    subject = int(key[2:])

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    have = [c for c in channels if c in raw.ch_names]
    if len(have) != len(channels):
        missing = [c for c in channels if c not in raw.ch_names]
        raise ValueError(f"{key}: missing EEG channels {missing} (have {raw.ch_names})")
    raw.pick(list(channels))                 # fixed order = `channels`
    sfreq = float(raw.info["sfreq"])
    eeg = raw.get_data() * 1e6               # V -> uV

    if feature_fn is None:
        raise ValueError("HMC is tf64-only; pass feature_fn (see build_hmc_dataset.py)")
    values = feature_fn(eeg, sfreq, EPOCH_SEC, EPOCH_SEC)   # (n_epochs, n_ch, n_feat)
    n_epochs = values.shape[0]
    if n_epochs == 0:
        return None

    ann = mne.read_annotations(hyp_path)
    labels = _epoch_labels(ann, n_epochs)

    keep = _trim_wake(labels, trim_wake_min)
    if keep.sum() == 0:
        return None
    return SleepRecording(
        subject=subject, night=1, key=key,
        values=values[keep].astype(np.float32),
        labels=labels[keep].astype(np.int64),
        source=str(psg_path),
    )


def pair_recordings(root: str | Path) -> list[tuple[Path, Path]]:
    """Match SNxxx.edf with SNxxx_sleepscoring.edf under <root>/recordings."""
    rec = Path(root) / "recordings"
    pairs = []
    for psg in sorted(rec.glob("SN[0-9][0-9][0-9].edf")):
        hyp = rec / f"{psg.stem}_sleepscoring.edf"
        if hyp.exists():
            pairs.append((psg, hyp))
    return pairs


def build_corpus(root: str | Path, feature_fn, trim_wake_min: float | None = None,
                 channels: tuple[str, ...] = DEFAULT_EEG_CHANNELS) -> list[SleepRecording]:
    recs = []
    for psg, hyp in pair_recordings(root):
        r = load_recording(psg, hyp, channels=channels,
                           trim_wake_min=trim_wake_min, feature_fn=feature_fn)
        if r is not None:
            recs.append(r)
    return recs


def split_masks(subjects: np.ndarray):
    """(train, val, test) boolean masks under the fixed published split."""
    subjects = np.asarray(subjects)
    train = subjects <= TRAIN_MAX_SUBJECT
    val = (subjects > TRAIN_MAX_SUBJECT) & (subjects <= VAL_MAX_SUBJECT)
    test = subjects > VAL_MAX_SUBJECT
    return train, val, test
