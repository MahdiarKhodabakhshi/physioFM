"""CHB-MIT scalp-EEG seizure data pipeline (F17, Phase 3).

Third task for the temporal-PC thesis, and the truer 2nd *dynamic* task after the
motor-imagery null ([[EXP-0014]]): seizure detection has genuine sequence-level
temporal structure (the signal evolves as a seizure develops), like sleep and
unlike the spatial-spectral emotion/MI signals.

Mirrors `sleep_edf.py`: per-epoch DE tokens + **per-epoch binary labels**
(seizure=1 / interictal=0), one recording (EDF) = one PC-pretraining sequence.
Two archives: `chbmit_de.npz` (label-less DETrials) + `chbmit_labels.npz`
(per-epoch labels + patient/file), so pretraining and the per-epoch evaluator reuse
the existing machinery unchanged. Seizures are ~1% of epochs, so the evaluator must
use imbalance-aware metrics (balanced acc / sensitivity / specificity / AUC).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .de import DETrial, DEFAULT_BANDS, compute_differential_entropy

EPOCH_SEC = 2.0  # 2 s epoch tokens: fine enough for seizure dynamics, long sequences/file
LABEL_NAMES = ["interictal", "seizure"]

# Standard 23-channel CHB-MIT bipolar montage (order from the -summary.txt files).
# Matching is case-insensitive; the duplicate T8-P8 is de-duplicated on load.
STD_CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1",
)
# canonical 18-ch subset present across chb01..chb24 (drops the montage-variable tail)
CORE_CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
)


@dataclass
class SeizureRecording:
    patient: int
    file_idx: int
    key: str
    values: np.ndarray   # (n_epochs, n_ch, n_band)
    labels: np.ndarray   # (n_epochs,) in {0,1}
    source: str


def parse_summary(summary_path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """Parse chbNN-summary.txt -> {edf_filename: [(start_sec, end_sec), ...]}."""
    text = Path(summary_path).read_text(errors="ignore")
    out: dict[str, list[tuple[float, float]]] = {}
    cur = None
    starts: list[float] = []
    ends: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("File Name:"):
            if cur is not None:
                out[cur] = list(zip(starts, ends))
            cur = line.split(":", 1)[1].strip()
            starts, ends = [], []
        elif "Seizure" in line and "Start Time" in line:
            m = re.search(r"(\d+)\s*seconds", line)
            if m:
                starts.append(float(m.group(1)))
        elif "Seizure" in line and "End Time" in line:
            m = re.search(r"(\d+)\s*seconds", line)
            if m:
                ends.append(float(m.group(1)))
    if cur is not None:
        out[cur] = list(zip(starts, ends))
    return out


def _pick_channels(raw, channels):
    """Case-insensitive channel pick honoring the requested order; None if any missing.
    Registers both the raw name and its de-duplicated base (T8-P8-0/-1 -> T8-P8), first
    occurrence winning, so the standard montage names resolve."""
    def norm(s):
        return s.upper().replace(" ", "")

    have: dict[str, str] = {}
    for c in raw.ch_names:
        n = norm(c)
        have.setdefault(n, c)
        base = re.sub(r"-[01]$", "", n)  # T8-P8-0 -> T8-P8
        have.setdefault(base, c)
    picks = []
    for want in channels:
        match = have.get(norm(want))
        if match is None:
            return None
        picks.append(match)
    return picks


def _epoch_seizure_labels(intervals, n_epochs: int) -> np.ndarray:
    labels = np.zeros(n_epochs, dtype=np.int64)
    for start, end in intervals:
        lo = int(np.floor(start / EPOCH_SEC))
        hi = int(np.ceil(end / EPOCH_SEC))
        labels[max(0, lo):min(n_epochs, hi)] = 1
    return labels


def load_recording(edf_path, intervals, channels=CORE_CHANNELS, bands=DEFAULT_BANDS, feature_fn=None):
    """One EDF -> per-epoch features + labels. ``feature_fn(eeg, sfreq, window_sec, step_sec)``
    replaces DE (next-phase plan) with identical epoching, so label companions stay aligned."""
    import mne

    edf_path = Path(edf_path)
    try:  # some CHB-MIT files are tiny placeholders (e.g. chb02_16+.edf, 303 B)
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    except Exception:
        return None
    picks = _pick_channels(raw, channels)
    if picks is None:
        return None
    raw.pick(picks)
    sfreq = float(raw.info["sfreq"])
    eeg = raw.get_data() * 1e6  # V -> uV
    if feature_fn is None:
        values = compute_differential_entropy(eeg, sfreq, EPOCH_SEC, EPOCH_SEC, bands)
    else:
        values = feature_fn(eeg, sfreq, EPOCH_SEC, EPOCH_SEC)
    n = values.shape[0]
    if n == 0:
        return None
    labels = _epoch_seizure_labels(intervals, n)
    # chb17 uses letter-suffixed sub-sessions (chb17a_/17b_/17c_); keep the numeric patient
    m = re.search(r"chb(\d+)[a-z]?_(\d+)", edf_path.stem)
    patient = int(m.group(1)) if m else -1
    file_idx = int(m.group(2)) if m else -1
    return SeizureRecording(patient, file_idx, edf_path.stem,
                            values.astype(np.float32), labels, str(edf_path))


def build_chbmit_corpus(root, patients=None, channels=CORE_CHANNELS,
                        seizure_files_only=False, limit=None, feature_fn=None):
    """Build per-recording seizure DE. seizure_files_only keeps interictal context
    manageable by taking only files that contain >=1 seizure (+ their labels)."""
    root = Path(root)
    pdirs = sorted(root.glob("chb*")) if patients is None else [root / p for p in patients]
    recs: list[SeizureRecording] = []
    for pdir in pdirs:
        if not pdir.is_dir():
            continue
        summ = pdir / f"{pdir.name}-summary.txt"
        if not summ.exists():
            continue
        ann = parse_summary(summ)
        for edf in sorted(pdir.glob("*.edf")):
            intervals = ann.get(edf.name, [])
            if seizure_files_only and not intervals:
                continue
            rec = load_recording(edf, intervals, channels=channels, feature_fn=feature_fn)
            if rec is not None:
                recs.append(rec)
            if limit and len(recs) >= limit:
                return recs
    return recs


def recordings_to_detrials(recs):
    return [DETrial(dataset="chbmit", subject=r.patient, session=r.file_idx, trial=i,
                    label=None, values=r.values, source=r.source)
            for i, r in enumerate(recs)]


def save_chbmit_archives(recs, de_path, labels_path):
    from .de import save_de_archive

    save_de_archive(recordings_to_detrials(recs), de_path)
    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
    label_arrays = np.empty(len(recs), dtype=object)
    for i, r in enumerate(recs):
        label_arrays[i] = r.labels.astype(np.int64)
    np.savez_compressed(
        labels_path, labels=label_arrays,
        patient=np.array([r.patient for r in recs], dtype=np.int64),
        file_idx=np.array([r.file_idx for r in recs], dtype=np.int64),
        key=np.array([r.key for r in recs]),
    )


def load_chbmit_labels(labels_path):
    with np.load(labels_path, allow_pickle=True) as z:
        labels = [np.asarray(a, dtype=np.int64) for a in z["labels"]]
        return labels, z["patient"].astype(np.int64), z["file_idx"].astype(np.int64), z["key"]
