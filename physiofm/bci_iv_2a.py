"""BCI Competition IV dataset 2a (motor imagery) -> DE trials.

Second dynamic task for the temporal-PC thesis (EXP-0009 generalization). 9
subjects, 22 EEG channels @250 Hz, 4 balanced MI classes (left hand, right hand,
feet, tongue), 288 trials/subject/session. Two sessions per subject: T (training)
and E (evaluation) — the canonical **session-holdout** protocol (train on T, test
on E) which is leakage-free by construction.

Each trial is turned into a *sequence* of DE windows over the motor-imagery period
so the predictive-coding objective has within-trial temporal structure to model
(motor imagery has genuine ERD/ERS dynamics). Downstream label is per-trial (4-way),
so the readout is trial-level (pool a trial's window features) — mirroring emotion,
but on a task with real temporal dynamics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .de import DEFAULT_BANDS, DETrial, compute_differential_entropy

N_EEG = 22  # first 22 channels are EEG; last 3 are EOG (dropped)
CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")
SESSION_KIND = {"T": 1, "E": 2}  # T=train session, E=eval session


def load_bci_trials(
    root: str | Path,
    mi_start_sec: float = 2.0,   # trial onset=fixation(0s); cue 2s; MI 3-6s
    mi_end_sec: float = 6.0,     # [2,6]s window covers cue + imagery (4 s)
    window_sec: float = 1.0,
    step_sec: float = 0.25,
    bands=DEFAULT_BANDS,
) -> list[DETrial]:
    """Load all A0{1..9}{T,E}.mat -> per-trial DE sequences (windows x 22 x 5)."""
    from scipy.io import loadmat

    root = Path(root)
    trials: list[DETrial] = []
    n_skipped = 0
    for subj in range(1, 10):
        for kind, sess_id in SESSION_KIND.items():
            path = root / f"A{subj:02d}{kind}.mat"
            if not path.exists():
                raise FileNotFoundError(f"missing {path}")
            runs = loadmat(path, struct_as_record=False, squeeze_me=True)["data"]
            tidx = 0
            for run in np.atleast_1d(runs):
                onsets = np.atleast_1d(getattr(run, "trial", []))
                if onsets.size == 0:  # skip eyes-open/closed/EOG calibration runs
                    continue
                fs = float(run.fs)
                eeg = np.asarray(run.X, dtype=np.float64)[:, :N_EEG].T  # (22, samples)
                ys = np.atleast_1d(run.y)
                a = int(round(mi_start_sec * fs))
                b = int(round(mi_end_sec * fs))
                need = int(round(window_sec * fs))
                for onset, y in zip(onsets, ys):
                    seg = eeg[:, int(onset) + a : int(onset) + b]  # (22, ~1000)
                    if seg.shape[1] < need or np.isnan(seg).any():
                        n_skipped += 1
                        continue
                    de = compute_differential_entropy(seg, fs, window_sec, step_sec, bands)
                    if de.shape[0] == 0:
                        n_skipped += 1
                        continue
                    trials.append(DETrial(
                        dataset="bci_iv_2a", subject=subj, session=sess_id,
                        trial=tidx, label=int(y) - 1, values=de, source=str(path),
                    ))
                    tidx += 1
    if not trials:
        raise FileNotFoundError(f"No BCI-IV-2a trials built from {root}")
    if n_skipped:
        print(f"[bci_iv_2a] skipped {n_skipped} trials (short/NaN)")
    return trials


def bci_fold_labels(trials: list[DETrial]) -> np.ndarray:
    """Per-trial 4-way label array aligned with `trials`."""
    return np.array([t.label for t in trials], dtype=np.int64)
