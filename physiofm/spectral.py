"""Rich time–frequency tokens (next-phase plan, Gate 0).

Why. DE compresses each (channel, epoch) to 5 numbers (log band power in δ/θ/α/β/γ) and
F10 showed the DE→label map is linearly saturated: a linear probe already reads DE
optimally, so no encoder trained on DE has representational headroom (requirement R1 in
docs/NEXT_PHASE_PLAN.md). Gate 0 asks the cheapest possible question before touching raw
EEG: does *less compression alone* — the same per-epoch pipeline but 64 log-spaced
log-power bins per channel instead of 5 bands — restore headroom (nonlinear > linear)?

What. ``compute_log_spectrogram`` returns ``epochs x channels x n_bins`` log-power computed
by Welch's method inside each epoch (nperseg = min(4 s, epoch) so the finest bin is 0.25 Hz), then
averaged into ``n_bins`` log-spaced bins between ``fmin`` and ``fmax``. It is a strict
superset of DE in information terms (DE = 5 coarse integrals of the same spectrum) and it
drops into the existing (T, C, B) archive container unchanged, so every downstream
consumer (standardizer, PhysioFM-S, evaluators) works with ``n_cb = C * 64``.
"""

from __future__ import annotations

import numpy as np

DEFAULT_N_BINS = 64
DEFAULT_FMIN = 0.5
DEFAULT_FMAX = 49.0  # below the 50 Hz Nyquist of the 100 Hz Sleep-EDF EEG (and matches DE's 51 Hz gamma cap)


def log_spaced_edges(n_bins: int = DEFAULT_N_BINS, fmin: float = DEFAULT_FMIN, fmax: float = DEFAULT_FMAX) -> np.ndarray:
    return np.geomspace(fmin, fmax, n_bins + 1)


def compute_log_spectrogram(
    eeg: np.ndarray,
    sfreq: float,
    window_sec: float,
    step_sec: float,
    n_bins: int = DEFAULT_N_BINS,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    nperseg_sec: float = 4.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-window Welch log-power in ``n_bins`` log-spaced bins.

    Args:
        eeg: channels x samples.
        sfreq: sampling rate (Hz).
        window_sec / step_sec: epoch length and hop (seconds), as in compute_differential_entropy.
    Returns:
        windows x channels x n_bins (float32), natural-log power.
    """
    from scipy import signal

    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2:
        raise ValueError(f"Expected channels x samples EEG, got shape {eeg.shape}")
    fmax = min(fmax, 0.98 * sfreq / 2.0)
    window = int(round(window_sec * sfreq))
    step = int(round(step_sec * sfreq))
    if eeg.shape[1] < window:
        return np.empty((0, eeg.shape[0], n_bins), dtype=np.float32)
    nperseg = min(int(round(nperseg_sec * sfreq)), window)

    starts = np.arange(0, eeg.shape[1] - window + 1, step)
    # (n_windows, n_ch, window) view without copying
    idx = starts[:, None] + np.arange(window)[None, :]
    chunks = eeg[:, idx]                                   # (n_ch, n_win, window)
    chunks = np.transpose(chunks, (1, 0, 2))               # (n_win, n_ch, window)
    freqs, psd = signal.welch(chunks, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2,
                              axis=-1, detrend="constant", scaling="density")
    edges = log_spaced_edges(n_bins, fmin, fmax)
    out = np.empty((chunks.shape[0], chunks.shape[1], n_bins), dtype=np.float64)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():  # bin narrower than the frequency resolution: take nearest line
            sel = np.zeros_like(freqs, dtype=bool)
            sel[np.argmin(np.abs(freqs - 0.5 * (lo + hi)))] = True
        out[:, :, b] = psd[:, :, sel].mean(-1)
    return np.log(np.maximum(out, eps)).astype(np.float32)
