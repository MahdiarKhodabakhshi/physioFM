"""Raw-EEG structured tokens (next-phase plan, Gate 2).

Token = ALL channels × ``token_sec`` (200 ms) of lightly high-passed raw signal — the
structured-patch idea applied to the raw waveform instead of to DE. Sleep-EDF: 2 ch × 20
samples @ 100 Hz = 40-d token, 150 tokens per 30 s epoch. CHB-MIT: 18 ch × 51 samples
@ 256 Hz (10 tokens per 2 s epoch) if run.

The tokens ride in the existing DE-archive container as ``(tokens, channels, samples)`` so the
standardizer / PhysioFM-S / evaluators see a generic ``(T, C, S)`` sequence with
``n_cb = C * S``; per-epoch consumers pool ``TOKENS_PER_EPOCH`` consecutive tokens
(``structured_data.TOKENS_PER_EPOCH``).

Per-electrode ablation (BrainGPT-style decomposition): the same signal, but every channel is
its own sequence of ``(1, S)`` tokens; ``build_raw_dataset.py --per_channel`` writes one
DETrial per (recording, channel), consecutive, so evaluators can merge groups of ``C`` trials
back into per-epoch decisions (``--merge_every C``).

Preprocessing: 4th-order Butterworth high-pass at 0.3 Hz (drift), no resampling, µV units.
No per-series instance normalisation anywhere (the Phase-1 lesson): the corpus standardizer is a
fixed per-(channel, sample-offset) affine map.
"""

from __future__ import annotations

import numpy as np

HIGHPASS_HZ = 0.3


def raw_token_feature_fn(token_sec: float = 0.2, highpass_hz: float = HIGHPASS_HZ):
    """Return feature_fn(eeg, sfreq, window_sec, step_sec) -> (n_epochs, n_ch, samples_per_epoch)
    of raw samples, epoched exactly like DE (window == step == epoch), so trimming / labels align.
    ``samples_per_epoch`` is truncated to a multiple of the token length."""
    from scipy import signal

    def fn(eeg, sfreq, window_sec, step_sec):
        eeg = np.asarray(eeg, dtype=np.float64)
        nyq = sfreq / 2.0
        if highpass_hz > 0:
            sos = signal.butter(4, highpass_hz / nyq, btype="highpass", output="sos")
            eeg = signal.sosfiltfilt(sos, eeg, axis=-1)
        window = int(round(window_sec * sfreq))
        step = int(round(step_sec * sfreq))
        tok = int(round(token_sec * sfreq))
        n_tok = window // tok
        keep = n_tok * tok
        if eeg.shape[1] < window:
            return np.empty((0, eeg.shape[0], keep), dtype=np.float32)
        starts = np.arange(0, eeg.shape[1] - window + 1, step)
        idx = starts[:, None] + np.arange(keep)[None, :]
        out = eeg[:, idx]                       # (n_ch, n_epochs, keep)
        return np.transpose(out, (1, 0, 2)).astype(np.float32)  # (n_epochs, n_ch, keep)

    return fn


def epochs_to_tokens(values: np.ndarray, token_len: int) -> np.ndarray:
    """(n_epochs, n_ch, keep) -> (n_epochs * n_tok, n_ch, token_len) structured tokens."""
    n_ep, n_ch, keep = values.shape
    n_tok = keep // token_len
    v = values[:, :, : n_tok * token_len].reshape(n_ep, n_ch, n_tok, token_len)
    return np.transpose(v, (0, 2, 1, 3)).reshape(n_ep * n_tok, n_ch, token_len)


def epochs_to_channel_tokens(values: np.ndarray, token_len: int) -> list[np.ndarray]:
    """(n_epochs, n_ch, keep) -> list over channels of (n_epochs * n_tok, 1, token_len)."""
    n_ep, n_ch, keep = values.shape
    n_tok = keep // token_len
    v = values[:, :, : n_tok * token_len].reshape(n_ep, n_ch, n_tok, token_len)
    return [v[:, c].reshape(n_ep * n_tok, 1, token_len) for c in range(n_ch)]
