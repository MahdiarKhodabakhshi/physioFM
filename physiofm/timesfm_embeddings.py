"""TimesFM embedding extraction for DE EEG trials."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .de import load_de_archive, trial_to_univariate_series


LOGGER = logging.getLogger(__name__)


def _load_prediction_model(model_id: str, adapter_dir: str | Path | None = None):
    import torch
    from transformers import TimesFm2_5ModelForPrediction

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, device


def extract_trial_embeddings(
    archive: str | Path,
    output_path: str | Path,
    adapter_dir: str | Path | None = None,
    model_id: str = "google/timesfm-2.5-200m-transformers",
    context_len: int = 32,
    batch_size: int = 64,
    max_trials: int | None = None,
) -> None:
    import torch

    model, device = _load_prediction_model(model_id, adapter_dir)
    trials = load_de_archive(archive)
    if max_trials is not None:
        trials = trials[:max_trials]

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for idx, trial in enumerate(trials, start=1):
            series = trial_to_univariate_series(trial.values)
            vectors = []
            for start in range(0, len(series), batch_size):
                batch = [
                    torch.tensor(item, dtype=torch.float32, device=device)
                    for item in series[start : start + batch_size]
                ]
                output = model(
                    past_values=batch,
                    forecast_context_len=context_len,
                    truncate_negative=False,
                    force_flip_invariance=False,
                )
                hidden = output.last_hidden_state.float().mean(dim=1)
                vectors.append(hidden.cpu().numpy())
            trial_embedding = np.concatenate(vectors, axis=0).mean(axis=0)
            embeddings.append(trial_embedding.astype(np.float32))
            if idx % 25 == 0:
                LOGGER.info("Embedded %d/%d trials", idx, len(trials))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.stack(embeddings, axis=0),
        dataset=np.array([trial.dataset for trial in trials]),
        subject=np.array([trial.subject for trial in trials], dtype=np.int64),
        session=np.array([trial.session for trial in trials], dtype=np.int64),
        trial=np.array([trial.trial for trial in trials], dtype=np.int64),
        label=np.array([trial.label if trial.label is not None else -999 for trial in trials], dtype=np.int64),
    )
