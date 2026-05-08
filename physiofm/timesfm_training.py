"""TimesFM fine-tuning utilities for DE channel-band series."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .de import load_de_archive, trial_to_univariate_series


LOGGER = logging.getLogger(__name__)


class RandomWindowDataset:
    def __init__(
        self,
        series: list[np.ndarray],
        context_len: int,
        horizon_len: int,
        num_samples: int,
        seed: int = 42,
    ) -> None:
        self.series = series
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.samples: list[tuple[int, int]] = []
        rng = np.random.default_rng(seed)
        min_len = context_len + horizon_len
        valid = [i for i, item in enumerate(series) if len(item) >= min_len]
        if not valid:
            raise ValueError(f"No series with at least {min_len} points")
        for _ in range(num_samples):
            idx = int(rng.choice(valid))
            max_start = len(series[idx]) - min_len
            start = int(rng.integers(0, max_start + 1))
            self.samples.append((idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        series_idx, start = self.samples[idx]
        values = self.series[series_idx]
        split = start + self.context_len
        end = split + self.horizon_len
        return (
            torch.tensor(values[start:split], dtype=torch.float32),
            torch.tensor(values[split:end], dtype=torch.float32),
        )


def collect_univariate_series(
    archives: list[str | Path],
    min_len: int,
    max_series: int | None = None,
) -> list[np.ndarray]:
    series: list[np.ndarray] = []
    for archive in archives:
        for trial in load_de_archive(archive):
            trial_series = trial_to_univariate_series(trial.values)
            for item in trial_series:
                if len(item) >= min_len and np.isfinite(item).all():
                    series.append(np.asarray(item, dtype=np.float32))
                    if max_series is not None and len(series) >= max_series:
                        return series
    return series


def finetune_lora(
    archives: list[str | Path],
    output_dir: str | Path,
    model_id: str = "google/timesfm-2.5-200m-transformers",
    context_len: int = 32,
    horizon_len: int = 1,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-4,
    num_samples: int = 2048,
    max_series: int | None = None,
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    seed: int = 42,
) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import TimesFm2_5ModelForPrediction

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_len = context_len + horizon_len
    series = collect_univariate_series(archives, min_len=min_len, max_series=max_series)
    LOGGER.info("Collected %d valid univariate DE series", len(series))

    dataset = RandomWindowDataset(series, context_len, horizon_len, num_samples, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    LOGGER.info("Loading %s on %s", model_id, device)
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for context, target in loader:
            context = context.to(device)
            target = target.to(device)
            output = model(
                past_values=context,
                future_values=target,
                forecast_context_len=context_len,
                truncate_negative=False,
                force_flip_invariance=False,
            )
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total += float(loss.item())
            count += 1
        avg = total / max(count, 1)
        LOGGER.info("Epoch %d/%d train_loss=%.6f", epoch, epochs, avg)
        if avg < best_loss:
            best_loss = avg
            model.save_pretrained(output_dir)
            LOGGER.info("Saved adapter to %s", output_dir)
