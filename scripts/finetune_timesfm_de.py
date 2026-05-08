#!/usr/bin/env python3
"""Fine-tune TimesFM with LoRA on univariate DE channel-band series."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.timesfm_training import finetune_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="google/timesfm-2.5-200m-transformers")
    parser.add_argument("--context_len", type=int, default=32)
    parser.add_argument("--horizon_len", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_samples", type=int, default=2048)
    parser.add_argument("--max_series", type=int)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    finetune_lora(
        archives=args.archive,
        output_dir=args.output_dir,
        model_id=args.model_id,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
        max_series=args.max_series,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
