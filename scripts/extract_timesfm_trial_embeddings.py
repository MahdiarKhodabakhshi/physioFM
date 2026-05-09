#!/usr/bin/env python3
"""Extract one TimesFM embedding per EEG trial from a DE archive."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.timesfm_embeddings import extract_trial_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter_dir")
    parser.add_argument("--model_id", default="google/timesfm-2.5-200m-transformers")
    parser.add_argument("--context_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_trials", type=int)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    extract_trial_embeddings(
        archive=args.archive,
        output_path=args.output,
        adapter_dir=args.adapter_dir,
        model_id=args.model_id,
        context_len=args.context_len,
        batch_size=args.batch_size,
        max_trials=args.max_trials,
    )


if __name__ == "__main__":
    main()
