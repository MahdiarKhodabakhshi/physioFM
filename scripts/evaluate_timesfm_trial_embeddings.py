#!/usr/bin/env python3
"""Evaluate TimesFM trial embeddings with subject-dependent emotion splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.embedding_evaluation import classify_trial_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--dataset", required=True, choices=["seed_iv", "seed_v"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = classify_trial_embeddings(args.embeddings, args.output_csv, args.dataset)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
