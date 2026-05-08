#!/usr/bin/env python3
"""Build canonical PhysioFM DE archives from local datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import load_trials, save_de_archive, summarize_trials


DEFAULT_INPUTS = {
    "seed_v": "datasets/SEED-V/EEG_DE_features",
    "seed_iv": "datasets/SEED-IV/eeg_feature_smooth",
    "seed": "datasets/SEED/ExtractedFeatures_4s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULT_INPUTS))
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir", default="data/physiofm/de_features")
    parser.add_argument("--feature_key", default="de_LDS")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir or DEFAULT_INPUTS[args.dataset])
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{args.dataset}_{args.feature_key}.npz"
    trials = load_trials(args.dataset, input_dir, feature_key=args.feature_key)
    save_de_archive(trials, output_path)
    summary = summarize_trials(trials)
    print(f"Saved: {output_path}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
