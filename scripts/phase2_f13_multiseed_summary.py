#!/usr/bin/env python3
"""Aggregate the F13 multi-seed sleep runs and run paired significance tests.

Reads `f13_sleep_seed*.csv` (per-seed fold-mean) and `f13_sleep_seed*_perfold.csv`
(per-fold) written by run_f13_multiseed.sh, then:
  * reports each feature set's accuracy as mean ± std ACROSS SEEDS, and
  * runs paired tests (physiofm_pc vs raw_de, physiofm_pc vs physiofm_rand) on the
    per-fold values (folds are matched; pc averaged over seeds per fold).

    python scripts/phase2_f13_multiseed_summary.py --dir results/phase3/f13/multiseed
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_perfold(path):
    """-> {feature: {fold: acc}} (acc in %)."""
    out = defaultdict(dict)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["classifier"] != "logreg":
                continue
            out[row["features"]][int(row["fold"])] = float(row["acc"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phase3/f13/multiseed")
    ap.add_argument("--include_seed42", action="store_true",
                    help="also fold in the original seed-42 run at ../f13_sleep_perfold.csv")
    args = ap.parse_args()
    d = Path(args.dir)

    # feature -> seed -> {fold: acc}
    per = defaultdict(dict)
    seeds = []
    for p in sorted(glob.glob(str(d / "f13_sleep_seed*_perfold.csv"))):
        m = re.search(r"seed(\d+)_perfold", p)
        seed = int(m.group(1))
        seeds.append(seed)
        for feat, folds in _read_perfold(p).items():
            per[feat][seed] = folds
    if args.include_seed42:
        p42 = d.parent / "f13_sleep_perfold.csv"
        if p42.exists():
            seeds.append(42)
            for feat, folds in _read_perfold(p42).items():
                per[feat][42] = folds
    if not per:
        raise SystemExit(f"no per-fold CSVs found under {d}")
    seeds = sorted(set(seeds))
    print(f"seeds: {seeds}")

    # per-seed fold-mean, then mean +/- std across seeds
    print("\n== accuracy, mean +/- std across seeds ==")
    feat_seedmeans = {}
    for feat, bys in per.items():
        seedmeans = np.array([np.mean(list(bys[s].values())) for s in sorted(bys)])
        feat_seedmeans[feat] = seedmeans
        print(f"  {feat:14s} {seedmeans.mean():.2f} +/- {seedmeans.std():.2f}  "
              f"(per-seed: {', '.join(f'{v:.1f}' for v in seedmeans)})")

    # paired tests on per-fold values (pc averaged over seeds per fold)
    from scipy import stats

    def perfold_seedavg(feat):
        bys = per[feat]
        folds = sorted(next(iter(bys.values())).keys())
        return np.array([np.mean([bys[s][f] for s in bys]) for f in folds])

    def paired(a_name, b_name):
        if a_name not in per or b_name not in per:
            return
        a, b = perfold_seedavg(a_name), perfold_seedavg(b_name)
        diff = a - b
        t, p_t = stats.ttest_rel(a, b)
        try:
            w, p_w = stats.wilcoxon(a, b)
        except ValueError:
            p_w = float("nan")
        wins = int((diff > 0).sum())
        print(f"  {a_name} vs {b_name}: mean diff {diff.mean():+.2f} pts "
              f"(per-fold {', '.join(f'{x:+.1f}' for x in diff)}); "
              f"paired t p={p_t:.4f}, wilcoxon p={p_w:.4f}, {a_name} wins {wins}/{len(diff)} folds")

    print("\n== paired per-fold tests (seed-averaged) ==")
    paired("physiofm_pc", "raw_de")
    paired("physiofm_pc", "physiofm_rand")

    # markdown summary
    out_md = d / "f13_multiseed_summary.md"
    lines = ["# F13 multi-seed summary\n", f"Seeds: {seeds}\n",
             "\n| Feature | acc mean±std (across seeds) |", "| --- | ---: |"]
    for feat, sm in feat_seedmeans.items():
        lines.append(f"| {feat} | {sm.mean():.2f} ± {sm.std():.2f} |")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    main()
