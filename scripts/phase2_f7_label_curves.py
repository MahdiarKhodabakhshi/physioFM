#!/usr/bin/env python3
"""F7 — Supervised limited-label curves, done properly (decide C2).

Frozen-encoder + the F4 matched head (balanced 2-layer MLP, per-fold validation
early stopping, class weighting), trained on a label-fraction of each fold's
training segments, evaluated on the test segments. Reports 10/50/100% label
curves for three methods on UN-SMOOTHED SEED-IV (where F1 showed dynamics):

  * raw-DE (310-d)              — the non-FM baseline
  * PhysioFM-S PC-pretrained    — frozen encoder embeddings
  * PhysioFM-S random-init      — frozen encoder embeddings

This avoids the documented instability of full-encoder SGD fine-tuning on ~600
labels/fold by using the stable balanced-MLP head; the question is whether a
consistent FM > raw-DE margin appears and **grows as labels shrink** (label
efficiency = the genuine FM win), or whether there is no margin (C2 negative).
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import load_de_archive
from physiofm.phase2_eval import CLASSIFIERS, FOLD_MASK, base_dataset, build_raw_de_segments
from physiofm.structured_data import ARCH, load_standardizer
from scripts.phase2_extract_eval import extract, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("f7_label_curves")

OUTDIR = Path("results/phase2/followup/f7")
SEED = 42


def eval_curve(feats, dataset, frac, classifier="mlp_bal"):
    """Subject-dependent eval with a label fraction of each fold's train set."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    base = base_dataset(dataset)
    fold_fn = FOLD_MASK[base]
    factory = CLASSIFIERS[classifier]
    y = LabelEncoder().fit_transform(feats.label)
    tsubj = np.array([t.subject for t in feats.trials])
    ttrial = np.array([t.trial for t in feats.trials])
    rng = np.random.default_rng(SEED)

    accs, f1s = [], []
    for s in sorted(set(tsubj.tolist())):
        for f in range(3):
            tr_t, te_t = fold_fn(tsubj, ttrial, s, f)
            train_trials = set(ttrial[tr_t].tolist())
            test_trials = set(ttrial[te_t].tolist())
            a = np.where((feats.subject == s) & np.isin(feats.trial, list(train_trials)))[0]
            b = np.where((feats.subject == s) & np.isin(feats.trial, list(test_trials)))[0]
            if a.size == 0 or b.size == 0:
                continue
            if frac < 1.0:
                # stratified per-class subsample, >=1 per present class
                keep = []
                for c in np.unique(y[a]):
                    ci = a[y[a] == c]
                    k = max(1, int(round(ci.size * frac)))
                    keep.append(rng.choice(ci, size=k, replace=False))
                a = np.concatenate(keep)
            if len(np.unique(y[a])) < 2:
                continue
            clf = factory()
            clf.fit(feats.X[a], y[a])
            pred = clf.predict(feats.X[b])
            accs.append(accuracy_score(y[b], pred))
            f1s.append(f1_score(y[b], pred, average="macro", zero_division=0))
    return float(np.mean(accs) * 100), float(np.std(accs) * 100), float(np.mean(f1s) * 100), len(accs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_dir", required=True)
    ap.add_argument("--random_dir", required=True)
    ap.add_argument("--dataset", default="seed_iv_raw")
    ap.add_argument("--label_fracs", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    ap.add_argument("--classifier", default="mlp_bal")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trials = load_de_archive(ARCH[args.dataset])
    pre_model, pre_args = load_model(Path(args.pretrained_dir), device)
    pre_mean, pre_std = load_standardizer(Path(args.pretrained_dir) / "standardizer.npz")
    rnd_model, rnd_args = load_model(Path(args.random_dir), device)
    rnd_mean, rnd_std = load_standardizer(Path(args.random_dir) / "standardizer.npz")

    providers = {
        "raw_de": build_raw_de_segments(trials),
        "physiofm_pretrained": extract(pre_model, pre_args, pre_mean, pre_std, trials, device),
        "physiofm_random_init": extract(rnd_model, rnd_args, rnd_mean, rnd_std, trials, device),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for feat_name, feats in providers.items():
        for frac in args.label_fracs:
            am, asd, fm, n = eval_curve(feats, args.dataset, frac, args.classifier)
            LOG.info("%s frac=%.2f acc=%.2f±%.2f f1=%.2f (folds=%d)", feat_name, frac, am, asd, fm, n)
            rows.append((feat_name, frac, am, asd, fm, n))

    with (OUTDIR / "f7_label_curves.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["features", "label_frac", "acc_mean", "acc_std", "f1_mean", "folds"])
        for feat, frac, am, asd, fm, n in rows:
            w.writerow([feat, f"{frac:.2f}", f"{am:.2f}", f"{asd:.2f}", f"{fm:.2f}", n])

    fracs = args.label_fracs
    lines = [f"# F7 — Limited-label curves on un-smoothed SEED-IV ({args.dataset})\n",
             f"Frozen encoder + {args.classifier} head, subject-dependent folds, seed 42. acc % / macro-F1 %.\n",
             "| Features | " + " | ".join(f"{int(fr*100)}% labels" for fr in fracs) + " |",
             "| --- | " + " | ".join(["---:"] * len(fracs)) + " |"]
    for feat in ["raw_de", "physiofm_pretrained", "physiofm_random_init"]:
        cells = []
        for fr in fracs:
            am, fm = next((a, f) for fe, frac, a, s, f, n in rows if fe == feat and abs(frac - fr) < 1e-6)
            cells.append(f"{am:.2f} / {fm:.2f}")
        lines.append(f"| {feat} | " + " | ".join(cells) + " |")
    lines.append(
        "\n**Read-off (C2 on raw DE).** An FM > raw-DE margin that grows as labels shrink is "
        "the genuine FM win (label efficiency); no margin closes C2 negative even on raw DE."
    )
    (OUTDIR / "f7_label_curves.md").write_text("\n".join(lines) + "\n")
    LOG.info("wrote %s", OUTDIR / "f7_label_curves.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
