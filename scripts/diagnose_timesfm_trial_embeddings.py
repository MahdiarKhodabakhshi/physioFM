#!/usr/bin/env python3
"""GPU diagnostics for Phase 1: frozen vs fine-tuned, input normalization, and
whether per-series TimesFM embeddings retain channel-band structure.

Writes new artifacts under results/phase1_timesfm_de/diagnostics/ only.
"""
from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import load_de_archive, trial_to_univariate_series
from physiofm.embedding_evaluation import seed_iv_fold_mask, seed_v_fold_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("phase1_diag")

SEED = 42
np.random.seed(SEED)

ARCH = {
    "seed_v": "data/physiofm/de_features/seed_v_de_LDS.npz",
    "seed_iv": "data/physiofm/de_features/seed_iv_de_LDS.npz",
}
FOLD = {"seed_v": seed_v_fold_mask, "seed_iv": seed_iv_fold_mask}
CHANCE = {"seed_v": 20.0, "seed_iv": 25.0}
MODEL_ID = "google/timesfm-2.5-200m-transformers"
CTX = 32
ADAPTER = "results/phase1_timesfm_de/seed_iv_v_lora"
OUTDIR = Path("results/phase1_timesfm_de/diagnostics")
OUTDIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(ARCH), default=sorted(ARCH))
    parser.add_argument("--models", nargs="+", choices=["frozen", "finetuned"], default=["frozen", "finetuned"])
    return parser.parse_args()


def load_model(use_adapter: bool):
    import torch
    from transformers import TimesFm2_5ModelForPrediction
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    m = TimesFm2_5ModelForPrediction.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map=dev)
    if use_adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, ADAPTER)
    m.eval()
    return m, dev


def embed_trials(model, dev, trials, normalize: bool, batch_size: int = 64):
    """Return (mean310, struct310) per trial.

    mean310: mean over the 310 per-series 1280-d embeddings -> (N,1280)  [current pipeline]
    struct310: per-series scalar = mean of its 1280-d embedding -> (N,310) [structure-preserving]
    """
    import torch

    mean_vecs, struct_vecs = [], []
    with torch.no_grad():
        for i, t in enumerate(trials, 1):
            series = trial_to_univariate_series(t.values)  # (310, T)
            per_series = []
            for s in range(0, len(series), batch_size):
                chunk = series[s:s + batch_size]
                batch = []
                for item in chunk:
                    arr = np.asarray(item, dtype=np.float32)
                    if normalize:
                        mu, sd = arr.mean(), arr.std()
                        arr = (arr - mu) / (sd + 1e-6)
                    batch.append(torch.tensor(arr, dtype=torch.float32, device=dev))
                out = model(past_values=batch, forecast_context_len=CTX,
                            truncate_negative=False, force_flip_invariance=False)
                h = out.last_hidden_state.float().mean(dim=1)  # (b,1280)
                per_series.append(h.cpu().numpy())
            emb = np.concatenate(per_series, axis=0)  # (310,1280)
            mean_vecs.append(emb.mean(axis=0).astype(np.float32))         # (1280,)
            struct_vecs.append(emb.mean(axis=1).astype(np.float32))       # (310,)
            if i % 100 == 0:
                LOG.info("embedded %d/%d", i, len(trials))
    return np.stack(mean_vecs), np.stack(struct_vecs)


def classify(X, subject, trial, label, ds):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    y = LabelEncoder().fit_transform(label)
    accs, f1s = [], []
    for s in sorted(set(subject.tolist())):
        for f in range(3):
            a, b = FOLD[ds](subject, trial, s, f)
            if a.sum() == 0 or b.sum() == 0:
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
            clf.fit(X[a], y[a]); p = clf.predict(X[b])
            accs.append(accuracy_score(y[b], p)); f1s.append(f1_score(y[b], p, average="macro", zero_division=0))
    return float(np.mean(accs) * 100), float(np.mean(f1s) * 100), len(accs)


def main():
    args = parse_args()

    import torch

    torch.manual_seed(SEED)
    rows = []
    for tag in args.models:
        use_adapter = tag == "finetuned"
        model, dev = load_model(use_adapter)
        for normalize in (False, True):
            for ds in args.datasets:
                path = ARCH[ds]
                trials = load_de_archive(path)
                subject = np.array([t.subject for t in trials])
                trial = np.array([t.trial for t in trials])
                label = np.array([t.label for t in trials])
                mean310, struct310 = embed_trials(model, dev, trials, normalize=normalize)
                np.savez_compressed(OUTDIR / f"{ds}_{tag}_norm{int(normalize)}.npz",
                                    mean1280=mean310, struct310=struct310,
                                    subject=subject, trial=trial, label=label)
                a1, f1_, n = classify(mean310, subject, trial, label, ds)
                a2, f2_, _ = classify(struct310, subject, trial, label, ds)
                rows.append((tag, "raw" if not normalize else "z-scored", ds,
                             a1, f1_, a2, f2_, n))
                LOG.info("RESULT %s norm=%d %s | mean1280 acc=%.2f | struct310 acc=%.2f",
                         tag, int(normalize), ds, a1, a2)
        del model
        torch.cuda.empty_cache()

    print("\n## (b)+(e) GPU diagnostics table\n")
    print("| TimesFM | input | dataset | mean-1280 Acc% | mean-1280 F1% | struct-310 Acc% | struct-310 F1% | chance% |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for tag, inp, ds, a1, f1_, a2, f2_, n in rows:
        print(f"| {tag} | {inp} | {ds} | {a1:.2f} | {f1_:.2f} | {a2:.2f} | {f2_:.2f} | {CHANCE[ds]:.0f} |")


if __name__ == "__main__":
    main()
