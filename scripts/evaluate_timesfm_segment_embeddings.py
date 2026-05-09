#!/usr/bin/env python3
"""STEP 3: segment-level, structure-preserving Phase 1 evaluation.

For every DE window (segment) we build a TimesFM embedding that preserves
channel-band structure, then classify segments with the PC-SSL subject-dependent
trial-wise splits (directly comparable to PC-SSL, which classifies windows).

Per window t in a trial: for each of the 310 (channel,band) series, take the
length-32 causal context ending at t (edge-padded on the left for early/short
windows, since TimesFM's patch_len=32 is the minimum input), embed it
(last_hidden_state mean over the single patch -> 1280-d). Then build two
features per window:
  * struct: random-projection 1280->K per series, concatenated over 310 series
            (310*K-d; dimension blocks keyed to specific channel-bands).
  * mean310: mean of each series' 1280-d embedding -> 310-d (one scalar per
             channel-band; directly comparable to raw-DE 310-d).

Raw-DE 310-d window features are the ceiling control.

Writes only under results/phase1_timesfm_de/diagnostics/segment/.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import load_de_archive, trial_to_univariate_series
from physiofm.embedding_evaluation import seed_iv_fold_mask, seed_v_fold_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("seg_eval")

SEED = 42
np.random.seed(SEED)

ARCH = {
    "seed_v": "data/physiofm/de_features/seed_v_de_LDS.npz",
    "seed_iv": "data/physiofm/de_features/seed_iv_de_LDS.npz",
}
FOLD = {"seed_v": seed_v_fold_mask, "seed_iv": seed_iv_fold_mask}
CHANCE = {"seed_v": 20.0, "seed_iv": 25.0}
MODEL_ID = "google/timesfm-2.5-200m-transformers"
ADAPTER = "results/phase1_timesfm_de/seed_iv_v_lora"
L = 32
K = 8
NSERIES = 310
OUTDIR = Path("results/phase1_timesfm_de/diagnostics/segment")
OUTDIR.mkdir(parents=True, exist_ok=True)

# fixed random projection 1280 -> K (shared across everything)
_RP = np.random.default_rng(SEED).standard_normal((1280, K)).astype(np.float32) / np.sqrt(1280)


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


def trial_window_contexts(values: np.ndarray) -> np.ndarray:
    """(T,310,L) edge-padded causal contexts for every window."""
    S = trial_to_univariate_series(values)  # (310, T)
    T = S.shape[1]
    out = np.empty((T, NSERIES, L), dtype=np.float32)
    for pos in range(T):
        lo = max(0, pos - L + 1)
        w = S[:, lo:pos + 1]
        if w.shape[1] < L:
            pad = np.repeat(w[:, :1], L - w.shape[1], axis=1)
            w = np.concatenate([pad, w], axis=1)
        out[pos] = w
    return out


def embed_contexts(model, dev, flat: np.ndarray, batch: int = 512) -> np.ndarray:
    """(N,L) -> (N,1280)."""
    import torch

    chunks = []
    with torch.no_grad():
        for i in range(0, len(flat), batch):
            x = torch.tensor(flat[i:i + batch], dtype=torch.float32, device=dev)
            out = model(past_values=x, forecast_context_len=L,
                        truncate_negative=False, force_flip_invariance=False)
            chunks.append(out.last_hidden_state.float().mean(dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def build_features(model, dev, trials, batch=512):
    struct_rows, mean_rows = [], []
    subj, trid, lab = [], [], []
    for i, t in enumerate(trials, 1):
        ctx = trial_window_contexts(t.values)            # (T,310,L)
        T = ctx.shape[0]
        emb = embed_contexts(model, dev, ctx.reshape(-1, L), batch=batch)  # (T*310,1280)
        emb = emb.reshape(T, NSERIES, 1280)
        struct = (emb @ _RP).reshape(T, NSERIES * K)     # (T,310*K)
        mean310 = emb.mean(axis=2)                       # (T,310)
        struct_rows.append(struct.astype(np.float32))
        mean_rows.append(mean310.astype(np.float32))
        subj.append(np.full(T, t.subject)); trid.append(np.full(T, t.trial)); lab.append(np.full(T, t.label))
        if i % 50 == 0:
            LOG.info("featurized %d/%d trials", i, len(trials))
    return (np.concatenate(struct_rows), np.concatenate(mean_rows),
            np.concatenate(subj), np.concatenate(trid), np.concatenate(lab))


def classify(X, subj, trid, lab, ds, trials):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    y = LabelEncoder().fit_transform(lab)
    tsubj = np.array([t.subject for t in trials]); ttrial = np.array([t.trial for t in trials])
    accs, f1s = [], []
    for s in sorted(set(tsubj.tolist())):
        for f in range(3):
            tr_t, te_t = FOLD[ds](tsubj, ttrial, s, f)
            train_trials = set(ttrial[tr_t].tolist()); test_trials = set(ttrial[te_t].tolist())
            a = (subj == s) & np.isin(trid, list(train_trials))
            b = (subj == s) & np.isin(trid, list(test_trials))
            if a.sum() == 0 or b.sum() == 0:
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, class_weight="balanced",
                                                   random_state=SEED, n_jobs=-1))
            clf.fit(X[a], y[a]); p = clf.predict(X[b])
            accs.append(accuracy_score(y[b], p)); f1s.append(f1_score(y[b], p, average="macro", zero_division=0))
    return float(np.mean(accs) * 100), float(np.mean(f1s) * 100), len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["seed_v", "seed_iv"])
    ap.add_argument("--models", nargs="+", default=["frozen", "finetuned"])
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    import torch

    torch.manual_seed(SEED)

    results = []
    for mtag in args.models:
        model, dev = load_model(mtag == "finetuned")
        for ds in args.datasets:
            trials = load_de_archive(ARCH[ds])
            if args.max_trials:
                trials = trials[:args.max_trials]
            LOG.info("=== %s / %s : %d trials ===", mtag, ds, len(trials))
            struct, mean310, subj, trid, lab = build_features(model, dev, trials, batch=args.batch)
            np.savez_compressed(OUTDIR / f"{ds}_{mtag}_segfeat.npz",
                                struct=struct, mean310=mean310, subject=subj, trial=trid, label=lab)
            a_s, f_s, n = classify(struct, subj, trid, lab, ds, trials)
            a_m, f_m, _ = classify(mean310, subj, trid, lab, ds, trials)
            results.append((mtag, ds, n, a_s, f_s, a_m, f_m))
            LOG.info("RESULT %s %s segments=%d | struct%dx%d acc=%.2f f1=%.2f | mean310 acc=%.2f f1=%.2f",
                     mtag, ds, n, NSERIES, K, a_s, f_s, a_m, f_m)
        del model
        torch.cuda.empty_cache()

    print("\n## STEP 3 segment-level structured TimesFM\n")
    print(f"| TimesFM | dataset | folds | struct({NSERIES}x{K}) Acc% | struct F1% | mean310 Acc% | mean310 F1% | chance% |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for mtag, ds, n, a_s, f_s, a_m, f_m in results:
        print(f"| {mtag} | {ds} | {n} | {a_s:.2f} | {f_s:.2f} | {a_m:.2f} | {f_m:.2f} | {CHANCE[ds]:.0f} |")


if __name__ == "__main__":
    main()
