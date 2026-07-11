#!/usr/bin/env python3
"""F2 — Sequence-level / order-aware readout.

A per-window probe with trial-constant labels cannot reward temporal modeling.
This classifies a whole **trial** from the frozen encoder's per-window hidden
states with three readouts:

  (a) last   : the accumulated causal hidden state at the last window (logreg).
  (b) gru    : an order-respecting GRU pool over the window-embedding sequence.
  (c) gru_shuf: the same GRU but with window order shuffled (the order control).

Compared pretrained vs random-init under (a)/(b), and (b) vs (c). Trial-level,
PC-SSL subject-dependent folds (train/test *trials* per subject), seed 42.

Decision rule (F2):
  * pretrained > random under (a)/(b) but tie under the per-window probe
    -> temporal pretraining helps, only a temporal readout exposes it.
  * (b) ~= (c) (shuffling does not hurt) -> the signal is order-invariant/static;
    the negative result holds even with a fair readout.
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
from physiofm.phase2_eval import base_dataset, CHANCE, FOLD_MASK, make_logreg
from physiofm.structured_data import ARCH, load_standardizer, standardize
from scripts.phase2_extract_eval import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("f2_sequence")

OUTDIR = Path("results/phase2/followup/f2")
SEED = 42


def extract_sequences(model, margs, mean, std, trials, device):
    """Return per-trial (P,d) embedding sequences + subject/trial/label arrays."""
    import torch

    seqs, subj, trid, lab = [], [], [], []
    with torch.no_grad():
        for t in trials:
            if t.label is None or t.values.shape[0] < 1:
                continue
            x = torch.from_numpy(standardize(t.values, mean, std)).unsqueeze(0).to(device)
            h = model.encode(x).squeeze(0).float().cpu().numpy()  # (P, d)
            seqs.append(h)
            subj.append(t.subject)
            trid.append(t.trial)
            lab.append(t.label)
    return seqs, np.array(subj), np.array(trid), np.array(lab)


class _GRUHead:
    """1-layer GRU + linear readout over a window-embedding sequence."""

    def __init__(self, d, n_cls, hidden=64, epochs=40, lr=1e-3, shuffle_time=False, device="cpu"):
        import torch.nn as nn

        self.device = device
        self.shuffle_time = shuffle_time
        self.epochs = epochs
        self.lr = lr
        self.net = nn.ModuleDict({
            "gru": nn.GRU(d, hidden, batch_first=True),
            "fc": nn.Linear(hidden, n_cls),
        }).to(device)

    def _forward(self, seqs, rng=None):
        import torch

        outs = []
        for s in seqs:
            if self.shuffle_time and rng is not None:
                s = s[rng.permutation(s.shape[0])]
            x = torch.from_numpy(np.ascontiguousarray(s)).unsqueeze(0).to(self.device)
            _, hN = self.net["gru"](x)
            outs.append(self.net["fc"](hN[-1]))
        return torch.cat(outs, dim=0)

    def fit(self, seqs, y):
        import torch
        import torch.nn.functional as F

        rng = np.random.default_rng(SEED)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        yt = torch.from_numpy(y.astype(np.int64)).to(self.device)
        cw = torch.from_numpy(
            (len(y) / (len(np.unique(y)) * np.bincount(y, minlength=int(y.max() + 1)) + 1e-6)).astype(np.float32)
        ).to(self.device)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self._forward(seqs, rng)
            loss = F.cross_entropy(logits, yt, weight=cw)
            loss.backward()
            opt.step()
        return self

    def predict(self, seqs):
        import torch

        self.net.eval()
        rng = np.random.default_rng(SEED + 1)
        with torch.no_grad():
            return self._forward(seqs, rng).argmax(-1).cpu().numpy()


def eval_readout(readout, seqs, subj, trid, lab, dataset, device):
    """Per subject×fold trial-level accuracy/F1 for a given readout name."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    base = base_dataset(dataset)
    fold_fn = FOLD_MASK[base]
    y = LabelEncoder().fit_transform(lab)
    n_cls = len(np.unique(y))
    d = seqs[0].shape[1]

    accs, f1s = [], []
    for s in sorted(set(subj.tolist())):
        for f in range(3):
            tr_m, te_m = fold_fn(subj, trid, s, f)
            tr = np.where(tr_m)[0]
            te = np.where(te_m)[0]
            if tr.size == 0 or te.size == 0:
                continue
            tr_seqs = [seqs[i] for i in tr]
            te_seqs = [seqs[i] for i in te]
            if readout == "last":
                Xtr = np.stack([s_[-1] for s_ in tr_seqs])
                Xte = np.stack([s_[-1] for s_ in te_seqs])
                clf = make_logreg()
                clf.fit(Xtr, y[tr])
                pred = clf.predict(Xte)
            else:  # gru / gru_shuf
                import torch

                torch.manual_seed(SEED)
                head = _GRUHead(d, n_cls, shuffle_time=(readout == "gru_shuf"), device=device)
                head.fit(tr_seqs, y[tr])
                pred = head.predict(te_seqs)
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return float(np.mean(accs) * 100), float(np.std(accs) * 100), float(np.mean(f1s) * 100), len(accs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_dir", required=True)
    ap.add_argument("--random_dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=["seed_v", "seed_iv"])
    ap.add_argument("--readouts", nargs="+", default=["last", "gru", "gru_shuf"])
    ap.add_argument("--tag", default="smoothed")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {
        "pretrained": load_model(Path(args.pretrained_dir), device),
        "random_init": load_model(Path(args.random_dir), device),
    }
    stds = {
        "pretrained": load_standardizer(Path(args.pretrained_dir) / "standardizer.npz"),
        "random_init": load_standardizer(Path(args.random_dir) / "standardizer.npz"),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        trials = load_de_archive(ARCH[ds])
        for which, (model, margs) in models.items():
            mean, std = stds[which]
            seqs, subj, trid, lab = extract_sequences(model, margs, mean, std, trials, device)
            for ro in args.readouts:
                am, asd, fm, n = eval_readout(ro, seqs, subj, trid, lab, ds, device)
                LOG.info("RESULT %s %s %s acc=%.2f±%.2f f1=%.2f (trials, folds=%d, chance=%.1f)",
                         ds, which, ro, am, asd, fm, n, CHANCE[base_dataset(ds)])
                rows.append((ds, which, ro, am, asd, fm, n))

    with (OUTDIR / f"f2_sequence_{args.tag}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "encoder", "readout", "acc_mean", "acc_std", "f1_mean", "folds"])
        for ds, which, ro, am, asd, fm, n in rows:
            w.writerow([ds, which, ro, f"{am:.2f}", f"{asd:.2f}", f"{fm:.2f}", n])

    lines = [f"# F2 — Sequence-level / order-aware readout ({args.tag} DE)\n",
             "Trial-level, subject-dependent folds, seed 42. acc % / macro-F1 %.\n"]
    for ds in args.datasets:
        lines.append(f"## {ds} (chance {CHANCE[base_dataset(ds)]:.0f}%)\n")
        lines.append("| Readout | pretrained | random-init |")
        lines.append("| --- | ---: | ---: |")
        for ro in args.readouts:
            p = next((am, fm) for d, w_, r, am, asd, fm, n in rows if d == ds and w_ == "pretrained" and r == ro)
            q = next((am, fm) for d, w_, r, am, asd, fm, n in rows if d == ds and w_ == "random_init" and r == ro)
            lines.append(f"| {ro} | {p[0]:.2f} / {p[1]:.2f} | {q[0]:.2f} / {q[1]:.2f} |")
        lines.append("")
    out_md = OUTDIR / f"f2_sequence_{args.tag}.md"
    out_md.write_text("\n".join(lines) + "\n")
    LOG.info("wrote %s", out_md)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
