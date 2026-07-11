#!/usr/bin/env python3
"""F5 — Larger input context (p_in) on un-smoothed SEED-IV.

Sweeps p_in x p_out and reports, for each config, the trial-level sequence-readout
accuracy (GRU + last-state) and the per-window linear probe, for the PC-pretrained
vs the matched random-init encoder. The metric of interest is the
**pretrained-minus-random gap** as a function of context length.

Decision rule (F5): a gap that grows with context (esp. via the temporal readout)
=> dynamics matter and more context helps; a flat gap => context is not the lever.
Pre-registered expectation: flat on smoothed DE, possibly positive on raw DE.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import load_de_archive
from physiofm.phase2_eval import subject_dependent_eval
from physiofm.structured_data import ARCH, load_standardizer
from scripts.phase2_extract_eval import extract, load_model
from scripts.phase2_f2_sequence_readout import eval_readout, extract_sequences

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("f5_context")

OUTDIR = Path("results/phase2/followup/f5")


def probe_logreg(model, margs, mean, std, trials, dataset, device) -> float:
    feats = extract(model, margs, mean, std, trials, device)
    return subject_dependent_eval(feats, dataset, "logreg")["accuracy_mean"] * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(OUTDIR))
    ap.add_argument("--dataset", default="seed_iv_raw")
    ap.add_argument("--p_in", nargs="+", type=int, default=[1, 4, 8])
    ap.add_argument("--p_out", nargs="+", type=int, default=[1, 16])
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trials = load_de_archive(ARCH[args.dataset])
    root = Path(args.root)

    rows = []
    for pin in args.p_in:
        for pout in args.p_out:
            tag = f"scratch_pin{pin}_pout{pout}_linear"
            pc_dir, rnd_dir = root / "pc" / tag, root / "rand" / tag
            if not (pc_dir / "model.pt").exists() or not (rnd_dir / "model.pt").exists():
                LOG.warning("missing models for %s, skipping", tag)
                continue
            res = {"p_in": pin, "p_out": pout}
            for which, mdir in (("pc", pc_dir), ("rand", rnd_dir)):
                model, margs = load_model(mdir, device)
                mean, std = load_standardizer(mdir / "standardizer.npz")
                res[f"{which}_logreg"] = probe_logreg(model, margs, mean, std, trials, args.dataset, device)
                seqs, subj, trid, lab = extract_sequences(model, margs, mean, std, trials, device)
                a_last, _, _, _ = eval_readout("last", seqs, subj, trid, lab, args.dataset, device)
                a_gru, _, _, _ = eval_readout("gru", seqs, subj, trid, lab, args.dataset, device)
                res[f"{which}_last"] = a_last
                res[f"{which}_gru"] = a_gru
            LOG.info("p_in=%d p_out=%d | gru PC=%.2f rand=%.2f gap=%.2f | logreg PC=%.2f rand=%.2f",
                     pin, pout, res["pc_gru"], res["rand_gru"], res["pc_gru"] - res["rand_gru"],
                     res["pc_logreg"], res["rand_logreg"])
            rows.append(res)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "f5_context.csv").open("w", newline="") as f:
        w = csv.writer(f)
        cols = ["p_in", "p_out", "pc_logreg", "rand_logreg", "pc_last", "rand_last", "pc_gru", "rand_gru"]
        w.writerow(cols + ["gru_gap"])
        for r in rows:
            w.writerow([r[c] if c in ("p_in", "p_out") else f"{r[c]:.2f}" for c in cols]
                       + [f"{r['pc_gru'] - r['rand_gru']:.2f}"])

    lines = [f"# F5 — Input-context sweep on un-smoothed SEED-IV ({args.dataset})\n",
             "Trial-level GRU readout (order-aware) and per-window logreg probe. "
             "acc %. gap = PC − random (GRU).\n",
             "| p_in | p_out | GRU PC | GRU rand | **GRU gap** | last PC | last rand | logreg PC | logreg rand |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(
            f"| {r['p_in']} | {r['p_out']} | {r['pc_gru']:.2f} | {r['rand_gru']:.2f} "
            f"| **{r['pc_gru'] - r['rand_gru']:.2f}** | {r['pc_last']:.2f} | {r['rand_last']:.2f} "
            f"| {r['pc_logreg']:.2f} | {r['rand_logreg']:.2f} |"
        )
    lines.append(
        "\n**Read-off.** A pretrained−random GRU gap that grows with `p_in` means longer "
        "input context lets the model exploit dynamics; a flat gap means context is not the "
        "lever even where dynamics exist."
    )
    (OUTDIR / "f5_context.md").write_text("\n".join(lines) + "\n")
    LOG.info("wrote %s", OUTDIR / "f5_context.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
