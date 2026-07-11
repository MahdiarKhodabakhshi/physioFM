#!/usr/bin/env python3
"""F6 — Scale check on un-smoothed SEED-IV.

Tracks the pretrained-minus-random gap as model size grows. If the gap stays ~0
the null is fundamental for this task; if it opens, the FM bet needs scale.
Run only on un-smoothed DE (post-F1) so there is signal to scale into.
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
LOG = logging.getLogger("f6_scale")

OUTDIR = Path("results/phase2/followup/f6")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(OUTDIR))
    ap.add_argument("--dataset", default="seed_iv_raw")
    ap.add_argument("--points", nargs="+", default=["128:4", "256:6", "512:8"])
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trials = load_de_archive(ARCH[args.dataset])
    root = Path(args.root)
    tag = "scratch_pin1_pout16_linear"

    rows = []
    for pt in args.points:
        h, l = pt.split(":")
        base = root / f"h{h}_l{l}"
        pc_dir, rnd_dir = base / "pc" / tag, base / "rand" / tag
        if not (pc_dir / "model.pt").exists():
            LOG.warning("missing %s, skipping", pc_dir)
            continue
        r = {"hidden": int(h), "layers": int(l)}
        for which, mdir in (("pc", pc_dir), ("rand", rnd_dir)):
            model, margs = load_model(mdir, device)
            n_params = sum(p.numel() for p in model.parameters())
            mean, std = load_standardizer(mdir / "standardizer.npz")
            feats = extract(model, margs, mean, std, trials, device)
            r[f"{which}_logreg"] = subject_dependent_eval(feats, args.dataset, "logreg")["accuracy_mean"] * 100
            seqs, subj, trid, lab = extract_sequences(model, margs, mean, std, trials, device)
            r[f"{which}_gru"], _, _, _ = eval_readout("gru", seqs, subj, trid, lab, args.dataset, device)
            r[f"{which}_params"] = n_params
        LOG.info("h=%s l=%s params=%.2fM | gru PC=%.2f rand=%.2f gap=%.2f | logreg gap=%.2f",
                 h, l, r["pc_params"] / 1e6, r["pc_gru"], r["rand_gru"], r["pc_gru"] - r["rand_gru"],
                 r["pc_logreg"] - r["rand_logreg"])
        rows.append(r)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "f6_scale.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hidden", "layers", "params_M", "pc_gru", "rand_gru", "gru_gap", "pc_logreg", "rand_logreg", "logreg_gap"])
        for r in rows:
            w.writerow([r["hidden"], r["layers"], f"{r['pc_params']/1e6:.2f}", f"{r['pc_gru']:.2f}",
                        f"{r['rand_gru']:.2f}", f"{r['pc_gru']-r['rand_gru']:.2f}", f"{r['pc_logreg']:.2f}",
                        f"{r['rand_logreg']:.2f}", f"{r['pc_logreg']-r['rand_logreg']:.2f}"])

    lines = [f"# F6 — Scale check on un-smoothed SEED-IV ({args.dataset})\n",
             "Fixed p_in=1, p_out=16. Tracks pretrained−random gap vs size. acc %.\n",
             "| hidden | layers | params (M) | GRU PC | GRU rand | **GRU gap** | logreg gap |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['hidden']} | {r['layers']} | {r['pc_params']/1e6:.2f} | {r['pc_gru']:.2f} "
                     f"| {r['rand_gru']:.2f} | **{r['pc_gru']-r['rand_gru']:.2f}** | {r['pc_logreg']-r['rand_logreg']:.2f} |")
    lines.append(
        "\n**Read-off.** Gap ≈ 0 and flat as size grows -> the null is fundamental for this "
        "task; gap opens with scale -> the FM bet needs scale, not abandonment."
    )
    (OUTDIR / "f6_scale.md").write_text("\n".join(lines) + "\n")
    LOG.info("wrote %s", OUTDIR / "f6_scale.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
