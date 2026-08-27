#!/usr/bin/env python3
"""Assemble cross-corpus transfer model dirs (EXP-0026).

  full : donor weights verbatim (token dims must match, e.g. 1x64 perch -> perch);
         standardizer = TARGET corpus stats (fine-tuning handles the domain shift).
  trunk: target random-init weights, with the donor's decoder trunk transplanted
         (state_dict keys 'layers.*' and 'out_norm.*'); patch_in / head stay
         target-shaped random. Standardizer = target's.

    python scripts/make_transfer_ckpt.py --mode full  --donor <dir> --target <dir> --out <dir>
    python scripts/make_transfer_ckpt.py --mode trunk --donor <dir> --target <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

TRUNK_PREFIXES = ("layers.", "out_norm.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "trunk"], required=True)
    ap.add_argument("--donor", required=True, help="pretrained source model dir")
    ap.add_argument("--target", required=True, help="target-corpus baseline dir (for shapes + standardizer)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    donor = torch.load(Path(args.donor) / "model.pt", map_location="cpu", weights_only=False)
    target = torch.load(Path(args.target) / "model.pt", map_location="cpu", weights_only=False)

    if args.mode == "full":
        assert donor["n_cb"] == target["n_cb"], \
            f"full transfer needs matching token dims (donor {donor['n_cb']} vs target {target['n_cb']})"
        ck = dict(donor)
    else:
        sd = dict(target["state_dict"])
        moved = 0
        for k, v in donor["state_dict"].items():
            if k.startswith(TRUNK_PREFIXES):
                assert sd[k].shape == v.shape, f"trunk shape mismatch at {k}"
                sd[k] = v
                moved += 1
        assert moved >= 70, f"only {moved} trunk tensors moved — key mismatch?"
        ck = dict(target); ck["state_dict"] = sd
    for a in ("variant", "p_in", "p_out", "hidden", "layers", "heads", "embedder"):
        assert donor["args"][a] == target["args"][a], f"arch mismatch on {a}"

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out / "model.pt")
    shutil.copy(Path(args.target) / "standardizer.npz", out / "standardizer.npz")
    (out / "TRANSFER").write_text(f"mode={args.mode}\ndonor={args.donor}\ntarget={args.target}\n")
    print(f"wrote {out} ({args.mode}; donor n_cb={donor['n_cb']}, target n_cb={target['n_cb']})")


if __name__ == "__main__":
    main()
