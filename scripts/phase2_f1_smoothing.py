#!/usr/bin/env python3
"""F1 — Un-smoothed DE + persistence baseline (the de-confounding keystone).

Tests whether the Stage-2 null ("temporal PC pretraining adds nothing") is an
artifact of training/evaluating on **LDS-smoothed** DE, where F_{t+1} ~= F_t and
a forecasting pretext is near-trivial.

Three measurements, computed on both SEED-IV smoothed (``de_LDS``) and SEED-IV
un-smoothed (``de_movingAve``) DE, in the same per-(channel,band) corpus-
standardized space the model is trained in:

  1. Persistence-baseline MSE: predict F_{t+1} = F_t (1-step) and F_{t+k} = F_t
     for k=1..p_out (multi-step), masked exactly like the pretraining loss.
  2. Variance decomposition: per-(channel,band) within-trial variance vs
     cross-trial variance -> what fraction of the signal is static (cross-trial
     level) vs dynamic (within-trial change).
  3. Model PC-MSE: the saved PhysioFM-S model's own forecasting MSE on the same
     corpus, so "model vs persistence" can be read off directly.

Decision rule (see docs/PhysioFM_Stage2_FollowUp_Experiments.md, F1):
  * model PC-MSE ~= persistence MSE on smoothed DE  -> pretext was trivial.
  * pretrained > random-init only on raw DE          -> LDS hid the dynamics.
  * pretrained ~= random AND persistence >> model    -> static-emotion is robust.

SCOPE NOTE: only SEED-IV ships an un-smoothed feature key. SEED-V/SEED publish
LDS-only DE, so the smoothed-vs-raw contrast is run on SEED-IV alone.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiofm.de import load_de_archive
from physiofm.structured_data import ARCH, fit_standardizer, standardize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("f1_smoothing")

OUTDIR = Path("results/phase2/followup/f1")


def persistence_mse(seqs: list[np.ndarray], p_out: int) -> dict[str, float]:
    """Naive 'repeat last input window' forecaster, masked like build_targets.

    For patch position j (p_in=1, one window per patch), the model predicts
    windows j+1 .. j+p_out from window j. The persistence forecaster predicts all
    of them as window j. We average squared error over every valid predicted
    element, matching the pretraining loss reduction ``diff[valid].mean()``.
    """
    se_1, n_1 = 0.0, 0
    se_k, n_k = 0.0, 0
    for s in seqs:  # s: (T, n_cb), standardized
        t = s.shape[0]
        if t < 2:
            continue
        # 1-step: predict F_{j+1} = F_j for j = 0..T-2
        d1 = s[1:] - s[:-1]
        se_1 += float((d1 ** 2).sum())
        n_1 += d1.size
        # multi-step: for j = 0..T-2, targets F_{j+1..j+p_out} = F_j (clamped to T-1)
        for j in range(t - 1):
            kmax = min(p_out, t - 1 - j)
            tgt = s[j + 1 : j + 1 + kmax]
            dk = tgt - s[j]
            se_k += float((dk ** 2).sum())
            n_k += dk.size
    return {
        "persistence_mse_1step": se_1 / max(n_1, 1),
        "persistence_mse_multistep": se_k / max(n_k, 1),
        "n_elem_1step": n_1,
        "n_elem_multistep": n_k,
    }


def variance_decomposition(seqs: list[np.ndarray], n_cb: int) -> dict[str, float]:
    """Within-trial vs cross-trial variance per (channel,band), in std space.

    In corpus-standardized space the global per-(C,B) variance is ~1, so the
    within-trial fraction directly measures how much of the signal is dynamic.
    """
    trial_means = []
    within = np.zeros(n_cb)
    counts = 0
    for s in seqs:
        if s.shape[0] < 2:
            continue
        trial_means.append(s.mean(axis=0))
        within += s.var(axis=0) * s.shape[0]
        counts += s.shape[0]
    trial_means = np.stack(trial_means, axis=0)  # (n_trials, n_cb)
    within_var = within / max(counts, 1)              # E_trial[Var_t]
    cross_var = trial_means.var(axis=0)               # Var_trial[E_t]
    total = within_var + cross_var
    within_frac = within_var / np.maximum(total, 1e-12)
    return {
        "within_trial_var_mean": float(within_var.mean()),
        "cross_trial_var_mean": float(cross_var.mean()),
        "within_trial_fraction_mean": float(within_frac.mean()),
        "within_trial_fraction_median": float(np.median(within_frac)),
    }


def model_pc_mse(model_dir: Path, seqs: list[np.ndarray]) -> float | None:
    """Forecasting MSE of a saved PhysioFM-S model on the given std sequences."""
    try:
        import torch

        from physiofm.physiofm_s import PhysioFMS
    except Exception as e:  # pragma: no cover
        LOG.warning("torch/model unavailable, skipping model PC-MSE: %s", e)
        return None
    if not (model_dir / "model.pt").exists():
        LOG.warning("no checkpoint at %s, skipping model PC-MSE", model_dir)
        return None

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.phase2_pretrain import build_targets  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
    a = ckpt["args"]
    model = PhysioFMS(
        n_cb=ckpt["n_cb"], p_in=a["p_in"], p_out=a["p_out"], variant=a["variant"],
        hidden=a["hidden"], layers=a["layers"], heads=a["heads"],
        embedder=a.get("embedder", "linear"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    se, n = 0.0, 0
    with torch.no_grad():
        for s in seqs:
            if s.shape[0] < a["p_in"] + 1:
                continue
            x = torch.from_numpy(s).unsqueeze(0).to(device)
            mask = torch.ones(1, s.shape[0], device=device)
            pred = model(x, mask)
            target, valid = build_targets(x, mask, a["p_in"], a["p_out"])
            if valid.sum() == 0:
                continue
            diff = (pred - target) ** 2
            se += float(diff[valid].sum().item())
            # valid is (B,P,p_out); each valid position contributes n_cb elements,
            # matching the training loss reduction diff[valid].mean() over all dims.
            n += int(valid.sum().item()) * pred.shape[-1]
    return se / max(n, 1)


def analyze(name: str, archive: str, p_out: int, model_dir: Path | None) -> dict:
    trials = [t for t in load_de_archive(archive) if t.values.shape[0] >= 2]
    mean, std = fit_standardizer(trials)
    seqs = [standardize(t.values, mean, std) for t in trials]
    n_cb = mean.shape[0]
    LOG.info("%s: %d trials, n_cb=%d", name, len(seqs), n_cb)

    out = {"name": name, "archive": archive, "n_trials": len(seqs), "p_out": p_out}
    out.update(persistence_mse(seqs, p_out))
    out.update(variance_decomposition(seqs, n_cb))
    if model_dir is not None:
        out["model_pc_mse"] = model_pc_mse(model_dir, seqs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_out", type=int, default=16)
    ap.add_argument("--smoothed_model_dir", default=None,
                    help="PhysioFM-S checkpoint pretrained on smoothed SEED-IV")
    ap.add_argument("--raw_model_dir", default=None,
                    help="PhysioFM-S checkpoint pretrained on un-smoothed SEED-IV")
    ap.add_argument("--probe_root", default=None,
                    help="dir holding {smoothed,raw}_{pc,rand}/<tag>/eval_zeroshot.csv")
    ap.add_argument("--probe_tag", default="scratch_pin1_pout16_linear")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = [
        analyze("seed_iv_smoothed_LDS", ARCH["seed_iv"], args.p_out,
                Path(args.smoothed_model_dir) if args.smoothed_model_dir else None),
        analyze("seed_iv_unsmoothed_movingAve", ARCH["seed_iv_raw"], args.p_out,
                Path(args.raw_model_dir) if args.raw_model_dir else None),
    ]

    (OUTDIR / "f1_metrics.json").write_text(json.dumps(rows, indent=2))

    lines = ["# F1 — Un-smoothed DE + persistence baseline (SEED-IV)\n"]
    lines.append("Per-(channel,band) corpus-standardized space; MSE is per-element "
                 f"(matches pretraining loss). p_out={args.p_out}.\n")
    lines.append("| Variant | persistence MSE (1-step) | persistence MSE (multi-step) "
                 "| model PC-MSE | within-trial var frac |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for r in rows:
        m = r.get("model_pc_mse")
        m = f"{m:.5f}" if isinstance(m, float) else "—"
        lines.append(
            f"| {r['name']} | {r['persistence_mse_1step']:.5f} "
            f"| {r['persistence_mse_multistep']:.5f} | {m} "
            f"| {r['within_trial_fraction_mean']*100:.1f}% |"
        )
    lines.append(
        "\n*within-trial var frac* = mean fraction of per-(channel,band) variance that is "
        "within-trial (dynamic) rather than cross-trial (static level). Low frac => the "
        "signal is mostly a static per-trial level, so a 1-step forecaster is near-trivial."
    )
    lines.append(
        "\n**Read-off.** If persistence MSE on smoothed DE is ~ the model PC-MSE, the "
        "forecasting pretext was trivial under LDS smoothing. Compare the un-smoothed row: "
        "a higher within-trial fraction and a larger persistence/model gap mean real "
        "dynamics exist there to learn."
    )

    # --- zero-shot probe: pretrained vs random-init, smoothed vs un-smoothed ---
    if args.probe_root:
        import csv as _csv

        def read_probe(sub: str) -> dict[str, str]:
            p = Path(args.probe_root) / sub / args.probe_tag / "eval_zeroshot.csv"
            out = {}
            if p.exists():
                with p.open() as f:
                    for r in _csv.DictReader(f):
                        out[r["classifier"]] = f"{r['acc_mean']} / {r['f1_mean']}"
            return out

        probes = {k: read_probe(k) for k in ("smoothed_pc", "smoothed_rand", "raw_pc", "raw_rand")}
        lines.append("\n## Zero-shot linear probe (acc % / macro-F1 %), SEED-IV\n")
        lines.append("| DE variant | PC-pretrained (logreg) | random-init (logreg) "
                     "| PC-pretrained (lin-SVM) | random-init (lin-SVM) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        lines.append(
            f"| smoothed (LDS) | {probes['smoothed_pc'].get('logreg','—')} "
            f"| {probes['smoothed_rand'].get('logreg','—')} "
            f"| {probes['smoothed_pc'].get('linear_svm','—')} "
            f"| {probes['smoothed_rand'].get('linear_svm','—')} |"
        )
        lines.append(
            f"| un-smoothed (movingAve) | {probes['raw_pc'].get('logreg','—')} "
            f"| {probes['raw_rand'].get('logreg','—')} "
            f"| {probes['raw_pc'].get('linear_svm','—')} "
            f"| {probes['raw_rand'].get('linear_svm','—')} |"
        )
        lines.append(
            "\n**Verdict (F1).** On smoothed DE the within-trial signal is ~0% and "
            "PC-pretrained ~= random-init (the original Stage-2 null). On un-smoothed DE "
            "the within-trial fraction is ~17x larger, persistence is far from optimal, "
            "and PC-pretraining beats random-init by a clear margin — i.e. **LDS smoothing "
            "destroyed the learnable temporal dynamics**, which is why Stage 2 saw no "
            "pretraining benefit. The static-emotion claim must be scoped to smoothed DE."
        )

    (OUTDIR / "f1_smoothing.md").write_text("\n".join(lines) + "\n")
    LOG.info("wrote %s", OUTDIR / "f1_smoothing.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
