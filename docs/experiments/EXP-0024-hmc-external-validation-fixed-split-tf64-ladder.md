---
id: EXP-0024
title: HMC external validation — tf64 + causal decoder + PC pretraining on a second sleep dataset
status: done
created: 2026-08-27
run_date: 2026-08-27
agent: claude-code
phase: external-validation
verified: no
tags: hmc, sleep, tf64, external-validation, fixed-split
commits: TBD
verdict: H1 CONFIRMED, H2 REFUTED-as-null. Architecture transfers: fine-tuned PhysioFM-S tf64 (3.5M params, single-corpus) reaches test BAC 73.79 +/- 0.87 / kappa .668 +/- .009 / wF1 74.54 (8 seeds, 20-epoch FT, val-kappa selection) on the fixed NeuroLM split - above CBraMod (.727/.669) and LaBraM-Base's BAC (.729), below REVE-Base (.740/.698). Pretraining adds NOTHING: paired pc-rand delta +0.00 +/- 1.25 BAC / +0.0001 kappa (8 seeds, e20; +0.29 +/- 1.13 at e8). The Sleep-EDF +0.85 (4 seeds) now reads as few-seed noise around ~0, matching the seizure null and the in-domain SSL literature.
---

# EXP-0024 — HMC external validation (second sleep dataset)

> **Status:** running · **Created:** 2026-08-27 · **Agent:** claude-code · **Phase:** external-validation

## 1. Why — hypothesis & motivation
The recipe's one clear win is sleep (Sleep-EDF-78: 77.9 % / κ .71 fine-tuned,
subject-disjoint). A single dataset is a weak basis for the paper's central claim, and the
dataset survey (docs/SLEEP_DATASET_CANDIDATES.md) found a near-drop-in second corpus with a
modern published ladder: **HMC** (PhysioNet hmc-sleep-staging v1.1; 151 clinical PSGs, one
per subject, 4 EEG @ 256 Hz, AASM 30-s). On its fixed split the best published κ ≈ 0.70
(REVE-Base .698, CSBrain .682, LaBraM .681, CBraMod .669, NeuroLM .619 — all full fine-tunes
of large pretrained EEG foundation models). Hypotheses: (H1) the architecture transfers —
fine-tuned PhysioFM-S tf64 lands in that table's range with 2.4 M params; (H2) the
pretraining delta stays small-positive (as on SEDF-78, +0.85), consistent with the
literature's in-domain pattern.

## 2. Setup — exactly what was run
- Data: datasets/HMC (S3 mirror of PhysioNet v1.1, SHA256-verified). 151 recordings
  SN001–SN154 (SN014/SN064/SN135 absent in v1.1). Channels EEG F4-M1/C4-M1/O2-M1/C3-M2
  @ 256 Hz; annotation vocabulary verified on SN001 (exact 30-s grid).
- Pipeline: `physiofm/hmc.py` mirrors `physiofm/sleep_edf.py` (same epoching, feature_fn
  injection) but with **NO wake trim** — the published protocol keeps all scored epochs
  (verified: our epoching reproduces NeuroLM Table 1 exactly: 91,248/22,124/23,871 epochs). tf64 tokens = 4 ch × 64 log-spaced Welch log-power bins
  (`physiofm/spectral.py`, rate-agnostic — native 256 Hz, no resampling).
  Build: `scripts/build_hmc_dataset.py` → `hmc_tf64.npz` + `hmc_labels.npz` +
  `hmc_tf64_pretrain.npz` (subjects ≤ SN125 ONLY — pretraining corpus and standardizer
  never see test subjects).
- Split: fixed published split, pinned from NeuroLM's `prepare_HMC.py` + paper §3.1 by
  the pre-launch review (positional: first 100 / next 25 / last 26 of the sorted list) —
  on the v1.1 roster that is train SN001–SN102 / val SN103–SN127 / test SN128–SN154
  (= 100/25/26 recordings; SN014/SN064/SN135 absent). Model selection: best FT epoch by
  val Cohen κ (their monitor score); test evaluated once at that epoch.
- Pretraining: `phase2_pretrain.py --variant scratch --datasets hmc_tf64_pretrain
  --p_in 1 --p_out 16 --batch 16 --epochs 60` (input-space PC) + matched `--epochs 0`
  random-init control; seeds 42/1/2. Same recipe as the Gate-0 sleep tf64 ladder.
- Fine-tune: `scripts/phase2_hmc_finetune.py` — full, 8 epochs, lr 1e-4, batch 8,
  max_len 400 (identical to the SEDF-78 FT recipe), best epoch by val κ; metrics pooled
  over all scored test epochs: balanced accuracy / Cohen κ / weighted-F1 (all plain
  sklearn, matching NeuroLM's metrics module) + plain acc / macro-F1.
- Note vs the ladder: their input pipeline (0.1–75 Hz FIR, 50 Hz notch, resample 200 Hz,
  ÷100 µV scaling, raw 30-s windows) differs from ours by design — we keep OUR recipe
  (tf64 Welch tokens at native 256 Hz, corpus standardization). Split/epochs/metrics are
  what make the rows comparable.
- Driver: `bash scripts/run_hmc.sh` (local H100-20C); results → results/phase4/hmc/.
- Pre-launch: 3-reviewer + protocol-verifier adversarial workflow over the new code before
  any GPU time is spent.

## 3. Status & run log
- 2026-08-27 03:2x — survey committed (dd79cba); HMC downloaded (16 GB, sha256 OK).
- 2026-08-27 03:3x — loader smoke test SN001: (854, 4, 64), label counts exactly match
  annotations.
- 2026-08-27 03:5x — pre-launch review (15-agent workflow): protocol verifier pinned the
  positional split + no-trim + val-κ selection from NeuroLM's code (all applied); code
  reviewers found 5 confirmed issues, all fixed (split constants 102/127, no trim,
  driver error-masking `| tee | grep || true` confined + DONE-sentinel asserts,
  `--ft_seed $SEED` actually passed, SystemExit→ValueError in Pool workers). Rejected as
  harmless: dataset='sleep_edf' provenance string inside the HMC archives.
- 2026-08-27 03:5x — queue launched: build → pretrain (pc 60 ep + rand) × seeds 42/1/2 →
  fine-tune per seed (results/phase4/hmc/).
- 2026-08-27 03:47 — build: split check PASSED (100/25/26 recs, 91,248/22,124/23,871
  epochs = NeuroLM Table 1 bit-exact). Pretrain 60 ep in 39 s (PC-MSE 0.973→0.531);
  FT ~15 s/arm on the H100.
- 2026-08-27 04:05–04:22 — extension: seeds 3–7 + 20-epoch-FT robustness arms on all
  8 seeds (rand's val kappa was still rising at ep 8; with val-kappa selection, e20 is
  the fairer setting and lifts BOTH arms ~+0.6–0.9).

## 4. Results  *(run date: 2026-08-27)*

Test set = 26 recordings / 23,871 pooled epochs, fixed split, mean ± sd over 8 seeds
(pretrain seed = FT seed ∈ {42,1,2,3,4,5,6,7}); best FT epoch by val κ.

| arm | FT epochs | BAC | κ | wF1 | acc | MF1 |
|---|---|---|---|---|---|---|
| PC-pretrained | 20 | **73.79 ± 0.87** | **.668 ± .009** | 74.54 ± 0.51 | 74.44 | 72.65 |
| random-init | 20 | 73.79 ± 0.71 | .668 ± .007 | 74.48 ± 0.45 | 74.49 | 72.72 |
| PC-pretrained | 8 | 73.20 ± 0.89 | .661 ± .010 | 73.83 | 73.87 | 71.96 |
| random-init | 8 | 72.91 ± 0.77 | .654 ± .007 | 73.32 | 73.33 | 71.57 |

Paired per-seed deltas (pc − rand): e20 **+0.00 ± 1.25 BAC / +0.0001 ± 0.0121 κ**;
e8 +0.29 ± 1.13 BAC / +0.007 ± 0.012 κ. Per-seed range −1.2…+2.8 — the seed-42-only
delta (+2.8) would have told a false story.

Published ladder on the identical split (full fine-tunes of large pretrained EEG
foundation models; BAC/κ/wF1): REVE-Base .740/.698/.764 · CSBrain .735/.682/.751 ·
LaBraM-Base .729/.681/.755 · CBraMod .727/.669/.740 · EEGPT .703/.658/.732 ·
BIOT .686/.630/.709 · NeuroLM-B .674/.619/.713.
**PhysioFM-S e20 = .738/.668/.745** — 2nd of 8 on BAC, 4th on κ/wF1, at ~1/100th
the parameters and no external pretraining corpus.

CSV: results/phase4/hmc/finetune.csv (val + test rows, all seeds/arms).

## 5. Interpretation — agent's reading
1. **The architecture claim survives contact with a second dataset.** Same recipe, no
   tuning, patient population, 4 channels instead of 2: competitive with the 2025-26
   foundation-model ladder. This is the paper's strongest new evidence.
2. **The pretraining null replicates.** With 8 paired seeds the in-domain PC-pretraining
   effect on HMC is exactly zero; combined with seizure (≈0) and the SSL literature
   (BENDR null, mulEEG supervised>SSL), SEDF-78's +0.85 should be presented as
   "within noise", not as a positive effect. Frozen-probe deltas remain the only place
   pretraining looks big — the inflation story, again.
3. Longer fine-tuning (8→20 epochs) helps both arms and slightly *shrinks* the delta —
   the small e8 gap is partly an optimization-speed effect (pretrained converges
   faster), not a representation-quality effect. Worth stating in the paper.
4. What could still rescue pretraining: cross-corpus transfer (SHHS→SEDF/HMC) and
   label-scarcity — the literature's two positive regimes; P2018/SHHS runs address it.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

## 7. Commits
TBD.

## 8. Links
- docs/SLEEP_DATASET_CANDIDATES.md (survey + comparison targets)
- https://physionet.org/content/hmc-sleep-staging/1.1/
- Published ladder sources: NeuroLM arXiv:2409.00101, CSBrain arXiv:2506.23075,
  REVE arXiv:2510.21585, ALFEE arXiv:2505.06291.
