---
id: EXP-0024
title: HMC external validation — tf64 + causal decoder + PC pretraining on a second sleep dataset
status: running
created: 2026-08-27
run_date: TBD
agent: claude-code
phase: external-validation
verified: no
tags: hmc, sleep, tf64, external-validation, fixed-split
commits: TBD
verdict: TBD
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

## 4. Results  *(run date: TBD)*
TBD — table: arm (pc/rand) × seed → test BAC / κ / wF1 / acc / MF1, vs published ladder.

## 5. Interpretation — agent's reading
TBD.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

## 7. Commits
TBD.

## 8. Links
- docs/SLEEP_DATASET_CANDIDATES.md (survey + comparison targets)
- https://physionet.org/content/hmc-sleep-staging/1.1/
- Published ladder sources: NeuroLM arXiv:2409.00101, CSBrain arXiv:2506.23075,
  REVE arXiv:2510.21585, ALFEE arXiv:2505.06291.
