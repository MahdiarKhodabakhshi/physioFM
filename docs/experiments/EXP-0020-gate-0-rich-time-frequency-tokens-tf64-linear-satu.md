---
id: EXP-0020
title: Gate 0 — Rich time–frequency tokens (tf64): linear-saturation pilot + PC ladder
status: done
created: 2026-08-18
run_date: 2026-08-18
agent: claude-code
phase: next-phase
verified: no
tags: gate0, tf64, linear-saturation, R1, sleep, seizure
commits: 951a916, d4e100a
verdict: PILOT FAILS ITS PRE-REGISTERED RULE, BUT LIFTS THE ARCHITECTURE. tf64 (64 log-spaced bins) vs DE: sleep linear 67.86 -> 72.83 (+5), best nonlinear (HGB) 69.60 -> 75.17, headroom +1.7 -> +2.3 (rule needed +2 over DE: fails); seizure identical linear (72.41 = 72.41), best nonlinear 74.02 -> 72.63 (no headroom). PC ladder on sleep tf64, fine-tuned, 4 pretraining seeds: input-PC 77.88+-0.37, latent 77.31+-0.36, random-init 77.03+-0.19 -> the whole model gains ~+2.3 over the DE pipeline (75.5) but the pretraining benefit shrinks +2.5 -> +0.85. Frozen: PC 76.75 vs random 65.51 vs dimension-matched projection 73.30 (+3.5). Seizure tf64 ladder skipped by the plan's stop rule.
---

# EXP-0020 — Gate 0 — Rich time–frequency tokens (tf64)

> **Status:** running · **Created:** 2026-08-18 · **Agent:** claude-code · **Phase:** next-phase (docs/NEXT_PHASE_PLAN.md)

## 1. Why — hypothesis & motivation
Requirement **R1** of the next-phase plan: the discriminative information must NOT be linearly
accessible from the input, or a linear probe already reads it optimally and no encoder has
headroom (F10, [[EXP-0010]]; the dimension-matched control, [[EXP-0017]] §3). Before building a
raw-EEG pipeline, the cheapest test is *less compression alone*: the same per-epoch pipeline
with 64 log-spaced log-power bins per channel (`physiofm/spectral.py`) instead of the 5 DE bands.

**Pre-registered predictions / decision rule (written before any result):**
- H0 (DE-like saturation): on tf64, best nonlinear head − linear head < 2 pts, or the headroom
  is not larger than DE's by ≥ 2 pts → tf64 is saturated like DE → **R1 fails for spectral
  features**; the plan says "stop, do not proceed to raw EEG *with this objective*" (Gate 2 then
  rests on Gate 1's objective fix, not on more spectral resolution).
- H1 (headroom): tf64 headroom ≥ 2 pts and ≥ DE headroom + 2 → run the PC ladder on tf64
  (input-PC vs latent-PC vs random-init, frozen + fine-tuned) and the dimension-matched control.
  Success there = pretraining beats a random 256-d projection of tf64 by more than the +3.3 seen
  on sleep DE.

## 2. Setup
- Archives: `scripts/build_tf_dataset.py --task sleep|chbmit` →
  `data/physiofm/tf_features/{sleep_edf,chbmit}_tf64.npz` (+ label companions, asserted
  identical to the DE ones). Sleep: 153 rec / 78 subj / 2 ch × 64 = 128-d per 30 s epoch;
  seizure: 682 rec / 24 patients / 18 ch × 64 = 1152-d per 2 s epoch. Welch, nperseg 4 s
  (sleep) / 2 s (seizure), bins geomspace(0.5, 49 Hz, 65).
- Saturation: `scripts/gate0_saturation.py --task sleep|seizure` — heads logreg (linear) vs
  mlp_bal (balanced 2-layer MLP) vs hgb (gradient-boosted trees), same folds as the model
  evaluators (sleep subject-disjoint 5-fold seed 42; seizure LOPO with a stratified train cap).
- Ladder (if H1): `scripts/run_gate1_pretrain.sh` with TASKS=sleep_edf_tf64 chbmit_tf64 →
  `scripts/run_gate1_eval.sh`; `scripts/diagnose_encoder.py` for the random-projection control.
- Interpreter `/home/mahdiar/.conda/envs/xcqa/bin/python`, local H100-20C (20 GB), SDPA attention.

## 3. Status & run log
- 2026-08-18 — created; archives built (sleep 12 min, CHB-MIT 2.5 min on 5 workers) and asserted label-aligned; saturation tests run locally (CPU); sleep tf64 ladder pretrained + evaluated locally (seed 42) and multi-seed (1,2,3) later the same night; seizure tf64 ladder started on the pod and stopped (stop rule).

## 4. Results *(2026-08-18; `results/phase4/gate0/`)*

Saturation (`saturation_sleep.csv`, `saturation_seizure.csv`):

| task | features | linear | MLP | HGB | headroom |
|---|---|---:|---:|---:|---:|
| sleep acc | DE | 67.86 | 67.20 | 69.60 | +1.7 |
| sleep acc | tf64 | 72.83 | 71.85 | 75.17 | +2.3 |
| seizure bal-acc / AUC | DE | 72.41 / .807 | 69.54 / .824 | 74.02 / .851 | +1.6 |
| seizure bal-acc / AUC | tf64 | 72.41 / .809 | 69.92 / .824 | 72.63 / .849 | +0.2 |

Sleep tf64 ladder (`sleep_edf_tf64/f13_sleep_frozen_seed42.csv`, `finetune.csv`, `diagnose_encoder_tf64.csv`):
frozen raw 72.83 / rand-proj 73.30 / PC 76.75 / latent 71.02 / rand 65.51 / concat 77.41;
fine-tuned (seeds 42,1,2,3): PC 77.43, 78.42, 77.72, 77.96 (77.88±0.37); latent 76.87, 77.24, 77.86, 77.28 (77.31±0.36); rand 77.11, 77.25, 77.02, 76.73 (77.03±0.19).

## 5. Interpretation — agent's reading

R1 is not restored by spectral resolution: nonlinear headroom stays small and seizure shows none. What tf64 does is raise the *linear* ceiling on sleep by five points (DE discarded linearly useful detail) and lift the fine-tuned architecture to 77.9 % / κ .71 — the best absolute number in the project — while the pretraining benefit shrinks to +0.85 (consistent sign, 4/4 seeds). Frozen probes again overstate pretraining by >10 points. Full narrative in `docs/NEXT_PHASE_RESULTS.md` §1.

## 6. ✅ Your verification — *(reserved for Mahdiar)*
- [ ] **Verified**
- **Notes / corrections:**

## 7. Commits

## 8. Links
- Plan: `docs/NEXT_PHASE_PLAN.md` §4 Gate 0 · Related: [[EXP-0010]], [[EXP-0017]], [[EXP-0021]]
