---
id: EXP-0022
title: Gate 2 — Raw-EEG structured tokens + per-electrode ablation
status: planned
created: 2026-08-18
run_date: 
agent: claude-code
phase: next-phase
verified: no
tags: gate2, raw-eeg, structured-patch, per-electrode, braingpt-ablation, sleep, seizure
commits: 
verdict: 
---

# EXP-0022 — Gate 2 — Raw-EEG structured tokens

> **Status:** planned · **Created:** 2026-08-18 · **Agent:** claude-code · **Phase:** next-phase

## 1. Why — hypothesis & motivation
DE (and, if Gate 0 fails, any spectral summary) deletes morphology (spindles, K-complexes,
spike-waves) and pre-solves the task linearly (R1). The plan's Gate 2 replaces the token with
**raw EEG: all channels × 200 ms** straight into the existing decoder, keeps the objective from
Gate 1, and runs the ablation that isolates our actual contribution: **structured multi-channel
patches vs BrainGPT-style per-electrode decomposition**, everything else identical.

**Pre-registered predictions / decision rule:**
- P1: on raw tokens the linear ceiling of the *input* is near chance, so the encoder must do
  the work; fine-tuned raw-token PhysioFM ≥ fine-tuned DE PhysioFM (75.4 % sleep) is the bar
  for "escaping the DE bottleneck".
- P2 (pretraining finally matters): latent-PC − random-init on raw tokens (fine-tuned) is
  larger than on DE (+2.2). If it is still ≈ 0, pretraining is not rescued by the substrate.
- P3 (the ablation that carries the paper): structured (C × 200 ms) tokens > per-electrode
  (1 × 200 ms) tokens under the identical objective/protocol.

## 2. Setup
`physiofm/raw_eeg.py` + `scripts/build_raw_dataset.py` → `data/physiofm/raw_tokens/
sleep_edf_raw200ms.npz` (tokens × 2 ch × 20 samples, 150 tokens/epoch, same recording order,
label companion asserted identical) and `_perch` (one sequence per channel, 1 × 20 tokens).
Model = PhysioFMS with n_cb=40 (or 20), sequences chunked to K=20 epochs (3000 tokens);
per-epoch readout = mean of the epoch's token states (`--tokens_per_epoch 150`). Same
pretraining recipe (60 epochs) with `--max_len 3000`; frozen + fine-tuned evals as Gate 1.
Seizure raw (18 ch × 51 samples @256 Hz, 10 tokens/epoch) follows if compute allows.

## 3. Status & run log
- 2026-08-18 — created (planned).

## 4. Results *(pending)*

## 5. Interpretation *(pending)*

## 6. ✅ Your verification — *(reserved for Mahdiar)*
- [ ] **Verified**

## 7. Commits

## 8. Links
- Plan: `docs/NEXT_PHASE_PLAN.md` §4 Gate 2 · Related: [[EXP-0013]] (F15, the blocked raw-EEG leg), [[EXP-0021]]
