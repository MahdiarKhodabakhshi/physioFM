---
id: EXP-0022
title: Gate 2 — Raw-EEG structured tokens + per-electrode ablation
status: done
created: 2026-08-18
run_date: 2026-08-19
agent: claude-code
phase: next-phase
verified: no
tags: gate2, raw-eeg, structured-patch, per-electrode, braingpt-ablation, sleep, seizure
commits: 951a916, d4e100a
verdict: PARITY, NOT ESCAPE; ABLATION GOES THE OTHER WAY ON 2-CH SLEEP. Raw 200 ms structured tokens (2 ch x 20 samples, 29.3M tokens, 10 pretraining epochs on an H100 80 GB, 3 fine-tuning epochs, seed 42): fine-tuned input-PC 75.46 (k .676), latent 74.65, random-init 74.23 -> equal to the DE pipeline (75.4), 2.4 below tf64 (77.9). Frozen: PC 70.84 vs random 56.15 (+14.7, the largest frozen inflation in the project; collapses to +1.2 fine-tuned). Per-electrode (BrainGPT-style, 1 x 20 tokens, channels averaged per epoch): PC 76.36 / latent 75.45 / rand 74.56 fine-tuned, 72.68 / 62.81 / 60.65 frozen -> per-electrode >= structured on every arm (+0.9 fine-tuned, +1.8 frozen for input-PC); with two channels there is little spatial structure to exploit, so the plan's structured-vs-per-electrode ablation cannot be decided on Sleep-EDF (CHB-MIT raw not run). Raw pretext: input MSE 0.80 (structured) / 0.79 (per-electrode); latent skill +0.36 / +0.37 (real, unlike DE).
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
- 2026-08-18 — created; raw archives built locally (structured 985 s, per-channel 618 s; label-aligned); pretraining + evals on the RunPod H100 (`scripts/run_pod_queue.sh` P1/P2); per-electrode input-PC arm run locally.
- 2026-08-19 — done.

## 4. Results *(2026-08-19; `results/phase4/gate2/`)*

| tokens | arm | frozen | fine-tuned (3 ep) |
|---|---|---:|---:|
| structured 2ch×200 ms | PC / latent / rand | 70.84 / 56.90 / 56.15 | 75.46 / 74.65 / 74.23 |
| per-electrode 1×200 ms | PC / latent / rand | 72.68 / 62.81 / 60.65 | 76.36 / 75.45 / 74.56 |
| ref DE pipeline | PC / rand | 72.62 / 62.86 | 75.37 / 73.18 |
| ref tf64 pipeline | PC / rand (4-seed) | 76.75 / 65.51 | 77.88 / 77.03 |

Files: `sleep_edf_raw/{f13_sleep_frozen_seed42.csv,finetune.csv}`, `sleep_edf_raw_perch/{f13_sleep_frozen_seed42.csv,f13_sleep_pc_frozen_seed42.csv,finetune.csv,finetune_pc_local.csv}`.

## 5. Interpretation — agent's reading

At this budget raw EEG does not move the ceiling (parity with DE, below tf64) and does not make pretraining matter (+1.2 fine-tuned; the +14.7 frozen gap is the strongest demonstration yet that frozen probes measure the poverty of a random-init feature extractor, not the value of pretraining). The per-electrode result is a caution for the "structured spatial patching is our contribution" framing: on a 2-channel montage the BrainGPT-style decomposition is at least as good; the claim needs a many-channel corpus. Caveats: single pretraining seed, short recipe (10 + 3 epochs), fp32; seizure raw not run. See `docs/NEXT_PHASE_RESULTS.md` §3.

## 6. ✅ Your verification — *(reserved for Mahdiar)*
- [ ] **Verified**

## 7. Commits

## 8. Links
- Plan: `docs/NEXT_PHASE_PLAN.md` §4 Gate 2 · Related: [[EXP-0013]] (F15, the blocked raw-EEG leg), [[EXP-0021]]
