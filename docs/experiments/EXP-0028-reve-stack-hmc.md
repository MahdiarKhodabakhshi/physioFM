---
id: EXP-0028
title: Stacking the causal sequence recipe on frozen REVE features — can we improve the SOTA foundation model?
status: running
created: 2026-08-31
run_date: TBD
agent: claude-code
phase: external-validation
verified: no
tags: reve, foundation-model, stacking, hmc, sequence
commits: 79ab57f
verdict: TBD (blocked on HF gated-weights access)
---

# EXP-0028 — Our sequence model on REVE's shoulders

> **Status:** running · **Created:** 2026-08-31 · **Agent:** claude-code · **Phase:** external-validation

## 1. Why — hypothesis & motivation
Supervisor's suggestion (meeting of Aug 27–31): REVE released weights — stack our method
on top and see if we improve their results. This is precisely the completion of
EXP-0027's diagnosis: our gap is the epoch-level feature extractor; REVE is a 69M-param
learned epoch encoder — but it classifies each HMC 30-s epoch **with zero inter-epoch
context**, and no sequence model over epoch embeddings exists anywhere in their paper or
repo. Their own numbers expose the headroom: frozen-Base probe 0.647 BA vs full
fine-tune 0.7401 ± 0.0075 (10 runs) on the identical split; frozen-Large probe 0.710.
Hypotheses: (H1) frozen REVE-Base features + our causal decoder (fine-tuned) closes
most of the probe→FT gap via sequence context; (H2) the REVE-Large variant exceeds
their published 0.7401 — "our recipe improves the SOTA foundation model, and makes it
real-time capable" (the decoder stays causal; REVE features are per-epoch, so streaming
latency = one epoch).

## 2. Setup
- Leakage: clean — REVE's §3.1.1 removed downstream-task recordings (incl. HMC) from
  its 61k-h pretraining corpus. Sleep-EDF/P2018/CHB-MIT/SEED appear nowhere in their
  paper (future extension corpora are fully held out).
- Extraction (`scripts/build_hmc_reve.py`): REVE's own HMC preprocessing verbatim
  (bandpass 0.1–75 Hz, notch 50 Hz, resample 200 Hz, µV/10 clip ±100 per their
  hmc.yaml, monopolar names F4/C4/O2/C3 → 3D positions via brain-bzh/reve-positions);
  frozen REVE forward → per-channel mean over the 33 one-second patches → token
  (4 × 512) [Base] / (4 × 1216) [Large]; standard DETrial container; NeuroLM
  epoch-count assert (91,248/22,124/23,871).
- Ladder (`scripts/run_reve_stack.sh`): identical to the tf64 ladder — PC pretrain
  (60 ep) + rand control on SN001–127 only, seeds 42/1/2; fixed-split FT e20 with
  val-κ selection. Control (`scripts/reve_probe_control.py`): frozen REVE per-epoch
  logreg on the same split = REVE without sequence context.
- Comparison rows (all same split, BAC/κ/wF1): REVE-Base FT 0.7401/.6982/.7638 (their
  Table 14, 10 runs, LoRA + early-stop + souping); our tf64 0.738/.668/.745 (8 seeds);
  probe control; stacked Base; stacked Large.
- Notes: their FT protocol is heavier (LoRA, 10 runs, souping +1.5 %); ours trains a
  ~3.5 M decoder on frozen features — if we match/beat them it is with far less
  adaptation compute. Weights license: gated (research OK, no re-hosting).

## 3. Status & run log
- 2026-08-31 — recon + deep-read done (two agents; scale_factor 10 caught from their
  configs, correcting an initial /100 assumption). Pipeline + driver + control written,
  committed (79ab57f). **Blocked on**: HF Responsible-Use gate acceptance + login
  (user), then transformers-5.12 load smoke test (their code pins 4.56; braindecode
  ≥1.3 wrapper is the fallback loader).

## 4. Results  *(run date: TBD)*
TBD.

## 5. Interpretation — agent's reading
TBD.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

## 7. Commits
79ab57f (pipeline).

## 8. Links
- REVE: arXiv:2510.21585 (NeurIPS 2025), github.com/elouayas/reve_eeg (MIT),
  huggingface.co/brain-bzh/reve-base (gated). EXP-0024 (our HMC ladder), EXP-0027
  (the gap-is-the-epoch-encoder diagnosis this experiment completes).
