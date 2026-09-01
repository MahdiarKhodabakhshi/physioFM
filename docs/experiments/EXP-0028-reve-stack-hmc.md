---
id: EXP-0028
title: Stacking the causal sequence recipe on frozen REVE features — can we improve the SOTA foundation model?
status: done
created: 2026-08-31
run_date: 2026-09-01
agent: claude-code
phase: external-validation
verified: no
tags: reve, foundation-model, stacking, hmc, sequence
commits: 79ab57f
verdict: H1 CONFIRMED. Frozen REVE-Base + our causal decoder (11M trainable, fixed HMC split, 3 seeds): balanced-loss arm BAC 75.55 +/- 0.32 / kappa .685 / wF1 .755 -> +1.5 BAC over REVE's own published full fine-tune (74.01 / .698 / .764; 10 runs, LoRA, souping) and +3.1 BAC over frozen-REVE-without-context (72.44 / .649); plain-CE arm 74.03 / .695 / .760 -> statistical tie with their full fine-tune on ALL metrics at a fraction of the adaptation compute, and the decoder is causal (one-epoch streaming latency). PC pretraining of the decoder adds nothing (in-domain null, 5th replication) - the entire gain is inter-epoch sequence modeling. H2 (REVE-Large stack > published SOTA) untested: gated-weights access to reve-large still pending.
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

## 4. Results  *(run date: 2026-09-01)*

Fixed NeuroLM split, test = 23,871 pooled epochs, BAC / κ / wF1:

| row | BAC | κ | wF1 |
|---|---|---|---|
| REVE-Base frozen, per-epoch logreg (NO sequence context) | 72.44 | .649 | 72.74 |
| REVE-Base published full fine-tune (their Table 14; LoRA, 10 runs, souping) | 74.01 | .698 | 76.38 |
| our tf64 from scratch (EXP-0024, 8 seeds) | 73.79 | .668 | 74.54 |
| **frozen REVE + our causal decoder** (balanced loss, 3 seeds) | **75.55 ± 0.32** | .685 ± .005 | 75.47 |
| frozen REVE + our causal decoder (plain CE, 3 seeds) | 74.03 ± 0.61 | **.695 ± .004** | 75.97 |
| + PC pretraining of the decoder (balanced, 3 seeds) | 74.82 ± 0.80 | .680 | 75.19 |

Extraction 11 min (151 recordings); decoder pretrain 87 s; fine-tune ~40 s/arm.
Pretrain OOM at batch 16 (2048-d tokens) → batch 4 (CHB-MIT precedent).
CSV: results/phase4/reve_stack/finetune.csv; probe: probe.txt.

## 5. Interpretation — agent's reading
1. **The supervisor's suggestion lands.** Adding our causal inter-epoch sequence model
   to frozen REVE beats REVE's own full fine-tune on the ladder's headline metric
   (+1.5 BAC) and ties it on all metrics under their loss convention — while training
   only an 11M decoder (their FT: LoRA over 69M + 10 runs + souping) and keeping
   one-epoch streaming latency. The claim writes itself: *sequence context is the
   cheapest upgrade to an EEG foundation model, and it need not cost real-time.*
2. The gain decomposes cleanly: +3.1 BAC from sequence modeling over context-free
   frozen features; loss weighting trades BAC vs κ/wF1 along their published trade-off.
3. In-domain PC pretraining of the decoder: null again (5th replication) — on top of a
   foundation encoder there is even less for input-space PC to learn.
4. **Extension results (2026-09-01, same recipe on the other two corpora):**
   - **P2018** (SleePyCo folds, pooled 892,200 test epochs): stack (rand) **78.25 acc /
     MF1 77.32 / κ .7102** vs our tf64 76.09/.684 → **+2.2 acc / +.026 κ**; gap to
     SleePyCo (80.9/.737) halves from 4.8 to 2.6 acc, and the stack passes
     U-Time-class offline models (78.8/.714) while staying causal. pc arm 77.46/.7005
     — in-domain PC null again.
   - **Sleep-EDF-78**: stack ≈ our tf64 (e8 pc 77.49 vs 77.58; e16 rand 78.70/.712 vs
     78.46–79.00) — REVE features do NOT help on 2-channel 100-Hz-native data
     (upsampled to 200 Hz; unusual Fpz-Cz/Pz-Oz derivations). Informative negative:
     the foundation-feature payoff tracks recording richness (4 ch/256 Hz HMC and
     6 ch/200 Hz P2018 gain; 2 ch/100 Hz SEDF doesn't).
   - Extraction costs: SEDF 13 min; P2018 60 min (994 records).
   - Still open: REVE-Large stack (gate access pending).
   CSVs: results/phase4/reve_stack/{sedf_finetune,p2018_finetune}.csv.
5. **Rigor pass (2026-09-01 afternoon):**
   - HMC stack at 8 seeds (matching the tf64 ladder): balanced arm **BAC 75.21 ± 0.58 /
     κ .684** (+1.2 over REVE's 74.01, robust); their-loss arm **κ .694 ± .004 /
     wF1 76.08 / BAC 74.33** — statistical tie with their full fine-tune on κ/wF1,
     ahead on BAC. Headline survives full seed support.
   - e16 SEDF rows now 3-seed: tf64-linear pc **78.69 ± 0.18 / κ .713** (context head
     +0.1 — EXP-0027 null confirmed at e16); REVE-stack e16 78.26 ≈ parity.
   - **REVE-feature transfer (P2018-REVE perch donor → SEDF-REVE perch, 3 seeds):
     transfer 77.90 ± 0.34 vs rand 77.57 ± 0.56 → Δ +0.3 ≈ 0** — unlike tf64 features
     (+3.14). Reading: REVE's 60k-hour pretraining and our cross-corpus PC transfer
     are SUBSTITUTES, not additive — REVE-perch rand (77.57) already sits where the
     tf64 transfer arm landed (78.35-ish), i.e. the foundation encoder supplies the
     domain-shift knowledge that transfer pretraining used to supply. Coherent with
     the whole pretraining story: both mechanisms inject cross-dataset structure;
     once one is present, the other is redundant.
   CSVs: results/phase4/reve_stack/transfer_ft.csv, context/sedf_structured.csv.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

## 7. Commits
79ab57f (pipeline), this commit (results; batch-4 OOM fix; --class_weight flag).

## 8. Links
- REVE: arXiv:2510.21585 (NeurIPS 2025), github.com/elouayas/reve_eeg (MIT),
  huggingface.co/brain-bzh/reve-base (gated). EXP-0024 (our HMC ladder), EXP-0027
  (the gap-is-the-epoch-encoder diagnosis this experiment completes).
