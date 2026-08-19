---
id: EXP-0021
title: Gate 1 — Latent-target predictive coding (JEPA-style) vs input-space PC vs random-init
status: done
created: 2026-08-18
run_date: 2026-08-18
agent: claude-code
phase: next-phase
verified: no
tags: gate1, latent-targets, jepa, objective, R2, sleep, seizure, fine-tuned
commits: 951a916, d4e100a
verdict: P1 FAILS, P2 FAILS. Latent-target PC (EMA target, stop-grad, time-normalised targets) does not beat input-space PC under fine-tuning: sleep (4 pretraining seeds) input-PC 75.54+-0.32, latent 73.69+-0.45, random-init 73.08+-0.49; seizure LOPO (seed 42) 78.36 / 79.74 / 80.21 bal-acc (AUC .863 / .879 / .874), indistinguishable at +-12 per-patient sd. Five objective variants on sleep (delta targets, cosine/no-norm, variance term, EMA .99, p_out 4): 72.7-74.0, all at the random-init level. Frozen, latent features are below the raw-feature probe (sleep 66.1 vs 67.9; seizure 71.4 vs 72.4; emotion 36.7-45.7 vs 55-63). Mechanism: on the long smooth DE corpora the latent predictor never beats trivial baselines (skill -0.2 sleep, -0.6 seizure, ~0 tf64; the target drifts to a smooth trajectory; without time-normalisation it collapses outright), and where the pretext IS learned (emotion/MI skill .4-.6, raw tokens .36) the frozen gain is negative or ~0 -> pretext skill still anti-correlates with transfer. Latent targets do not realign the objective.
---

# EXP-0021 — Gate 1 — Latent-target predictive coding

> **Status:** running · **Created:** 2026-08-18 · **Agent:** claude-code · **Phase:** next-phase

## 1. Why — hypothesis & motivation
Requirement **R2** of the next-phase plan: the prediction target must not be the raw input.
[[EXP-0017]] §4e showed the pretext IS learned (40–65 % below persistence on 4/5 datasets) but
pretext skill *anti-correlates* with downstream gain — input-space MSE rewards modelling the
smooth, autocorrelated component of the features, which is not the discriminative one. The fix
that keeps the contribution intact (decoder-only causal transformer + predictive coding) is to
predict the *embedding* of the future (JEPA / BYOL / data2vec style): EMA target encoder,
stop-gradient, MLP predictor from h_j to the next p_out target embeddings, targets
instance-normalised over time within each sequence (data2vec) so a time-constant embedding
cannot fit them.

**Pre-registered predictions / decision rule:**
- P1 (transfer, the one that matters): under end-to-end fine-tuning on sleep (subject-disjoint
  5-fold) and seizure (LOPO), latent-PC > input-PC. Success = latent-PC − random-init > +2.2
  (sleep) and > 0 (seizure) with paired per-fold/per-patient support; the honest full-protocol
  reference is [[EXP-0017]] §4b–c (input-PC +2.19 sleep, ≈0 seizure).
- P2 (mechanism): the pretext-vs-transfer correlation flips sign — latent forecasting skill
  (predictor error relative to persistence-in-latent-space) *correlates* with downstream gain
  across the 5 datasets, instead of anti-correlating.
- Null: latent-PC ≈ input-PC ≈ random-init after fine-tuning → the objective is not the
  limiter; the DE substrate is (consistent with R1), and Gate 2 must carry the plan.

## 2. Setup
`scripts/phase2_pretrain.py --objective latent` (new): predictor 2-layer MLP (d→2d→p_out·d),
EMA momentum 0.996 → 1.0 (cosine), targets = EMA encoder `encode()` states, instance-normalised
over valid time steps per sequence, MSE loss; monitors: within-sequence embedding std,
effective rank of token embeddings, persistence-in-latent error. Matched arms per task, same
seed 42, p_in 1, p_out 16 (8 for MI), 60 epochs, SDPA attention:
`scripts/run_gate1_pretrain.sh` → `results/phase4/gate1/<task>/seed42/{pc,latent,rand}/`.
Eval: `scripts/run_gate1_eval.sh` — frozen probe (`phase2_f13_sleep.py`,
`phase2_chbmit_eval.py`) and fine-tuned (`phase2_sleep_finetune.py --mode full --epochs 8`,
`phase2_chbmit_finetune.py --epochs 4`), all arms including the new `physiofm_latent`.
Pretext diagnostic in latent space: `scripts/diagnose_pretext_latent.py`.

## 3. Status & run log
- 2026-08-18 — created; pretraining locally (sleep pc reproduces the pod's best PC-MSE 0.22886 exactly under SDPA + seed 42); code review caught (a) an SDPA fallback bug in the bidirectional twin, (b) missing trivial baselines in the latent diagnostic, (c) RAM blow-up in raw-token extraction — all fixed before the affected runs. CHB-MIT frozen evals OOM-killed twice on the 62 GB local box → run on the RunPod H100 (2 TB RAM). Multi-seed + variant sweep local; diagnostics local + pod.

## 4. Results *(2026-08-18/19; `results/phase4/gate1/`)*

Sleep fine-tuned (`sleep_edf/finetune.csv`), 4 pretraining seeds: PC 75.37/75.09/75.92/75.76; latent 74.12/73.89/72.93/73.82; rand 73.18/72.55/73.83/72.76. Variants (seed 42): latent_delta 73.87, latent_cos_nonorm 72.74, latent_varreg 73.20, latent_ema099 74.00, latent_pout4 74.02. Frozen (seed 42): raw 67.86, PC 72.62, latent 66.08, rand 62.86.
Seizure (`chbmit/f17_chbmit_frozen_seed42.csv`, `chbmit/finetune.csv`): frozen raw 72.37/.806, PC 77.42/.852, latent 71.36/.777, rand 67.47/.741; fine-tuned PC 78.36/.863, latent 79.74/.879, rand 80.21/.874.
Small datasets, frozen (`seed_iv_raw/`, `seed_iv/`, `bci_iv_2a/`): un-smoothed emotion raw 55.35 / PC 54.75 / latent 36.68 / rand 41.97; smoothed emotion 62.75 / 61.52 / 45.74 / 55.98; MI 51.00 / 42.32 / 39.39 / 43.63.
Latent pretext diagnostic (`diagnose_pretext_latent.csv` + `logs/pretext_latent_local.log`; skill = 1 − model/best trivial): sleep DE −0.20/−0.21/−0.16 (seeds 1-3), seizure −0.59, sleep tf64 −0.01, sleep raw +0.36, per-electrode raw +0.37, emotion un-smoothed +0.62, smoothed +0.39, MI +0.59; variants on sleep: delta +0.05, cos/no-norm −1.02 (collapse: persistence distance 0.002), varreg +0.19, ema099 −0.25, pout4 −0.46.

## 5. Interpretation — agent's reading

The R2 hypothesis — that input-space MSE was the reason pretraining fails — is falsified: moving the target into latent space (in five variants) never beats input-space PC and lands at random-init under fine-tuning. The diagnostic explains why on the per-epoch corpora: the EMA target becomes temporally smooth and the objective is solved by smoothness (skill ≤ 0), i.e. the latent analogue of the input-space degeneracy. Where the latent pretext is genuinely learned (short trials, raw tokens) it still does not transfer. Combined with EXP-0017, forecasting-style SSL on these EEG features is worth ≈ +2.5 (DE) or less on sleep and nothing on seizure regardless of target space. See `docs/NEXT_PHASE_RESULTS.md` §2.

## 6. ✅ Your verification — *(reserved for Mahdiar)*
- [ ] **Verified**
- **Notes / corrections:**

## 7. Commits

## 8. Links
- Plan: `docs/NEXT_PHASE_PLAN.md` §4 Gate 1 · Related: [[EXP-0017]], [[EXP-0020]], [[EXP-0022]]
