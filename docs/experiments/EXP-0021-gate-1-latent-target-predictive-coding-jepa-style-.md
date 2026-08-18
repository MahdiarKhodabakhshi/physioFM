---
id: EXP-0021
title: Gate 1 — Latent-target predictive coding (JEPA-style) vs input-space PC vs random-init
status: running
created: 2026-08-18
run_date: 2026-08-18
agent: claude-code
phase: next-phase
verified: no
tags: gate1, latent-targets, jepa, objective, R2, sleep, seizure, fine-tuned
commits: 
verdict: 
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
- 2026-08-18 — created; pretraining running locally (sleep pc reproduces the pod's best PC-MSE
  0.22886 exactly under SDPA + seed 42).

## 4. Results *(pending)*

## 5. Interpretation — agent's reading *(pending)*

## 6. ✅ Your verification — *(reserved for Mahdiar)*
- [ ] **Verified**
- **Notes / corrections:**

## 7. Commits

## 8. Links
- Plan: `docs/NEXT_PHASE_PLAN.md` §4 Gate 1 · Related: [[EXP-0017]], [[EXP-0020]], [[EXP-0022]]
