---
id: EXP-0023
title: Gate 3 — Streaming/causal evaluation: causal vs bidirectional twin under a latency constraint
status: planned
created: 2026-08-18
run_date: 
agent: claude-code
phase: next-phase
verified: no
tags: gate3, streaming, causal, bidirectional, latency
commits: 
verdict: 
---

# EXP-0023 — Gate 3 — Streaming / causal evaluation

> **Status:** planned · **Created:** 2026-08-18 · **Agent:** claude-code · **Phase:** next-phase

## 1. Why
The plan's "defensible claim": a causal decoder can legitimately win where decisions must be
made at time *t* from data ≤ *t*. We build the bidirectional twin of the same stack
(`PhysioFMS(causal=False)`), fine-tune both on the same folds, and evaluate (a) offline
(whole window visible) and (b) online (only the past visible at each epoch, i.e. the
bidirectional model is re-run on the truncated prefix). Prediction: offline the bidirectional
twin ≥ causal; online the causal model ≥ bidirectional and is O(1) per decision vs O(L).

## 2. Setup
`scripts/gate3_streaming_eval.py` on sleep (DE, and raw if Gate 2 lands): same 5 subject-
disjoint folds, --mode full fine-tuning, per-epoch accuracy/κ offline vs online, wall-clock per
decision.

## 3. Status & run log
- 2026-08-18 — created (planned).

## 4. Results *(pending)*

## 5. Interpretation *(pending)*

## 6. ✅ Your verification — *(reserved for Mahdiar)*
- [ ] **Verified**

## 7. Commits

## 8. Links
- Plan: `docs/NEXT_PHASE_PLAN.md` §4 Gate 3
