---
id: EXP-0027
title: Sequence-context head — the offline operating point (closing the gap to bidirectional SOTA)
status: running
created: 2026-08-27
run_date: 2026-08-27
agent: claude-code
phase: external-validation
verified: no
tags: context, head, offline, sleep_edf, hmc, p2018
commits: TBD
verdict: TBD
---

# EXP-0027 — Sequence-context head

> **Status:** running · **Created:** 2026-08-27 · **Agent:** claude-code · **Phase:** external-validation

## 1. Why — hypothesis & motivation
Every bidirectional supervised SOTA model classifies an epoch from a window of
neighbouring epoch features (SeqSleepNet/SleepTransformer/SleePyCo: 10–35 epochs);
our linear head concedes exactly that, and the measured gap is ~3–5 acc (EXP-0025:
−4.8 to SleePyCo at 6 ch; SEDF −6 to Convention-A SOTA). Hypothesis: adding the
missing sequence stage — a small bidirectional transformer head (~1.06 M params,
sinusoidal PE, 2 layers) over the causal encoder's per-epoch features — recovers
most of that gap, giving the paper an *offline* operating point while the linear
head keeps the *streaming* claim (a bounded-lookahead mask supports a future
latency-accuracy curve; lookahead=0 is provably causal, verified by test).

## 2. Setup
`physiofm/context_head.py` (+ `--head context` in all three fine-tune harnesses;
`apply_head` passes valid lengths so padding never pollutes attention). Everything
else identical to today's linear-head rows — same encoders, folds, seeds, class
weights, chunking (max_len 400), val-κ selection where the protocol has it. Runs
(`scripts/run_context.sh`): SEDF structured (pc/rand, ft seeds 42/1/2), SEDF perch
incl. the P2018-transfer hybrid (the best-combo hunt), HMC e20 (seeds 42/1/2, its
own pretrain seeds), P2018 6 ch (SleePyCo split, per-fold pretrains, seed 42).
Pre-registered caveats: +1.06 M params over the linear-head rows; whole-chunk
(≤400-epoch) context vs the ladder's 10–35; HMC's FM ladder classifies from short
windows, so the context row there is a separate operating point, not a ladder entry.

## 3. Status & run log
- 2026-08-27 ~15:00 — module written; lookahead=0 causality unit-verified; end-to-end
  smoke test passed (1 ep, 2 folds). Adversarial review workflow running; queue next.

## 4. Results  *(run date: TBD)*
TBD.

## 5. Interpretation — agent's reading
TBD.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

## 7. Commits
TBD.

## 8. Links
- EXP-0024/0025/0026; docs/EXTERNAL_VALIDATION_RESULTS.md; Gate 3 (EXP-0023) for the
  streaming side of the operating-point story.
