---
id: EXP-0027
title: Sequence-context head — the offline operating point (closing the gap to bidirectional SOTA)
status: done
created: 2026-08-27
run_date: 2026-08-27
agent: claude-code
phase: external-validation
verified: no
tags: context, head, offline, sleep_edf, hmc, p2018
commits: TBD
verdict: HYPOTHESIS REFUTED - and the refutation is good news for the paper's framing. Bidirectional sequence context on top of the causal encoder buys ~+0.4-1.0 acc at the 8-epoch budget, ~0 at matched 16-epoch budget (SEDF: ctx-e16 pc 79.00 vs lin-e16 pc 78.46, but lin-e16 rand 78.91 > ctx-e16 rand 78.38 - single seeds, pure noise); ladder-matched window (+-10 effective) == unrestricted whole-chunk; zero gain on HMC (BAC ~73.7 both heads), +0.1-0.5 pooled on P2018; nothing stacked on the P2018-transfer arm (78.54 vs 78.60). Conclusion: the causal encoder already carries the sequential information (consistent with EXP-0023's bidirectional twin being only +0.9 offline) - the residual 2-5 acc gap to SleePyCo/XSleepNet lives in the epoch-level feature extractor (fixed 64-bin Welch tokens vs their learned intra-epoch encoders), not in missing future context. Side finding: 8 FT epochs under-trains SEDF - e16 lifts every arm to 78.4-79.0 (new best structured SEDF), head irrelevant.
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

## 4. Results  *(run date: 2026-08-27)*

**SEDF-78 structured, paired on the seed-42 encoders, ft seeds 42/1/2 (acc mean ± seed-sd):**

| head | budget | pc | rand |
|---|---|---|---|
| linear | e8 | 77.58 ± 0.15 | 76.08 ± 0.51 |
| context (whole chunk) | e8 | 77.95 ± 0.34 | 77.05 ± 0.61 |
| context (window ±10 eff.) | e8 | 77.88 ± 0.23 | 76.59 ± 0.48 |
| linear | e16 (1 seed) | 78.46 / κ.709 | **78.91 / κ.715** |
| context | e16 (1 seed) | **79.00 / κ.717** | 78.38 / κ.709 |

**SEDF perch (ft seeds 42/1/2):** transfer+context 78.54 mean ≈ transfer+linear 78.60;
pc 77.81 ≈ 77.76; rand 75.72 ≈ 75.36. **HMC (e20, 3 seeds):** context BAC 73.56 (pc) /
73.86 (rand) ≈ linear 73.79/73.79. **P2018 (pooled):** context 76.27/κ.686 (pc),
76.60/.690 (rand) vs linear 76.14/.684, 76.09/.684.

CSVs: results/phase4/context/{sedf_structured,sedf_perch,hmc,p2018}.csv.

## 5. Interpretation — agent's reading
1. **The gap to bidirectional SOTA is not about context.** Adding the exact machinery
   the SOTA models have (bidirectional sequence stage, matched effective window) moves
   ≤1 acc and ~0 at matched budget, on three datasets. Combined with EXP-0023 (the
   fully bidirectional twin gains only +0.9 offline), the causal encoder demonstrably
   already extracts the sequential structure from the past alone.
2. **This strengthens the streaming claim**: the price of causality is ≈1 point, not
   the 3-5 the leaderboard gap suggests. The residual gap is the epoch-level feature
   extractor — fixed 64-bin Welch log-power + linear embedding vs learned CNN/attention
   intra-epoch encoders on raw waveforms. That (learned epoch encoders under the same
   causal recipe) is the honest SOTA path, a real architecture project, not a head.
3. Budget note: SEDF's fixed 8-epoch FT under-trains all arms; e16 gives ~+0.8-2.8 and
   the project's best structured SEDF numbers (78.4-79.0) regardless of head. If SEDF
   numbers go in the paper, rerun the headline arms at e16 with val-based selection
   (multi-seed) — currently e16 rows are single-seed.
4. Params caveat pre-registered: +1.06 M head (30.6 % of gradient-receiving params) —
   the null makes the capacity confound moot.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

## 7. Commits
(pipeline commit from launch), this commit (results).

## 8. Links
- EXP-0024/0025/0026; docs/EXTERNAL_VALIDATION_RESULTS.md; Gate 3 (EXP-0023) for the
  streaming side of the operating-point story.
