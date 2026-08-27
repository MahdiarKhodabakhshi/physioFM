---
id: EXP-0026
title: Cross-corpus transfer — P2018-pretrained donors fine-tuned on Sleep-EDF-78 and HMC
status: running
created: 2026-08-27
run_date: 2026-08-27
agent: claude-code
phase: external-validation
verified: no
tags: transfer, pretraining, p2018, sleep_edf, hmc, perch
commits: TBD
verdict: TBD
---

# EXP-0026 — Cross-corpus transfer (the literature's positive regime)

> **Status:** running · **Created:** 2026-08-27 · **Agent:** claude-code · **Phase:** external-validation

## 1. Why — hypothesis & motivation
In-domain PC pretraining is now a triple-replicated null (SEDF ~+0.85 noise; HMC
+0.00 ± 1.25, 8 seeds; P2018 Δκ .0001 at 994 subjects — EXP-0024/0025). The published
sleep literature shows the same in-domain pattern but consistent POSITIVE transfer
across corpora (SleepTransformer SHHS→SEDF-78 +3.5 acc / +.046 κ; Phan et al.
MASS→SEDF +3.0; L-SeqSleepNet +2.3…+6.6). This is also the proposal's Phase-3 promise
(joint pretraining → transfer). Hypothesis: PC pretraining on the biggest corpus
(P2018, 994 subjects) transfers a benefit to smaller target corpora that same-corpus
pretraining cannot provide.

## 2. Setup — exactly what was run
Donors (seed 42, pc 60 ep, input-space PC): pretrained on ALL 994 P2018 records —
(a) structured 6×64; (b) per-electrode 1×64 (5,964 sequences). Donor corpora contain
no target-corpus subjects by construction.

Transfer mechanics (`scripts/make_transfer_ckpt.py`):
- **full** (per-electrode only — 1×64 tokens are channel-count-agnostic): donor
  weights verbatim; standardizer = target corpus stats.
- **trunk**: target random-init model with the donor's decoder transplanted
  (`layers.*` + `out_norm.*`); input/output blocks stay target-shaped random.

Evaluations (each = transfer arm vs matched random-init vs same-corpus-PC, identical
harness, FT full, best-epoch-by-val-κ on HMC / fixed 8 ep on SEDF):
1. SEDF-78 structured 2×64, 5-fold: rand / sedf-pc (gate-0 seed-42 models) /
   **trunk-transfer**.
2. SEDF-78 per-electrode 1×64 (`--merge_every 2`), 5-fold: rand / sedf-perch-pc /
   **full-transfer** (new perch archives sliced from tf64, `build_perch_tf64.py`).
3. HMC structured 4×64, fixed split, FT 20 ep: rand / hmc-pc (seed-42) /
   **trunk-transfer**.

## 2b. Comparability caveats (pre-registered, from the pre-launch review)
- **HMC is the exposure-clean comparison**: its pc baseline pretrained on SN001–125
  only, test = SN128–154 — fully symmetric with the donor. On SEDF the same-corpus pc
  arms follow the locked gate-0 convention (corpus-wide unlabeled pretraining incl.
  test subjects), which biases AGAINST the transfer arm; state this when interpreting
  transfer ≤ same-corpus-pc there.
- **Not compute-matched**: at matched 60 epochs the donors take ~6–19× more optimizer
  steps than target pc baselines (63 vs 10 steps/ep structured; 373 vs 20 perch).
  A transfer win is "more data + more compute under the same recipe".
- The SEDF pc/rand arms are RE-RUN under this harness (a seed-derivation fix changed
  batch-shuffle order vs gate-0), so rows are internally comparable; gate-0 numbers
  are not byte-identical references.
- Standardizers: always the target corpus's, all arms alike.

## 3. Status & run log
- 2026-08-27 05:0x — pre-launch adversarial review (13 agents): 5 confirmed findings
  fixed (dead `--ft_seed` in the SEDF harness now threaded; skip-guards moved to
  last-written artifacts; unused HMC perch build dropped as a leakage footgun;
  documented the two caveats above). The review also caught that the CONCURRENT
  P2018 c3 queue had crashed on an ARCH KeyError (`p2018_tf64_c3` never registered,
  lookup before the fallback) — fixed, relaunched.
- 2026-08-27 05:1x — queue launched (`scripts/run_transfer.sh`).

## 4. Results  *(run date: TBD)*
TBD.

## 5. Interpretation — agent's reading
TBD.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

## 7. Commits
TBD.

## 8. Links
- EXP-0024 (HMC), EXP-0025 (P2018), docs/SLEEP_DATASET_CANDIDATES.md §02.
