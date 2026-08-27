---
id: EXP-0025
title: Physio2018 (994 subjects) — SleePyCo-split validation + the scale test of PC pretraining
status: running
created: 2026-08-27
run_date: 2026-08-27
agent: claude-code
phase: external-validation
verified: no
tags: physio2018, sleep, tf64, external-validation, scale
commits: bb8bf5d
verdict: TBD (6ch done; c3 arm running)
---

# EXP-0025 — Physio2018: third sleep corpus, 994 subjects

> **Status:** running · **Created:** 2026-08-27 · **Agent:** claude-code · **Phase:** external-validation

## 1. Why — hypothesis & motivation
Two questions the smaller corpora can't answer: (H1) where does the causal
architecture land against the *supervised bidirectional* SOTA ladder when the
benchmark is big and the protocol is split-identical to a published row; (H2) does
in-domain PC pretraining start paying when the pretraining corpus is 6.5× larger
(994 subjects vs 153 recordings) — the "not enough data" explanation for the null.

## 2. Setup — exactly what was run
- Data: CinC-2018 training half (994 labeled records, MGH; 6 EEG @ 200 Hz, WFDB).
  S3 mirror, all files fetched, zero failures. Loader `physiofm/physio2018.py`:
  stages from WFDB `.arousal` aux_notes (30-s-aligned, persist-until-next; unscored
  head + 'undefined' dropped; no wake trim). tf64 tokens at native 200 Hz.
- Label fidelity audit (all 994 records): our extraction is bit-exact vs the WFDB
  annotations everywhere, and vs the sample-wise `-arousal.mat` reference vectors in
  991/994. The 3 exceptions (tr03-0314 +12 REM, tr05-0326 +46 N2, tr07-0602 +1 N2)
  are records whose distributed WFDB annotations lack the initial stage marks that
  the sample-wise vectors contain → we have 892,200 scored epochs vs the vectors'
  892,259 vs SleePyCo Table 2's 892,262 (their +3 unexplained, their pipeline).
  0.007 % — immaterial; kept WFDB-only loading for reproducibility.
- Split: SleePyCo's published fold file, vendored (MIT):
  `data/physiofm/splits/p2018_sleepyco_folds.npy` — 5 folds, test = contiguous
  199/198-record blocks, val = 50; **split-identical** to SleePyCo's row,
  protocol-matched to XSleepNet2 (random folds).
- Per-fold pretraining (pc 60 ep / rand) on the fold's non-test records only;
  standardizer likewise. FT full, 8 ep, lr 1e-4, max_len 400, best epoch by val κ;
  metrics pooled over all test epochs across folds (their convention):
  acc / MF1 / unweighted κ (+ BAC/wF1 for our cross-dataset table).
- Variants: 6×64 (our full recipe) and C3-M2-only 1×64 (input-identical to the
  published single-channel rows). Driver `scripts/run_p2018.sh`, seed 42.

## 3. Status & run log
- 2026-08-27 04:05 — build (994 recs, ~6 min, 7 workers); count audit vs SleePyCo
  Table 2 → −62 flagged → full-corpus audit traced it to the 3 records above.
- 2026-08-27 04:12–04:39 — fold corpora + 10 pretrains (6ch).
- 2026-08-27 04:39–04:50 — 6ch fine-tune sweep done. c3 arm running.

## 4. Results  *(run date: 2026-08-27)*
6×64, pooled over 892,200 test epochs (5 folds), seed 42:

| arm | acc | MF1 | κ | BAC | wF1 |
|---|---|---|---|---|---|
| PC-pretrained | 76.14 | 75.14 | .6840 | 77.85 | 76.27 |
| random-init | 76.09 | 75.23 | .6839 | 78.08 | 76.28 |

Published (C3-A2 single-channel, bidirectional, multi-epoch context): SleePyCo
80.9 / 78.9 / .737 · XSleepNet2 80.3 / 78.6 / .732 · SeqSleepNet 79.4 / 77.6 / .719 ·
U-Time 78.8 / 77.4 / .714. Per-fold κ range .654–.710.

C3-only arm: TBD.

## 5. Interpretation — agent's reading
1. **H2 refuted at scale: Δκ = 0.0001 with 994 subjects.** The in-domain pretraining
   null is not a small-data artifact. Third corpus, same zero (SEDF +0.85 few-seed
   noise; HMC +0.00 ± 1.25 over 8 seeds; here 0.0001 at 6.5× data).
2. H1: the causal single-epoch-token model sits 2.7–4.8 acc below the bidirectional
   supervised ladder — the same relative position as on SEDF-78. Combined with HMC
   (competitive with the FM ladder) and Gate 3 (causal wins streaming +2.8/+5.0),
   the architecture story is consistent: a different operating point, ~3–5 acc below
   offline bidirectional SOTA, ahead online.
3. TBD after c3: how much of the gap is the channel count vs the context/model class.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

## 7. Commits
bb8bf5d (pipeline), TBD (results).

## 8. Links
- docs/SLEEP_DATASET_CANDIDATES.md; EXP-0024 (HMC).
- SleePyCo arXiv:2209.09452 (+ split file, MIT); XSleepNet arXiv:2007.05492.
