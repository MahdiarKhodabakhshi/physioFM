---
id: EXP-0014
title: F16 — Motor imagery (BCI-IV-2a), the 2nd dynamic task
status: done
created: 2026-07-12
run_date: 2026-07-12
agent: claude-code
phase: phase3
verified: no
tags: bci-iv-2a, motor-imagery, phase3, temporal-pc, second-dynamic-task, null-result
commits:
verdict: NULL for the FM on motor imagery. PC ≈ random-init (42.4 vs 43.6%, pc even lower) and BOTH are below raw-DE (51.1%) — temporal PC adds nothing and the encoder underperforms the linear ceiling. Unlike sleep, so this does NOT give the 2nd-dynamic-task confirmation. Consistent reading: MI's discriminative signal is spatial-spectral ERD (like emotion), not sequence-temporal (like sleep) — it refines the thesis ("PC helps ∝ temporal structure in the DE-window sequence") but weakens the broad cross-task foundation-model claim. Seizure (CHB-MIT), with genuine sequence-level dynamics, is the truer test.
---

# EXP-0014 — F16 — Motor imagery (BCI-IV-2a), the 2nd dynamic task

> **Status:** blocked (GPU) · **Created:** 2026-07-12 · **Agent:** claude-code · **Phase:** phase3

---

## 1. Why — hypothesis & motivation

F13 ([[EXP-0009]]) established, from four angles, that PC pretraining beats random-init
and the raw-DE ceiling on **sleep** and that the gain is specifically **temporal**. But
one dynamic task = "SSL helps sleep", not a cross-task **foundation model** (the proposal's
core claim). F16 is the second dynamic task: **motor imagery** has genuine ERD/ERS temporal
dynamics, so the thesis predicts **PC-pretrained > random-init** here too — unlike static
emotion. A confirmatory result makes the cross-task foundation-model claim defensible.

**Prediction:** pc > random-init on BCI-IV-2a (mirrors sleep); the raw-DE ceiling is strong
(51%) so the FM win, if any, is expected in pretraining-gain / label-efficiency more than peak.

## 2. Setup — exactly what was run

**Data — BCI Competition IV 2a** (BNCI Horizon mirror, `.mat`): 9 subjects, 22 EEG ch @250 Hz,
4 balanced MI classes, 288 trials/subject/session, two sessions (T=train, E=eval).
`datasets/BCI-IV-2a/A0{1..9}{T,E}.mat`.

**Feature pipeline** (`physiofm/bci_iv_2a.py`, `scripts/build_bci_dataset.py`): per trial,
take the MI window [2,6]s post trial-onset, 22 EEG ch, DE per (channel,band) over 1.0 s
windows / 0.25 s step → a **13-window sequence** per trial (n_cb=110); per-trial 4-way label;
session 1=T / 2=E. → `data/physiofm/de_features/bci_iv_2a_de.npz` (5,184 trials, balanced).

**Model/eval** (`scripts/phase2_bci_eval.py`, `scripts/run_bci.sh`): reuse PhysioFM-S; PC
pretrain + matched random-init on the MI corpus (`p_in=1`, `p_out=8` for the short trials).
Trial-level readout = frozen-encoder features mean-pooled over a trial's windows. Eval under
the canonical **session-holdout** (train T → test E, per subject) — leakage-free by design.
Report acc/macro-F1/κ mean ± std across the 9 subjects, pc vs rand vs raw-DE.

## 3. Status & run log

- 2026-07-12 — pipeline built and validated on CPU: downloaded BCI-IV-2a (18 files, ~740 MB),
  built the DE archive (5,184 trials, `(13,22,5)`, balanced 1296/class, T/E = 2592/2592, 0 NaN
  skips), registered `bci_iv_2a` in `structured_data.ARCH`, wrote loader/build/eval/driver.
  **raw-DE session-holdout baseline: 51.0 ± 14.3% / κ 0.347** (chance 25%).
- 2026-07-12 — pod restarted; ran `run_bci.sh` on the H100 (PC pretrain 60 ep + random-init +
  eval, ~2 min). **Result: PC 42.4 ≈ rand 43.6, both < raw-DE 51.1 — a null (see §4/§5).**

## 4. Results *(run 2026-07-12)*

Session-holdout (train T → test E), 9 subjects, chance 25%. `results/phase3/f16/f16_bci.csv`.
PC pretrain 60 epochs (best PC-MSE 0.211), matched random-init, `p_in=1 p_out=8`.

| Features | acc | macro-F1 | κ |
| --- | ---: | ---: | ---: |
| raw_de (logreg) | **51.1 ± 14.3** | 48.4 | 0.348 |
| physiofm_pc | 42.4 ± 9.9 | 40.2 | 0.231 |
| physiofm_rand | 43.6 ± 9.5 | 41.5 | 0.248 |

**pc − rand = −1.3 (null); FM − raw ≈ −8 (FM underperforms the linear ceiling).**

## 5. Interpretation — agent's reading

**A null for the FM, and honestly so.** Two findings: (1) PC pretraining ≈ random-init (pc
even marginally lower) — temporal PC adds nothing on MI, the mirror of the emotion null rather
than the sleep win; (2) both FM variants sit ~8 pts *below* raw-DE. Finding (1) is the clean,
readout-robust result (pc/rand share the mean-pool readout); finding (2) is partly readout-
sensitive (trial mean-pool may dilute a discriminative window) but the *pretraining* clearly
does not help regardless.

**Why this is consistent with — and sharpens — the thesis.** MI's discriminative signal is
**event-related desynchronization**: a spatial-spectral pattern (which sensorimotor channels /
mu-beta bands lose power), largely *static over the imagery period*. That makes MI like emotion
(spatial-spectral, [[EXP-0010]]) more than like sleep (sequence-temporal). So the honest thesis
is narrower than "dynamic tasks": *PC pretraining helps in proportion to temporal structure in
the DE-window sequence at the modeled timescale* — sleep has it (night-long stage dynamics),
emotion and MI largely don't.

**Strategic consequence.** With sleep the lone positive, this does **not** support a broad
cross-task "foundation model" claim. Options: (a) reframe as a mechanistic "when does temporal
SSL help EEG" study (sleep positive; emotion + MI negatives; clean mechanism); (b) test
**seizure (CHB-MIT)**, which has genuine sequence-level temporal dynamics and is the truer 2nd
positive candidate; (c) MI caveats worth a check before over-claiming the null — very short
13-window trials, weak DE-vs-CSP features, and mean-pool readout. Recommend (b) as the decisive
next test, with (a) as the fallback framing.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

- [ ] **Verified**
- **Notes / corrections:**

## 7. Commits

- (this session) new: `physiofm/bci_iv_2a.py`, `scripts/build_bci_dataset.py`,
  `scripts/phase2_bci_eval.py`, `scripts/run_bci.sh`; edited: `physiofm/structured_data.py`
  (ARCH `bci_iv_2a`).

## 8. Links

- Generalizes: [[EXP-0009]] (F13 sleep — the 1st dynamic task).
- Data: BCI Competition IV 2a — https://bnci-horizon-2020.eu/database/data-sets/001-2014/
