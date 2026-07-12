---
id: EXP-0014
title: F16 — Motor imagery (BCI-IV-2a), the 2nd dynamic task
status: blocked
created: 2026-07-12
run_date:
agent: claude-code
phase: phase3
verified: no
tags: bci-iv-2a, motor-imagery, phase3, temporal-pc, second-dynamic-task
commits:
verdict: (in progress) pipeline built + validated, DE archive ready, raw-DE baseline 51.0±14.3% (session-holdout, chance 25%). PC-vs-random pretraining is GPU-blocked (pod stopped) — one command (`bash scripts/run_bci.sh`) when the pod is back. Turns "SSL helps sleep" into a cross-task claim if PC>random holds here too.
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
  **raw-DE session-holdout baseline: 51.0 ± 14.3% / κ 0.347** (chance 25%). PC-vs-random
  pretraining **blocked** — the pod was stopped; unblock = restart pod, `bash scripts/run_bci.sh`.

## 4. Results *(partial — baseline only)*

Session-holdout (train T → test E), 9 subjects, chance 25%. `results/phase3/f16/f16_bci.csv`.

| Features | acc | macro-F1 | κ |
| --- | ---: | ---: | ---: |
| raw_de (logreg) | 51.00 ± 14.31 | 48.32 | 0.347 |
| physiofm_pc | _pending pod_ | | |
| physiofm_rand | _pending pod_ | | |

## 5. Interpretation — agent's reading *(partial)*

The 51% raw-DE baseline is in the expected BCI-IV-2a range for simple band features (CSP-based
winners reach ~60–70%); the large ±14 std is the known "BCI-illiteracy" inter-subject spread.
The pipeline reproduces the standard leakage-free protocol, so the pending PC-vs-random result
is a clean test of the thesis on a 2nd dynamic task. If pc > random here as on sleep, the
cross-task foundation-model claim is supported; if pc ≈ random, MI's within-trial dynamics
(only 13 windows) may be too short for temporal PC — itself an informative boundary on the thesis.

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
