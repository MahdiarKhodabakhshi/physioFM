---
id: EXP-0019
title: F19 — VitalDB anaesthesia depth (loader only; not run)
status: blocked
created: 2026-07-29
run_date: 
agent: claude-code
phase: phase3
verified: no
tags: vitaldb, anaesthesia, not-run
commits: d410e1a
verdict: NOT RUN. `physiofm/vitaldb_anesthesia.py` (2-ch BIS-monitor EEG @128 Hz, 10 s epochs, BIS discretised into 5 depth bands, SQI/artefact rejection) was written and committed; the build script, the ~100–200-case download, ARCH registration and — mandatory first — the linear-saturation check (BIS is itself a spectral formula) were never done because the project pivoted to the final report and then to the next-phase plan (EXP-0020..0023). Kept as a documented option.
---

# EXP-0019 — F19 — VitalDB anaesthesia depth (loader only)

> **Status:** blocked (never run) · **Agent:** claude-code · **Phase:** phase3

## 1. Why
After EXP-0017/0018 the remaining "theory-friendly" per-epoch task was anaesthesia depth
(continuously evolving state, per-second BIS label, 2-ch EEG — structurally like sleep).

## 2. Setup (planned)
`physiofm/vitaldb_anesthesia.py`: tracks BIS/EEG1_WAV, BIS/EEG2_WAV, BIS/BIS, BIS/SQI; 10 s
epochs; labels burst-suppression / deep / general / moderate / awake; MIN_SQI 50, |EEG| < 500 µV.
Planned ladder: raw-DE linear-saturation check FIRST (stop if BIS is linearly recoverable), then
PC vs random-init vs raw-DE, fine-tuned, patient-disjoint.

## 3. Status & run log
- 2026-07-29 — loader written and committed (d410e1a); nothing downloaded or run.
- 2026-08-18 — superseded in priority by the next-phase plan (EXP-0020..0023).

## 4. Results
— none —

## 5. Interpretation
n/a

## 6. ✅ Your verification — *(reserved for Mahdiar)*
- [ ] **Verified**

## 7. Commits
- d410e1a

## 8. Links
- `docs/FINAL_REPORT.md`; [[EXP-0018]].
