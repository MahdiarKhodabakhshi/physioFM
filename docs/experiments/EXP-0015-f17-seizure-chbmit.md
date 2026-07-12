---
id: EXP-0015
title: F17 — Seizure detection (CHB-MIT), the 2nd dynamic-task confirmation
status: done
created: 2026-07-12
run_date: 2026-07-12
agent: claude-code
phase: phase3
verified: no
tags: chb-mit, seizure, phase3, temporal-pc, second-dynamic-task, cross-task
commits:
verdict: CONFIRMED (5-patient subset). PC pretraining beats matched random-init massively on seizure — balanced-acc +21.5 pts, AUC 0.916 vs 0.574 (random ≈ chance) — the 2nd dynamic-task win after sleep and the mirror of the motor-imagery null. The FM matches the raw-DE ceiling (AUC ~0.91; trades sensitivity 43 for specificity 96 vs raw's 54/85). This is the cross-task evidence: PC helps on BOTH sequence-temporal tasks (sleep, seizure), null on BOTH spatial-spectral tasks (emotion, MI). Caveats: 5-patient subset, single seed, LOPO — full 24-patient corpus + multi-seed for the paper.
---

# EXP-0015 — F17 — Seizure detection (CHB-MIT), the 2nd dynamic-task confirmation

> **Status:** done · **Created/Run:** 2026-07-12 · **Agent:** claude-code · **Phase:** phase3

---

## 1. Why — hypothesis & motivation

After motor imagery came back a **null** ([[EXP-0014]]) — its signal is spatial-spectral
ERD (emotion-like), not sequence-temporal — the cross-task foundation-model claim needed a
task with *genuine* sequence-level temporal dynamics. **Seizure detection** is that task: the
EEG evolves as a seizure develops, over long recordings. Pre-registered prediction: PC-pretrained
> random-init (like sleep [[EXP-0009]]), because temporal PC can model seizure onset/evolution.

## 2. Setup — exactly what was run

**Data — CHB-MIT** (PhysioNet, AWS S3 mirror), 5-patient subset chb01/02/03/05/08. 256 Hz,
18-ch common bipolar montage. `physiofm/chbmit.py` parses each `-summary.txt` for seizure
start/end times, reads EDFs, computes **per-2 s-epoch DE** + **per-epoch binary label**
(seizure/interictal). Archive: 174 recordings, 307,482 epochs, **0.40% seizure** (severely
imbalanced), n_cb=90, ~1800-epoch sequences. Two dataset quirks fixed: 303 B placeholder EDFs
(skip), and the `T8-P8-0/-1` duplicate-channel naming (match de-duplicated base).

**Model/eval** — PhysioFM-S PC-pretrain (`p_in=1 p_out=16`, 60 ep) + matched random-init on the
seizure corpus; `scripts/phase2_chbmit_eval.py` under **leave-one-patient-out** with
**imbalance-aware** metrics (balanced-acc / sensitivity / specificity / ROC-AUC; class-weighted
logreg). `bash scripts/run_chbmit.sh`.

## 3. Status & run log

- 2026-07-12 — pipeline built + validated; subset downloaded via S3 (PhysioNet HTTP was 0.10 MB/s
  → S3 6.8 MB/s); DE archive built (174 rec / 307k epochs / 0.40% seizure); pretrain+eval on the
  H100 (~3 min). Result below.

## 4. Results *(run 2026-07-12)*

Leave-one-patient-out, 5 patients, seizure = 0.40% of epochs. `results/phase3/f17/f17_chbmit.csv`.

| Features | balanced acc | sensitivity | specificity | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| **physiofm_pc** | 69.5 ± 14.6 | 43.2 | 95.8 | **0.916 ± 0.055** |
| raw_de | 69.5 ± 8.9 | 53.7 | 85.3 | 0.913 ± 0.047 |
| **physiofm_rand** | 47.9 ± 2.7 | 11.1 | 84.7 | **0.574 ± 0.091** |

**pc − rand: +21.5 balanced-acc, +0.34 AUC** (random-init ≈ chance). **pc ≈ raw_de** (AUC 0.916
vs 0.913; pc trades sensitivity for specificity).

## 5. Interpretation — agent's reading

**The pre-registered prediction is confirmed, strongly, and it rescues the cross-task claim.**
On seizure, PC pretraining is *essential*: the matched random-init encoder is near-chance
(AUC 0.574) while the PC-pretrained encoder reaches AUC 0.916 — a **+0.34 AUC / +21.5 bal-acc**
gap. This is the sleep pattern (PC ≫ random on a sequence-temporal task), the mirror of the
motor-imagery and emotion nulls. So the thesis now has **two independent dynamic-task wins
(sleep, seizure)** vs **two spatial-spectral nulls (emotion, MI)** — a genuine, defensible
cross-task foundation-model result rather than "SSL helps sleep."

The FM **matches but does not beat** the raw-DE ceiling on AUC/bal-acc (raw per-epoch band power
is already a strong seizure feature), trading sensitivity (43 vs 54) for specificity (96 vs 85).
So, as on sleep, the FM's headline value is the huge **pretraining gain over random** (and
expected label-efficiency), not peak over a strong linear baseline.

**Caveats:** 5-patient subset, single seed, LOPO on few patients (hence the ±14.6 bal-acc
spread). Needs the full 24-patient corpus + multi-seed + a label-efficiency curve to be
paper-grade — but the direction is unambiguous.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

- [ ] **Verified**
- **Notes / corrections:**

## 7. Commits

- (this session) new: `physiofm/chbmit.py`, `scripts/build_chbmit_dataset.py`,
  `scripts/phase2_chbmit_eval.py`, `scripts/run_chbmit.sh`; edited `structured_data.py` (ARCH).

## 8. Links

- Confirms the thesis on a 2nd dynamic task: [[EXP-0009]] (sleep +), against [[EXP-0014]] (MI −).
- Data: CHB-MIT — https://physionet.org/content/chbmit/1.0.0/
