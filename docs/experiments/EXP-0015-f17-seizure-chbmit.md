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
verdict: CONFIRMED but MODEST on the full 24-patient corpus. PC pretraining beats matched random-init (+8.1 bal-acc, +0.082 AUC: 0.822 vs 0.740) and slightly beats raw-DE (+3.1 bal-acc) — the 2nd sequence-temporal-task win (sleep-like +8/+10), so the cross-task claim holds (PC helps on sleep+seizure, null on emotion+MI). IMPORTANT: the 5-patient subset badly overstated it (subset pc−rand +0.34 AUC with random≈chance was an easy-patient artifact; full-corpus random recovers to 0.74). Huge ±0.20 per-patient AUC spread. Owed: label-efficiency curve (in progress), multi-seed, paired per-patient test.
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

## 4. Results

### 4a. Full corpus — 24 patients *(run 2026-07-12, definitive)*

Leave-one-patient-out, **all 24 patients**, 682 recordings / 1.76M epochs / 0.32% seizure.
`results/phase3/f17/f17_chbmit_full.csv`.

| Features | balanced acc | sensitivity | specificity | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| **physiofm_pc** | 75.5 ± 14.3 | 65.2 | 85.8 | **0.822 ± 0.204** |
| raw_de | 72.4 ± 15.6 | 67.1 | 77.7 | 0.806 ± 0.191 |
| **physiofm_rand** | 67.4 ± 13.7 | 51.2 | 83.7 | **0.740 ± 0.202** |

**pc − rand: +8.1 bal-acc, +0.082 AUC. pc − raw: +3.1 bal-acc, +0.016 AUC.** PC still beats
both — the cross-task claim holds — but the effect is **modest**, and the per-patient spread is
huge (±0.20 AUC: seizure detection varies wildly by patient). A paired per-patient test is
needed to call pc−raw significant.

### 4b. 5-patient subset *(run 2026-07-12, superseded — overstated the effect)*

| Features | bal acc | sens | spec | AUC |
| --- | ---: | ---: | ---: | ---: |
| physiofm_pc | 69.5 ± 14.6 | 43.2 | 95.8 | 0.916 ± 0.055 |
| raw_de | 69.5 ± 8.9 | 53.7 | 85.3 | 0.913 ± 0.047 |
| physiofm_rand | 47.9 ± 2.7 | 11.1 | 84.7 | 0.574 ± 0.091 |

The subset showed pc−rand **+0.34 AUC** (random ≈ chance) — **an artifact of a few easy
patients**. On the full corpus random-init recovers to AUC 0.74 (more patients → the random
encoder + logreg has far more training data), shrinking the gap to +0.082. A cautionary tale
about small-subset previews.

## 5. Interpretation — agent's reading *(full corpus)*

**The pre-registered prediction is confirmed — PC > random on seizure — but the honest,
full-corpus effect is modest, not the dramatic subset number.** On all 24 patients the
PC-pretrained encoder beats matched random-init by **+8.1 bal-acc / +0.082 AUC** and slightly
beats the raw-DE ceiling (+3.1 bal-acc). This is still the sleep pattern (PC > random on a
sequence-temporal task, unlike the emotion/MI nulls), so the **cross-task claim holds**: PC helps
on both sequence-temporal tasks (sleep, seizure), null on both spatial-spectral tasks
(emotion, MI). But the magnitude here (+8 bal-acc) is sleep-like (+10), not the subset's +21.

**Key lesson: the 5-patient subset badly overstated the effect.** Random-init went from
near-chance (0.57 AUC) on the easy subset to 0.74 on the full corpus — the pretraining "rescue"
was mostly a small-sample artifact. The trustworthy number is the full-corpus +0.082 AUC, with
a large ±0.20 per-patient spread that a paired test must address before claiming significance.

**Open (in progress):** the **label-efficiency curve** is the decisive follow-up — if random-init
only caught up because it had abundant data, PC should re-open a gap at low labels (as on sleep).
That, plus multi-seed and a paired per-patient test, determines how strong the seizure leg is.

**Caveats:** single seed; LOPO with huge inter-patient variance (±0.20 AUC); a paired
per-patient significance test still owed.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

- [ ] **Verified**
- **Notes / corrections:**

## 7. Commits

- (this session) new: `physiofm/chbmit.py`, `scripts/build_chbmit_dataset.py`,
  `scripts/phase2_chbmit_eval.py`, `scripts/run_chbmit.sh`; edited `structured_data.py` (ARCH).

## 8. Links

- Confirms the thesis on a 2nd dynamic task: [[EXP-0009]] (sleep +), against [[EXP-0014]] (MI −).
- Data: CHB-MIT — https://physionet.org/content/chbmit/1.0.0/
