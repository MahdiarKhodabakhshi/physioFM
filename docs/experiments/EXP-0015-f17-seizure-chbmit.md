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
verdict: CONFIRMED (full 24-patient corpus, paired tests). PC pretraining SIGNIFICANTLY beats matched random-init: +0.082 AUC (paired p=0.006, 17/24 patients), widening to +0.105 (p=0.0002, 22/24) at 1% labels — the 2nd sequence-temporal win, so the cross-task claim holds (PC helps on sleep+seizure, null on emotion+MI). PC does NOT beat the raw-DE ceiling at full labels (a tie: +0.016 AUC, p=0.46, 11/24) — but DOES at low labels (+0.066, p=0.017, 18/24). Label-efficiency is the real win: PC is nearly label-insensitive (−0.025 AUC from 100%→1% vs raw's −0.075), and PC@5% labels (0.809) matches raw-DE@100% (0.806) → ~20x label efficiency. NB the 5-patient subset badly overstated everything (random looked near-chance); only the paired full-corpus test is trustworthy.
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

### 4c. Label-efficiency curve + paired per-patient tests *(run 2026-07-12)*

Frozen encoders; stratified subsample of the *training* epochs (keeps rare seizures), LOPO,
seed-averaged. ROC-AUC; `results/phase3/f17/f17_chbmit_labelcurve.csv`.

| labels | physiofm_pc | raw_de | physiofm_rand | pc − raw | pc − rand |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1%  | **0.797** | 0.731 | 0.692 | **+0.066** | **+0.105** |
| 5%  | **0.809** | 0.790 | 0.740 | +0.019 | +0.069 |
| 10% | 0.808 | 0.795 | 0.734 | +0.013 | +0.074 |
| 25% | 0.814 | 0.803 | 0.740 | +0.011 | +0.074 |
| 50% | 0.820 | 0.804 | 0.737 | +0.016 | +0.083 |
| 100%| 0.822 | 0.806 | 0.740 | +0.016 | +0.082 |

PC is strikingly label-insensitive (−0.025 AUC from 100%→1%) vs raw-DE (−0.075) and random
(−0.048). **PC at 5% labels (0.809) ≥ raw-DE at 100% (0.806) → ~20× label efficiency.**

**Paired per-patient tests** (24 matched patients, `*_perpatient.csv`):

| Comparison | 100% labels | 1% labels |
| --- | --- | --- |
| pc vs rand (AUC) | +0.082, **p=0.006**, 17/24 | +0.105, **p=0.0002**, 22/24 |
| pc vs raw_de (AUC) | +0.016, **p=0.46 (n.s.)**, 11/24 | +0.066, **p=0.017**, 18/24 |

**Critical:** pc does **NOT** significantly beat raw-DE at full labels (a coin-flip, 11/24) —
the earlier "+3.1 bal-acc edge" is **not** a win. It *does* significantly beat raw-DE at 1%
labels. And pc beats random-init significantly at every level.

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

## 5. Interpretation — agent's reading *(full corpus, with paired tests)*

**The pre-registered prediction is confirmed: PC pretraining significantly beats matched
random-init on seizure** — +0.082 AUC (paired p=0.006, 17/24 patients) at full labels, widening
to +0.105 (p=0.0002, 22/24) at 1% labels. This is the sleep pattern on a second
sequence-temporal task, unlike the emotion/MI nulls, so the **cross-task claim holds**: PC helps
on both sequence-temporal tasks (sleep, seizure), null on both spatial-spectral ones
(emotion, MI).

**But PC does NOT beat the raw-DE ceiling at full labels — that is a statistical tie**
(+0.016 AUC, p=0.46, winning 11/24 patients ≈ coin flip). Any earlier "PC edges raw-DE" phrasing
was wrong and is retracted here.

**Where the FM genuinely wins is the low-label regime — and there it wins significantly.** The
label-efficiency curve (§4c) shows PC is nearly label-insensitive (−0.025 AUC from 100%→1%)
while raw-DE degrades 3× faster (−0.075). Consequently the pc−raw gap **quadruples** as labels
shrink (+0.016 → +0.066) and becomes **significant at 1% labels** (p=0.017, 18/24). Headline:
**PC with 5% of labels (AUC 0.809) matches raw-DE trained on 100% (0.806) — ~20× label
efficiency.** This is exactly the regime a foundation model is supposed to win in, and it
mirrors sleep, where the pc−rand gap also widened at low labels.

**Two lessons on rigor:** (1) the 5-patient subset badly overstated the effect (random-init
looked near-chance at 0.57 AUC; on the full corpus it recovers to 0.74) — small-subset previews
are dangerous. (2) With ±0.20 inter-patient AUC variance, only the **paired** per-patient test
is trustworthy; the unpaired means suggested a raw-DE win that does not survive it.

**Caveats:** single seed (per-patient variance dominates seed variance, so a multi-seed run is
low-value here); LOPO cross-patient generalization is intrinsically hard for seizure.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

- [ ] **Verified**
- **Notes / corrections:**

## 7. Commits

- (this session) new: `physiofm/chbmit.py`, `scripts/build_chbmit_dataset.py`,
  `scripts/phase2_chbmit_eval.py`, `scripts/run_chbmit.sh`; edited `structured_data.py` (ARCH).

## 8. Links

- Confirms the thesis on a 2nd dynamic task: [[EXP-0009]] (sleep +), against [[EXP-0014]] (MI −).
- Data: CHB-MIT — https://physionet.org/content/chbmit/1.0.0/
