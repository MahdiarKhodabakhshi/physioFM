---
id: EXP-0018
title: F18 — Seizure prediction (pre-ictal vs interictal, CHB-MIT), the theory-driven test (retro-logged)
status: done
created: 2026-07-29
run_date: 2026-07-29
agent: claude-code
phase: phase3
verified: no
tags: chb-mit, seizure-prediction, mechanism-test, negative-result, retro-logged
commits: db3c93e, a3ac42d
verdict: PREDICTION FAILED. The objective-misalignment mechanism (EXP-0017 §4e) predicted that seizure PREDICTION — the one EEG task whose downstream objective is itself forecasting — is where PC pretraining must finally pay off. Patient-specific leave-one-seizure-out (21 patients, 140 pre-ictal events, frozen probe): raw-DE 65.15 bal-acc / 0.710 AUC, PhysioFM pretrained 66.79 / 0.769, PhysioFM random-init 71.34 / 0.793 -> random-init beats pretraining (-4.6 bal-acc / -0.024 AUC) even in the frozen regime that flatters pretraining everywhere else. Both transformer arms beat raw-DE (+0.06-0.08 AUC): the architecture works, the pretraining does not. A first cross-patient (LOPO) attempt put every arm incl. raw-DE at chance (AUC 0.49-0.55) and was discarded as uninformative (pre-ictal signatures are patient-specific; the literature protocol is patient-specific). Single seed; results CSV was written on the RunPod pod (results/phase3/f18/) and is not on local disk — numbers here are from docs/FINAL_REPORT.md §5.
---

# EXP-0018 — F18 — Seizure prediction (retro-logged on 2026-08-18 from the 2026-07-29 session)

> **Status:** done · **Run:** 2026-07-29 · **Agent:** claude-code · **Phase:** phase3
> Retro-logged: this experiment was run and reported in the session transcript and
> `docs/FINAL_REPORT.md` §5 but never got a notebook entry.

## 1. Why
EXP-0017 §4e found the pretext IS learned but pretext skill anti-correlates with transfer, and
proposed the mechanism "PC pretraining helps only when what is predictable is also what is
discriminative". Seizure *prediction* (decide from EEG now whether a seizure is coming) is the
one task where the downstream objective is forecasting, so the mechanism made a falsifiable
prediction: PC-pretrained > random-init here.

## 2. Setup
`scripts/build_chbmit_prediction.py` relabels the existing CHB-MIT DE (2 s epochs): pre-ictal =
30-min window ending 5 min before onset (SPH 5 min, SOP 30 min); ictal/post-ictal excluded (-1);
interictal from seizure-free recordings. 6.0 % pre-ictal (104,934 epochs). Encoders = the F17
PC-pretrained / random-init models (frozen). `scripts/phase2_seizure_prediction.py`: patient-
specific leave-one-seizure-out over the patient's own seizures (21 patients with >= 2 seizures,
140 events), balanced logistic regression, output `results/phase3/f18/f18_seizure_prediction.csv`
(pod only).

## 3. Status & run log
- 2026-07-29 — cross-patient LOPO run: all arms at chance incl. raw-DE (0.49–0.55 AUC) → protocol
  error (pre-ictal signatures are individual), discarded as uninformative.
- 2026-07-29 — patient-specific run (a3ac42d): results below.

## 4. Results *(2026-07-29; from FINAL_REPORT §5 / transcript — CSV on the pod only)*

| arm | bal-acc | AUC |
| --- | ---: | ---: |
| raw-DE (no model) | 65.15 | 0.710 |
| PhysioFM (pretrained), frozen | 66.79 | 0.769 |
| PhysioFM (no pretrain), frozen | **71.34** | **0.793** |

## 5. Interpretation — agent's reading
The mechanism's prediction failed: random-init beats pretrained by 0.024 AUC / 4.6 bal-acc, in
the frozen regime. Both transformer arms beat raw-DE — the structured-patch causal architecture
works; predictive-coding pretraining on DE does not. This retired the objective-misalignment
story as an *explanation* and left the simplest reading: a random-init structured transformer is
already a strong EEG encoder. Caveats: single seed; patient-specific protocol is easier than
cross-patient and NOT comparable to F17.

## 6. ✅ Your verification — *(reserved for Mahdiar)*
- [ ] **Verified**
- **Notes / corrections:**

## 7. Commits
- db3c93e (task + first protocol), a3ac42d (patient-specific protocol)

## 8. Links
- [[EXP-0017]] (mechanism it tested), `docs/FINAL_REPORT.md` §5.
