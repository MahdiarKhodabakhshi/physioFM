---
id: EXP-0009
title: F13 — Pre-registered temporal-PC test on sleep staging (Sleep-EDF)
status: done
created: 2026-06-28
run_date: 2026-07-11
agent: claude-code
phase: phase3-handoff
verified: no
tags: sleep-edf, phase3, temporal-pc, pre-registered
commits:
verdict: CONFIRMED (full corpus, 78 subj / 195k epochs; 3-seed). PC-pretrained beats matched random-init (+14.5 pts, p<1e-4, 5/5 folds) AND the raw-DE linear ceiling (+5.2 pts, paired-t p=0.0008, 5/5 folds); pc is seed-stable at 73.0±0.4. Label-efficiency curve: the pc−rand gain widens to +12.7 at 1% labels and pc@1% beats both baselines @100% (~100× label efficiency). The mirror image of the emotion null and the keystone positive result. Remaining: order-shuffle temporal control; a 2nd dynamic task (data-blocked).
---

# EXP-0009 — F13 — Pre-registered temporal-PC test on sleep staging (Sleep-EDF)

> **Status:** blocked (data) · **Created:** 2026-06-28 · **Run:** — · **Agent:** claude-code · **Phase:** phase3-handoff

---

## 1. Why — hypothesis & motivation

Stage 2 found temporal predictive-coding (PC) pretraining adds nothing for emotion
DE; the flip-path (EXP-0001/0002/0005–0007) showed it *does* help once LDS smoothing
is removed, i.e. **PC helps in proportion to the temporal dynamics in the features**.
That is currently a within-SEED, SEED-IV-only correlational story. F13 turns it into
a **falsifiable, pre-registered prediction** on a task with genuinely strong temporal
structure: **sleep staging** (stage transitions + sequence context, large public
data, low channel count).

**Pre-registered prediction (write before running):**
- **pretrained PC encoder > random-init** on sleep staging (the opposite of the
  smoothed-emotion null), and the pretrained−random gap is comparable to or larger
  than the un-smoothed-SEED-IV gap.
- An **order-shuffled** control (à la F2) costs more here than on emotion.

**Decision rule (from the spec):**
- pretrained > random on sleep but not on (smoothed) emotion → the clean thesis:
  *time-series FM objectives help biosignal tasks in proportion to their genuine
  temporal dynamics.* This is the paper.
- no benefit on sleep either → the temporal-PC-in-a-transformer idea is in deeper
  trouble; pivot to structure/invariance (Tier 2: F9/F11).

---

## 2. Setup — exactly what was run (PLAN — not yet executed)

**Data — Sleep-EDF Database Expanded (PhysioNet, sleep-edfx 1.0.0).**
- Sleep Cassette (SC) subset: ~153 PSG recordings / 78 subjects. Each subject has a
  `*-PSG.edf` (signals) + `*-Hypnogram.edf` (stage annotations).
- Signals: EEG **Fpz-Cz** and **Pz-Oz** at 100 Hz (+ EOG/EMG, optional).
- Place under `datasets/SLEEP-EDF/` (gitignored, like the SEED datasets).

**Feature pipeline (reuse the DE currency).**
1. New `physiofm/sleep_edf.py`: read EDF (needs `mne` or `pyedflib`), window into
   **30 s epochs**, compute **DE per (channel, band)** with the existing
   `compute_differential_entropy` over the 5 bands → canonical `DETrial`-style array
   `epochs × channels(2) × bands(5)`. Map the hypnogram to per-epoch labels
   **W / N1 / N2 / N3 / REM** (merge R&K stage 4 into N3; drop MOVEMENT/UNKNOWN).
   *No* LDS smoothing (the whole point — keep the dynamics).
2. New `scripts/build_sleep_dataset.py` (mirrors `scripts/build_de_dataset.py`) →
   `data/physiofm/de_features/sleep_edf_<key>.npz`.

**Model / eval (reuse PhysioFM-S).**
- Register `sleep_edf` in `physiofm/structured_data.py` `ARCH` and add a
  **subject-disjoint** fold mask (standard sleep protocol; per-epoch labels, not
  trial-constant — so the readout is naturally per-epoch / sequence-level, no
  trial-constant-label confound).
- Pretrain matched PC vs random-init via `scripts/phase2_pretrain.py --variant
  scratch --datasets sleep_edf --p_in {1,4,8} --p_out {1,16} --epochs 0|N`.
- Read out per-epoch stage with the frozen harness + the F2 GRU/last readouts incl.
  the shuffled-order control. Report acc / macro-F1 (+ Cohen's κ, the sleep-staging
  convention), pretrained vs random.

**Dependencies to confirm in env `xcqa`:** an EDF reader (`mne` or `pyedflib`).

---

## 3. Status & run log

- 2026-06-28 — created; **blocked**: Sleep-EDF not on disk. Unlock = download
  sleep-edfx into `datasets/SLEEP-EDF/`. See §8 for the exact command.
- 2026-06-28 — **pipeline implemented and validated** (chose option B). Installed
  `mne` 1.12.1 into env `xcqa`; started the Sleep-EDF Cassette download (`wget`,
  running in background under `datasets/SLEEP-EDF/`). New code:
  `physiofm/sleep_edf.py`, `scripts/build_sleep_dataset.py`,
  `scripts/phase2_f13_sleep.py`, `scripts/run_f13.sh`, and `sleep_edf` added to
  `physiofm/structured_data.py` ARCH. Smoke-tested on the first downloaded
  recordings (subject 0, both nights): EDF→DE→labels gives `(841, 2, 5)` per-epoch
  DE with a realistic stage distribution (W/N1/N2/N3/REM); `phase2_pretrain.py
  --datasets sleep_edf` auto-derives `n_cb=10` and trains; the frozen encoder
  yields `(epochs, 256)` per-epoch features. **Still blocked on the full corpus
  download** before the real matched PC-vs-random run.
- **Next (when download completes):** `bash scripts/run_f13.sh` (set `EPOCHS=` to
  taste) → builds the full corpus, pretrains PC + random-init, evaluates both +
  raw-DE under subject-disjoint 5-fold → `results/phase3/f13/f13_sleep.csv`.
- 2026-07-06 — **unblocked (partial).** Resumed the Sleep-EDF download and launched a
  **preliminary** run on the 18 pairs on disk so far (9 subjects, 18.8k epochs,
  realistic stage distribution). `EPOCHS=30 bash scripts/run_f13.sh`. This is an
  early directional read of the pre-registered signal (PC vs random on a dynamic
  task); the **definitive** run rebuilds on the full ~153-pair corpus once the
  download finishes.
- 2026-07-11 — **DEFINITIVE run, full corpus.** Full Sleep-EDF Cassette DE archive
  (153 recordings, **78 subjects**, 195,469 epochs) built locally, transferred to a
  rented RunPod **H100 80GB** pod (the 20GB local GPU OOM'd on whole-night sequences —
  cause was `batch × length²` attention at the emotion default batch, not model size;
  `BATCH=16` uses <3GB). Matched PC (60 epochs, best PC-MSE 0.229) vs random-init
  (`--epochs 0`), evaluated logreg under subject-disjoint 5-fold via
  `phase2_f13_sleep.py --classifiers logreg`. Results below.

---

## 4. Results

### 4a. DEFINITIVE — full corpus *(run date: 2026-07-11)*

Full Sleep-EDF Cassette: **153 recordings, 78 subjects, 195,469 epochs**.
Subject-disjoint 5-fold, logreg; acc % / macro-F1 % / κ (mean ± std over folds).
Chance = 20% (5 classes: W/N1/N2/N3/REM).

| Features | acc | macro-F1 | κ |
| --- | ---: | ---: | ---: |
| **physiofm_pc** (PC-pretrained) | **72.63 ± 2.85** | **67.57** | **0.635** |
| raw_de (linear ceiling) | 67.86 ± 2.30 | 61.88 | 0.575 |
| **physiofm_rand** (random-init) | 62.88 ± 2.43 | 56.06 | 0.509 |

**pc − rand = +9.75 pts / +0.126 κ** (pre-registered prediction confirmed).
**pc − raw_de = +4.77 pts / +0.060 κ** (FM now *beats* the raw-DE ceiling — see §5).
Results: `results/phase3/f13/f13_sleep.csv`.

### 4c. Label-efficiency curve *(run date: 2026-07-11)*

Frozen encoders; the classifier trains on a stratified fraction of each fold's
training epochs, tested on the full fold (subject-disjoint, seed-averaged at
frac<1). Accuracy %; `results/phase3/f13/f13_label_curves.csv`.

| labels | physiofm_pc | raw_de | physiofm_rand | pc − rand |
| ---: | ---: | ---: | ---: | ---: |
| 1%  | **70.9** | 67.0 | 58.2 | **+12.7** |
| 5%  | 72.5 | 67.8 | 60.8 | +11.8 |
| 10% | 72.6 | 67.7 | 61.9 | +10.8 |
| 25% | 72.8 | 67.9 | 62.5 | +10.2 |
| 50% | 72.7 | 67.9 | 62.8 | +9.9 |
| 100%| 72.6 | 67.9 | 62.9 | +9.8 |

The pretraining gain **widens as labels shrink** (+9.8 → +12.7 at 1%), and
**PC-FM at 1% labels (70.9%) beats both raw-DE and random-init at 100% labels**
(67.9 / 62.9) — pretraining is worth ~100× the labeled data. PC-FM is nearly flat
(−1.7 pts from 100%→1%) while random-init falls hardest (−4.7).

### 4d. Multi-seed robustness + paired tests *(run date: 2026-07-11)*

Seeds 1/2/3 (fresh PC + matched random-init each), subject-disjoint 5-fold, logreg
(`results/phase3/f13/multiseed/`). Accuracy % mean ± std across seeds:

| Feature | acc (mean ± std over seeds) |
| --- | ---: |
| physiofm_pc | **73.03 ± 0.38** |
| raw_de | 67.86 ± 0.00 (deterministic) |
| physiofm_rand | 58.51 ± 1.05 |

Paired per-fold tests (seed-averaged, matched folds):
- **pc vs raw_de: +5.17 pts, paired-t p=0.0008, wins 5/5 folds** → the peak-accuracy
  win is significant, not seed-luck. (Wilcoxon p=0.0625 is the *floor* for n=5, so the
  5/5 sweep + paired-t is the informative test.)
- **pc vs rand: +14.52 pts, paired-t p<0.0001, wins 5/5 folds.**

PC-FM is seed-stable (±0.38); random-init varies more (±1.05, as expected for a random
encoder) but is always far below. The seed-42 headline (§4a) sits inside this band.

### 4b. Preliminary — partial corpus *(run date: 2026-07-06, superseded)*

Partial corpus (18 recordings, **9 subjects**, 18.8k epochs; 30-epoch pretrain).

| Features | logreg acc / κ | mlp_bal acc / κ |
| --- | ---: | ---: |
| raw_de | 72.10 ± 3.52 / 0.628 | 67.41 / 0.560 |
| **physiofm_pc** | **68.79 ± 3.19 / 0.588** | 66.09 / 0.548 |
| **physiofm_rand** | **50.01 ± 7.00 / 0.350** | 52.65 / 0.350 |

---

## 5. Interpretation — agent's reading *(definitive, full corpus)*

**The pre-registered prediction is confirmed at full scale, and more strongly than
the preliminary suggested.** On 78 subjects, `physiofm_pc` (72.6% / κ 0.635) beats
matched `physiofm_rand` (62.9% / κ 0.509) by **+9.8 pts / +0.13 κ**. This is the
mirror image of emotion (EXP-0001/0011), where PC ≈ random — confirming the thesis
that *PC pretraining helps in proportion to a task's temporal dynamics*. The gap is
smaller than the preliminary's +19 because random-init rose from a noisy 50.0
(9 subj) to a reliable 62.9 (78 subj); the +9.8 across 78 subjects is the trustworthy
number.

**The preliminary's main caveat is now reversed — the FM beats the raw-DE ceiling on
peak accuracy.** At 9 subjects raw-DE (72.1) *edged* the FM (68.8), so the win was
"pretraining-gain, not peak." At full scale that flips: `physiofm_pc` (72.6)
**exceeds** raw-DE (67.9) by **+4.8 pts / +0.06 κ**. Raw-DE *degraded* with more
subjects (67.9 vs 72.1 — cross-subject DE distribution shift), while the FM *improved*
(72.6 vs 68.8 — more pretraining data). So the PC-pretrained FM is the single best
model here, beating both the random-init control and the linear ceiling.

**This is the keystone positive result** the Option-A framing rests on: a genuine
foundation-model win on a dynamic task, opposite the static-emotion null.

**Label-efficiency confirmed (§4c):** the F7-analog curve shows the pretraining
gain *widening* at low labels (+9.8→+12.7) and PC-FM at 1% labels beating both
baselines at 100% — the expected headline SSL result, now measured.

**Robustness confirmed (§4d):** 3-seed repeat gives pc 73.03 ± 0.38 (seed-stable),
and paired per-fold tests make both wins significant — pc > raw_de (+5.2, p=0.0008,
5/5 folds) and pc > rand (+14.5, p<0.0001, 5/5 folds). The single-seed headline is not
seed-luck, and the peak-accuracy win over raw-DE is statistically sound.

**Remaining:** (1) an order-shuffle temporal control (pre-registered §1) to show the
gain is specifically *temporal*. (2) A second dynamic task for the cross-task claim
(data-blocked).

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

<!-- to be filled as the sleep pipeline lands -->
- (uncommitted as of 2026-06-28) new: `physiofm/sleep_edf.py`,
  `scripts/build_sleep_dataset.py`, `scripts/phase2_f13_sleep.py`,
  `scripts/run_f13.sh`; edited: `physiofm/structured_data.py` (ARCH `sleep_edf`).

---

## 8. Links

- Related entries: [[EXP-0001]], [[EXP-0002]], [[EXP-0005]] (the flip-path this generalizes)
- Spec: `docs/PhysioFM_Stage2_FollowUp_Experiments.md` (F13, Tier 4)
- Data: Sleep-EDF Expanded — https://physionet.org/content/sleep-edfx/1.0.0/
  - Sleep Cassette only (~recommended start):
    `wget -r -N -c -np -nH --cut-dirs=4 https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/ -P datasets/SLEEP-EDF/`
  - (full DB ≈ 8 GB; the SC subset is the bulk and the standard benchmark.)
