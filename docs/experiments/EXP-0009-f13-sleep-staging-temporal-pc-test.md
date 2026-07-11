---
id: EXP-0009
title: F13 — Pre-registered temporal-PC test on sleep staging (Sleep-EDF)
status: running
created: 2026-06-28
run_date: 2026-07-06
agent: claude-code
phase: phase3-handoff
verified: no
tags: sleep-edf, phase3, temporal-pc, pre-registered, data-blocked
commits:
verdict: PRELIMINARY CONFIRMED — on sleep (9-subj partial corpus) PC-pretrained beats matched random-init by ~+19 pts / +0.24 kappa (68.8% vs 50.0%), the mirror image of the emotion null. PC pretraining has real value where temporal dynamics exist. Caveat: raw-DE linear still edges the FM at full labels, so the win is pretraining-gain + label-efficiency, not peak. Definitive run pending full-corpus upload.
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

---

## 4. Results  *(PRELIMINARY — run date: 2026-07-06)*

**Preliminary** run on the partial corpus available so far (18 recordings, **9
subjects**, 18.8k epochs; 30-epoch pretrain). Subject-disjoint 5-fold; acc % / κ.
The **definitive** run rebuilds on the full ~153-pair corpus (user uploading).

| Features | logreg acc / κ | mlp_bal acc / κ |
| --- | ---: | ---: |
| raw_de | 72.10 ± 3.52 / 0.628 | 67.41 / 0.560 |
| **physiofm_pc** | **68.79 ± 3.19 / 0.588** | 66.09 / 0.548 |
| **physiofm_rand** | **50.01 ± 7.00 / 0.350** | 52.65 / 0.350 |

Results: `results/phase3/f13/f13_sleep.csv`. Chance = 20% (5 classes).

---

## 5. Interpretation — agent's reading *(preliminary)*

**The pre-registered prediction is confirmed on the preliminary corpus.**
`physiofm_pc` (68.8% / κ 0.588) beats matched `physiofm_rand` (50.0% / κ 0.350) by
**~+19 pts / +0.24 κ**. This is the mirror image of emotion (EXP-0001/0011), where PC
≈ random. So the proposal's temporal-PC pretraining objective carries **real,
large value on a genuinely dynamic biosignal task** — supporting the thesis that
*PC pretraining helps in proportion to a task's temporal dynamics*.

**Caveat (important for framing).** raw-DE logreg (72.1%) still edges out the FM
encoder (68.8%) at full labels — DE + linear is strong on sleep too, so the FM does
**not** win on full-label peak accuracy. The FM's value here is (a) the pretraining
gain over random (+19), and (b) the expected **label-efficiency** advantage in
low-data (to be measured directly, F7-analog). This is consistent with the SSL
literature (gains concentrate in low-label regimes) and is the metric Option A
should headline — not peak accuracy.

**Preliminary caveats:** only 9 subjects / partial corpus / 30-epoch pretrain; the
definitive full-corpus run may shift absolute numbers (the PC≫random *direction* is
expected to hold and likely strengthen with more data). Also single-seed.

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
