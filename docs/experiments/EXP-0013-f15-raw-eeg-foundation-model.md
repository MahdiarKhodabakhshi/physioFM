---
id: EXP-0013
title: F15 — Raw-EEG foundation model (escape the DE bottleneck)
status: blocked
created: 2026-06-28
run_date:
agent: claude-code
phase: phase3
verified: no
tags: raw-eeg, foundation-model, masked-modeling, clisa, labram, data-blocked
commits:
verdict: (pre-registered) pretraining on RAW EEG (not DE) opens headroom the DE ceiling denied — if raw-EEG-FM beats the DE linear ceiling (esp. on LOSO), the DE feature was the limiter; if it ties, the limit is the task/data. BLOCKED on raw EEG download.
---

# EXP-0013 — F15 — Raw-EEG foundation model (escape the DE bottleneck)

> **Status:** blocked (data) · **Created:** 2026-06-28 · **Run:** — · **Agent:** claude-code · **Phase:** phase3

---

## 1. Why — hypothesis & motivation

The deepest root cause from the diagnosis: **DE is a lossy, hand-crafted bottleneck
that is already linearly saturated** (EXP-0010 F10). Every SSL pretext we tried died
or tied *on DE* (EXP-0011 F9), because DE throws away the raw waveform and keeps ~5
band-power numbers per channel — there is little left for a representation to learn.
Every EEG foundation model that actually works (LaBraM, EEGPT, BrainGPT, NeurIPT,
Uni-NTFM) pretrains on the **raw signal** and learns its own features, on large
corpora, with masked / tokenized objectives.

**Hypothesis.** Pretraining on **raw EEG** (instead of DE) with a masked /
contrastive objective — optionally plus the subject-invariance term from F14 — gives
the encoder information DE discards, opening headroom above the DE linear ceiling,
**most visibly on the unsaturated cross-subject (LOSO) regime**.

**Pre-registered prediction & decision rule.**
- raw-EEG-FM **>** DE linear ceiling (esp. LOSO) → the **DE feature was the
  limiter**; the FM bet works once it operates on the raw signal. This is the
  positive foundation-model result the proposal wanted, relocated to the right input.
- raw-EEG-FM **≈** DE ceiling → the limit is the **task/data**, not the feature; the
  saturation is fundamental and the project's contribution stays the diagnosis.

This is the full "best shot" path: **raw EEG + masked/contrastive SSL + subject
invariance + scale**, the combination the literature converges on.

---

## 2. Setup — exactly what to run (PLAN — not yet executed; data-blocked)

**⚠️ Data block.** Raw EEG is **not on disk** — `datasets/` holds only DE features
(SEED `ExtractedFeatures_4s`, SEED-IV `eeg_feature_smooth`, SEED-V
`EEG_DE_features`). SEED's raw waveforms (`Preprocessed_EEG/`, 62ch @ 200 Hz) are a
separate download from the BCMI/SEED portal (registration required). **Unlock =
obtain the SEED raw EEG corpus into `datasets/SEED*/Preprocessed_EEG/`.**

**New code to build (once data lands):**
1. `physiofm/raw_eeg.py` — load raw EEG `.mat`, window (e.g. 1 s @ 200 Hz), light
   filtering; trial/subject/label bookkeeping mirroring `physiofm/de.py`.
2. `scripts/build_raw_eeg_dataset.py` — write a raw-EEG archive (gitignored).
3. A small **EEG encoder**: conv patch tokenizer → causal/transformer stack
   (LaBraM-lite). Two SSL objectives to compare:
   - **masked reconstruction / masked spectral prediction** (BERT-/MAE-style);
   - **CLISA contrastive inter-subject alignment** (same-emotion-different-subject
     positives) — directly couples F14's invariance idea to the raw signal.
4. `scripts/phase2_f15_raweeg.py` — freeze encoder, read out emotion through the same
   eval harness; compare **raw-EEG-FM vs DE-FM (EXP-0001) vs raw-DE linear ceiling**,
   on **subject-dependent AND LOSO**.

- **Pretraining corpus:** raw EEG across SEED + SEED-IV + SEED-V (labels unused);
  cross-corpus + scale per the FM literature.
- **Output dir:** `results/phase3/f15/`.

---

## 3. Status & run log

- 2026-06-28 — created (pre-registered); **blocked**: raw EEG not on disk (only DE
  features present). Unlock = download SEED `Preprocessed_EEG`. Run **F14 / EXP-0012
  first** (cheap, existing infra) — it decides whether subject-invariance is the
  lever before paying for the raw-EEG rebuild.

---

## 4. Results  *(run date: TBD)*

_(pending data + build)_

---

## 5. Interpretation — agent's reading

_(pending results — do not fill until the run exists)_

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

<!-- to be filled as the raw-EEG pipeline lands -->

---

## 8. Links

- Motivated by: [[EXP-0010]] (DE saturation), [[EXP-0011]] (no SSL helps on DE)
- Pairs with: [[EXP-0012]] (F14 — invariance objective; run first)
- Refs: LaBraM / EEG-FM review (arXiv:2507.11783); EEGPT (NeurIPS 2024);
  EEG-SCMM (arXiv:2408.09186); CLISA (arXiv:2109.09559)
- ⚠️ Eval honesty: strict inductive LOSO; beware transductive-DA inflation — cf.
  [[EXP-0008]].
