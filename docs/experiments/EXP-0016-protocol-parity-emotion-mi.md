---
id: EXP-0016
title: Protocol parity for emotion + motor imagery (and two corrections it forced)
status: done
created: 2026-07-28
run_date: 2026-07-28
agent: claude-code
phase: phase3
verified: no
tags: parity, emotion, motor-imagery, shuffle-control, label-efficiency, correction
commits:
verdict: TWO CORRECTIONS. (1) The emotion "null" was partly a PROTOCOL ARTIFACT — the old −3.2 gain came from combined-corpus pretraining while every other task was pretrained on its own data; under matched single-dataset pretraining (3 seeds) smoothed emotion is +2.38, a weak POSITIVE. The 4-task picture is therefore a GRADED SPECTRUM (sleep +14.5, emotion-unsmoothed +11.0, seizure +8.1, emotion-smoothed +2.4, MI −1.8), not a clean 2×2. (2) The order-shuffle control is CONFOUNDED on trial-constant-label tasks: on un-smoothed emotion shuffling IMPROVED PC by +7.4 (53.5→60.8), because with one label per trial, shuffling lets each causal position sample the whole trial (better pooling). The control is only interpretable for per-epoch-label tasks (sleep, seizure). Label-efficiency replicates on emotion (PC > raw-DE at 1% labels on both variants).
---

# EXP-0016 — Protocol parity for emotion + motor imagery

> **Status:** done · **Run:** 2026-07-28 · **Agent:** claude-code · **Phase:** phase3

---

## 1. Why — hypothesis & motivation

The headline claim is a comparison **across** four tasks, so all four must be evaluated
with the **same** analyses. Sleep ([[EXP-0009]]) and seizure ([[EXP-0015]]) had the modern
suite — batched frozen-encoder extraction, multi-seed, label-efficiency curves, an
order-shuffle control, per-fold outputs for paired tests. Emotion's results predate that
harness (and used a *different pretraining corpus*), and motor imagery ([[EXP-0014]]) had
a single run. This closes the gap so the cross-task comparison is apples-to-apples.

## 2. Setup

New tooling: `scripts/phase2_emotion_parity.py` (emotion under the frozen subject-dependent
harness + batched extraction + `--shuffle_time` + `--label_fracs` + parallel fits + per-fold
CSV); `--shuffle_time`/`--label_fracs`/per-subject CSV added to `scripts/phase2_bci_eval.py`;
`scripts/compute_temporal_structure.py` (within-sequence variance fraction + k-step
predictability gap); driver `scripts/run_parity.sh`.

**Validation:** the new emotion path reproduces the canonical raw-DE ceiling exactly
(SEED-IV 62.75 ± 20.28), confirming it is the same frozen harness, not a new pipeline.

Emotion: SEED-IV, both feature variants on *identical* trials/labels/folds — smoothed
(`de_LDS`, the public benchmark) and un-smoothed (`de_movingAve`); PC vs matched random-init,
3 seeds, `p_in=1 p_out=16`. MI: existing F16 models (`p_out=8`).

## 3. Results *(run 2026-07-28)*

### 3a. Emotion headline — matched single-dataset pretraining, 3-seed means

| Variant | physiofm_pc | physiofm_rand | raw_de | pc − rand |
| --- | ---: | ---: | ---: | ---: |
| smoothed (`de_LDS`) | 59.26 | 56.88 | 62.75 | **+2.38** |
| un-smoothed (`de_movingAve`) | 51.70 | 40.70 | 55.34 | **+11.01** |

### 3b. The corrected cross-task picture (accuracy points, PC − random-init)

| Task | median seq. len | gain |
| --- | ---: | ---: |
| Sleep | 1127 | **+14.5** (3 seeds) |
| Emotion, un-smoothed | 36 | **+11.0** (3 seeds) |
| Seizure | 1800 | **+8.1** (1 seed, bal-acc) |
| Emotion, smoothed | 36 | **+2.4** (3 seeds) |
| Motor imagery | 13 | **−1.8** (3 seeds) |

### 3c. Label efficiency (replicates the sleep/seizure pattern on emotion)

At 1% labels: smoothed emotion pc 48.18 > raw 46.22 > rand 42.03; un-smoothed emotion
pc 38.92 > raw 31.36 ≈ rand 31.34 (**+7.6 over raw-DE**). MI shows a small low-label
benefit (pc 30.65 vs rand 28.14 at 1%) that inverts at full labels (42.36 vs 43.63).

### 3d. Order-shuffle control — and its confound

| Task (label granularity) | PC normal | PC shuffled | Δ |
| --- | ---: | ---: | ---: |
| Sleep (per-epoch) | 72.6 | 67.4 | **−5.2** (gain destroyed) |
| Emotion smoothed (trial-constant) | 59.99 | 59.84 | −0.15 (no effect, as predicted) |
| Emotion un-smoothed (trial-constant) | 53.48 | **60.84** | **+7.36 (IMPROVED)** |
| MI (trial-constant) | 42.36 | 40.78 | −1.58 |

raw-DE was unchanged under shuffling on every task (it is per-window and order-blind) —
the manipulation's sanity check passes.

### 3e. Temporal-structure metrics (`results/phase3/temporal_structure.csv`)

within-sequence variance fraction / k-step predictability gap τ: sleep 0.60 / 0.11 ·
seizure 0.44 / 0.18 · **MI 0.29 / 0.27** · emotion-unsmoothed 0.18 / 0.26 ·
emotion-smoothed 0.0008 / 0.03.

## 4. Interpretation — agent's reading

**Correction 1 — the emotion null was partly a protocol artifact.** The previously reported
−3.2 gain used models pretrained on the **combined** SEED-IV+V+SEED corpus and evaluated on
SEED-IV, whereas sleep/seizure/MI were each pretrained on their own data. Matching the
protocol flips it to **+2.38**. Smoothed emotion is a *weak positive*, not a negative. The
ordering survives, but **"a clean 2×2" is no longer accurate — it is a graded spectrum** and
should be described that way.

**Correction 2 — the order-shuffle control does not mean the same thing on every task.**
It is a valid mechanism test only where labels are **per-epoch** (sleep, seizure): there,
scrambling order removes the context a window needs to classify *itself*, and the gain
collapses. Where labels are **trial-constant** (emotion, MI), every window shares one label,
so shuffling merely lets each causal position aggregate a random sample of the whole trial —
which acts as *denoising* and can **improve** accuracy (+7.4 on un-smoothed emotion). The
sleep/seizure shuffle evidence stands; the emotion/MI shuffle must not be read the same way.

**The simple dose-response does not hold.** Neither metric in §3e predicts the gain: MI has
the *highest* predictability τ (0.27; 0.42 with window overlap removed — so it is not an
overlap artifact) yet shows **no** gain, while sleep has the *lowest* τ of the positives and
the largest gain. Plausible reading: τ as defined measures **linear** predictability (ridge
vs persistence), and structure a linear model already captures is exactly the structure a
frozen linear probe can read off *without* pretraining. The quantity that should matter is
the **nonlinear-minus-linear** predictability gap — untested. This is a negative result for
the naive "data-only score predicts SSL uplift" idea and materially raises the risk of that
framing (relevant to the ICML plan in `docs/ICML_proposal.md`).

## 5. ✅ Your verification — *(reserved for Mahdiar)*

- [ ] **Verified**
- **Notes / corrections:**

## 6. Commits

- (this session) new: `scripts/phase2_emotion_parity.py`, `scripts/compute_temporal_structure.py`,
  `scripts/run_parity.sh`; edited: `scripts/phase2_bci_eval.py`, both figure scripts.

## 7. Links

- Corrects the emotion numbers in [[EXP-0001]] / `docs/PHASE2.md`; parity target set by
  [[EXP-0009]] (sleep) and [[EXP-0015]] (seizure); MI baseline from [[EXP-0014]].
