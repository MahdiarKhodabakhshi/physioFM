---
id: EXP-0017
title: Why it fails — encoder diagnostic (dimension-matched control + concat test)
status: done
created: 2026-07-29
run_date: 2026-07-29
agent: claude-code
verified: no
phase: phase3
tags: diagnostic, root-cause, ablation, label-granularity, negative-result
commits:
verdict: ROOT CAUSE FOUND. Against a DIMENSION-MATCHED control (raw DE pushed through a random 256-d nonlinear projection), predictive-coding pretraining adds real information ONLY on sleep (+3.3); on emotion it is a wash (−0.2) and on motor imagery it is actively harmful (−9.3). Part of the sleep "win" over raw-DE is just dimensionality (random projection alone gives +1.4). The splitting variable is LABEL GRANULARITY, not temporal structure per se: where labels are PER-EPOCH (sleep, seizure) temporal context informs the label and the causal encoder helps; where labels are TRIAL-CONSTANT (emotion, MI) temporal ordering is irrelevant to the label, so a causal sequence model supplies the wrong inductive bias and destroys per-window spectral detail. Concat [raw‖PC] beats raw only on sleep (+5.4); on emotion it ties (−0.3) and on MI it hurts (−7.3). Actionable: the method is matched to per-epoch-label tasks; for trial-constant tasks a permutation-invariant/bidirectional readout is the right architecture, not a causal one.
---

# EXP-0017 — Why it fails: encoder diagnostic

> **Status:** done · **Run:** 2026-07-29 · **Agent:** claude-code

---

## 1. Why

We beat the raw-DE linear baseline at full labels on only sleep (seizure ties, emotion and
MI lose). Two very different explanations fit: **(H1)** the 256-d encoder is a *bottleneck*
that destroys high-dimensional per-window spectral detail (fixable — readout/architecture);
**(H2)** the encoder adds *no new information* over the current window (fatal for the
premise). Also suspicious: the margin over raw-DE decreases monotonically with raw-DE
dimensionality (sleep 10-d **+5.2**, seizure 90-d +3.1, MI 110-d −9.4, emotion 310-d −3.5),
which hints the "win" may partly be a **capacity** effect rather than a pretraining effect.

## 2. Setup

`scripts/diagnose_encoder.py`, frozen features + the same balanced-logreg harness. Four
feature sets per task:
- `raw_de` — features straight to the classifier.
- `physiofm_pc` — our pretrained encoder (256-d).
- `raw_randproj256` — **dimension-matched control**: raw DE through a *random* 256-d
  nonlinear projection (tanh). Isolates "what pretraining learned" from "more dimensions".
- `concat_raw+pc` — raw DE **and** encoder features together. If the encoder carries
  complementary temporal information, this must beat raw alone.

## 3. Results *(2026-07-29, `results/phase3/diagnose_encoder.csv`)*

| Task | raw-DE | rand-proj 256 | PC | **PC − rand-proj** | concat − raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sleep (per-epoch labels) | 67.86 | 69.30 | 72.63 | **+3.33** | **+5.39** |
| Emotion (trial-constant) | 62.75 | 60.17 | 59.99 | **−0.18** | −0.30 |
| Motor imagery (trial-constant) | 51.08 | 49.92 | 40.59 | **−9.33** | −7.25 |

Supporting: `physiofm_rand` (random-weight *transformer*) scores 62.86 / 55.93 / 44.17 on
sleep / emotion / MI — i.e. **below the simple random projection** on emotion (60.17) and MI
(49.92). The causal transformer architecture *itself* costs accuracy before pretraining
recovers any.

## 4. Interpretation

**(a) Part of the sleep win is capacity, not pretraining.** A random projection of raw DE
already gains +1.4 over raw. Against that fair control the true pretraining contribution on
sleep is **+3.3**, not +5.2. Still real, but smaller than headlined.

**(b) On emotion, pretraining buys nothing.** PC (59.99) ≈ random projection (60.17).
Everything our pretrained encoder delivers on emotion is reproducible by a *random* 256-d
map of the same features. Concatenation confirms it: [raw‖PC] (62.45) ≈ raw (62.75), so the
encoder holds **no information complementary to the current window**. H2 holds here.

**(c) On MI it is actively harmful** (−9.3 vs the control), and concat *hurts* too.

**(d) The splitting variable is LABEL GRANULARITY.** Sleep and seizure have **per-epoch**
labels — the class of *this* window depends on its temporal context (sleep architecture,
seizure onset/evolution), so a causal encoder that summarises the past is exactly the right
inductive bias. Emotion and MI have **trial-constant** labels — every window in a trial
shares one class, so *when* something occurred carries no information about the label. What
those tasks need is a good permutation-invariant summary of the trial's spectral content; a
causal transformer instead supplies a running one-sided summary and smears the clean
per-window spectral detail the linear probe needs. This also explains the shuffle-control
confound in [[EXP-0016]] (shuffling *helps* on trial-constant tasks because it turns each
causal window into a random sample of the whole trial — i.e. better pooling).

**(e) Consequence for the thesis.** The honest rule is **not** "PC helps in proportion to
temporal structure in the features" — MI has the highest measured predictability and gains
nothing. It is closer to: **PC helps when the LABEL depends on temporal position.**
Sequence length is confounded with label granularity in our four tasks (sleep 1127 /
seizure 1800 per-epoch vs emotion 36 / MI 13 trial-constant), so we cannot fully separate
them with current data.

## 5. What this implies (actionable)

1. **The method is matched to per-epoch-label tasks.** Sleep and seizure are the honest
   application domain; emotion and MI are architecture/task mismatches, not tuning failures.
2. **More seeds will not help** — per-seed spread is ~1–2 pts, these gaps are 3–9 pts and
   consistent in sign.
3. **A residual/concat path helps only where the encoder has complementary info** (sleep
   +5.4 over raw); it cannot rescue emotion/MI.
4. **Untested and worth testing:** a **bidirectional / permutation-invariant** encoder for
   trial-constant tasks (the causal constraint is the wrong bias there), and **fine-tuning**
   instead of a frozen probe on the data-rich tasks (all SOTA baselines fine-tune).
5. The **DE bottleneck** (~600× compression before the model sees anything) remains the most
   likely cause of the residual gap to raw-signal SOTA — that is [[EXP-0013]] / F15.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

- [ ] **Verified**
- **Notes / corrections:**

## 7. Links

- Follows [[EXP-0016]] (parity + the shuffle confound this explains); positioning in
  `docs/RESULTS_POSITION.md`.
