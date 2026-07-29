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
verdict: ROOT CAUSE FOUND. Against a DIMENSION-MATCHED control (raw DE pushed through a random 256-d nonlinear projection), predictive-coding pretraining adds real information ONLY on sleep (+3.3); on emotion it is a wash (−0.2) and on motor imagery it is actively harmful (−9.3). Part of the sleep "win" over raw-DE is just dimensionality (random projection alone gives +1.4). The splitting variable is LABEL GRANULARITY, not temporal structure per se: where labels are PER-EPOCH (sleep, seizure) temporal context informs the label and the causal encoder helps; where labels are TRIAL-CONSTANT (emotion, MI) temporal ordering is irrelevant to the label, so a causal sequence model supplies the wrong inductive bias and destroys per-window spectral detail. Concat [raw‖PC] beats raw only on sleep (+5.4); on emotion it ties (−0.3) and on MI it hurts (−7.3). SEPARATELY AND MORE SERIOUSLY, end-to-end fine-tuning collapses the pretraining advantage on BOTH per-epoch tasks: sleep +9.8->+2.2, seizure +8.1->-1.6 (random-init fine-tuned BEATS pretrained, 80.2 vs 78.7 bal-acc). At full labels the pretraining benefit is an artifact of FREEZING the encoder. The decisive follow-up is done (4d): under fine-tuning PC keeps a REAL but modest advantage at every label budget on sleep (+1.9 to +2.4), so this is a positive result — but the label-efficiency SIGNATURE (gap widening as labels shrink: +9.8->+12.7 frozen) does NOT survive; fine-tuned the gap is FLAT. The 'advantage grows when labels are scarce' claim must be dropped. ROOT CAUSE (4e): the pretext IS learned (model beats persistence and ridge on 4/5 datasets, cutting MSE 40-65%) but pretext skill is ANTI-CORRELATED with downstream gain (best forecaster = MI = worst transfer; worst forecaster = sleep = best transfer). The failure is OBJECTIVE MISALIGNMENT: what makes DE predictable (smooth autocorrelation) is not what makes it discriminative. PC pretraining helps only when what is predictable is also what is discriminative — a property of the task, unfixable by scaling or tuning. Actionable: the method is matched to per-epoch-label tasks; for trial-constant tasks a permutation-invariant/bidirectional readout is the right architecture, not a causal one.
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

## 4b. Follow-up: was the FROZEN PROBE the problem? *(run 2026-07-29)*

`scripts/phase2_sleep_finetune.py`, same subject-disjoint folds, end-to-end fine-tuning
(`--mode full`, 8 epochs, class-weighted CE, nights chunked to 400 epochs).

| Arm | frozen probe | fine-tuned | Δ |
| --- | ---: | ---: | ---: |
| physiofm_pc | 72.63 | **75.37** (κ 0.672) | **+2.74** |
| physiofm_rand | 62.86 | **73.18** (κ 0.645) | **+10.32** |
| raw_de (linear) | 67.86 | n/a | |

**PC advantage over random-init: +9.77 frozen → +2.19 fine-tuned.**
**Gap to published SOTA (81–85%): −10.4 frozen → −7.6 fine-tuned.**

Two consequences, both important:

1. **Fine-tuning recovers only ~2.7 of the ~10-point SOTA gap.** So the frozen probe was
   costing us something real but is *not* the main limiter. The residual ~7.6 points is most
   plausibly the **DE feature bottleneck** (~600× compression before the model sees anything),
   which only the raw-EEG leg ([[EXP-0013]]/F15) can address.
2. **Most of the apparent "pretraining benefit" is a frozen-probe phenomenon.** Random-init
   gains +10.3 from fine-tuning and nearly catches PC (73.18 vs 75.37). Bluntly: *training the
   same architecture from scratch end-to-end (73.2) beats pretraining-then-freezing (72.6).*
   With abundant labels and full fine-tuning, pretraining is worth only ~2 points.

This does not invalidate the pretraining result — it **scopes** it. Pretraining matters when
you cannot fine-tune, or do not have the labels to; that is exactly the label-efficiency
regime where we already showed PC > raw-DE on 5/5 tasks. It does mean the headline
"+9.8 from pretraining" must be reported as *frozen-encoder* and paired with the fine-tuned
number, or a reviewer will (correctly) call it an artifact of the evaluation protocol.

## 4c. Replication on seizure — the collapse is general *(run 2026-07-29)*

`scripts/phase2_chbmit_finetune.py`, same LOPO protocol, 4 epochs, class-weighted CE with
3:1 interictal-chunk subsampling (positives are 0.3% of epochs).

| Task | arm | frozen | fine-tuned | Δ |
| --- | --- | ---: | ---: | ---: |
| Sleep (acc %) | physiofm_pc | 72.63 | 75.37 | +2.74 |
| Sleep | physiofm_rand | 62.86 | 73.18 | **+10.32** |
| Seizure (bal-acc %) | physiofm_pc | 75.51 | 78.66 (AUC 0.867) | +3.15 |
| Seizure | physiofm_rand | 67.41 | **80.21** (AUC 0.874) | **+12.80** |

**Pretraining advantage: sleep +9.77 → +2.19; seizure +8.10 → −1.55 (REVERSES).**

The finding replicates and strengthens: on seizure, the matched **random-init model
fine-tuned end-to-end BEATS the pretrained one** (80.21 vs 78.66 bal-acc; AUC 0.874 vs
0.867). Across both per-epoch-label tasks, fine-tuning gains the pretrained model ~3 points
and the random-init model ~10–13 points.

**Conclusion: at full labels, the pretraining benefit is an artifact of freezing the
encoder.** Given enough labels and end-to-end training, the same architecture trained from
scratch matches or beats predictive-coding pretraining on both of our positive tasks.

**Critical open question this creates.** *Every* label-efficiency result we have
(`RESULTS_POSITION.md` §4, PC > raw-DE on 5/5 tasks at 1% labels) was computed with the
**frozen probe**. If the pretraining advantage is frozen-probe-specific, the label-efficiency
claim must be re-tested **under fine-tuning at low label fractions** before it can be
published. If PC still beats random-init at 1–5% labels when both are fine-tuned, the
standard SSL story holds and is the paper. If it does not, the project has no positive
full-protocol result. **This is now the single most important experiment.**

## 4d. THE DECISIVE TEST — does label efficiency survive fine-tuning? *(run 2026-07-29)*

Sleep, both arms fine-tuned end-to-end, stratified masking of TRAINING labels (sequences
intact, so the encoder keeps full temporal context — the setting most favourable to
pretraining). `results/phase3/f13/f13_sleep_ft_labelcurve.csv`.

| labels | FROZEN PC | rand | gap | FINE-TUNED PC | rand | gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 70.94 | 58.20 | **+12.74** | 72.12 | 70.22 | **+1.90** |
| 5% | 72.51 | 60.75 | +11.76 | 73.32 | 70.90 | +2.42 |
| 10% | 72.64 | 61.89 | +10.75 | 73.85 | 71.41 | +2.44 |
| 100% | 72.63 | 62.86 | +9.77 | 75.37 | 73.18 | +2.19 |

**Answer: partially.** Pretraining does keep a **real, consistent advantage at every label
budget** (+1.9 to +2.4) — so this is a genuine positive result, not a null. **But the
label-efficiency SIGNATURE does not survive.** Frozen, the gap *widens* as labels shrink
(+9.8 → +12.7), which is the classic SSL story we were going to headline. Fine-tuned, the
gap is **flat** (+2.2 → +1.9): pretraining helps by a fixed ~2 points regardless of how many
labels you have.

**So the claim "the advantage grows as labels get scarce" is a frozen-probe artifact and must
be dropped.** What survives is: *predictive-coding pretraining is worth a consistent ~2
accuracy points on sleep under a full fine-tuning protocol.*

Caveat on scale: 1% of sleep's 195k epochs is still ~2,000 labelled epochs, so "1% labels"
is not an extreme low-data regime here in absolute terms. A truly small-label test would need
a task with far fewer labels overall.

**Still owed:** the same low-label fine-tuning sweep on **seizure**, where the full-label
fine-tuned gap was *negative* (−1.55). If seizure also shows no low-label advantage, the
surviving positive result is sleep-only and ~2 points.

## 4e. ROOT CAUSE — the pretext is learned, but it is MISALIGNED *(run 2026-07-29)*

`scripts/diagnose_pretext.py`. Does the model succeed at its own objective — predicting the
next `p_out` DE windows — against persistence (zero-parameter copy) and a ridge linear map,
under the identical masked-MSE metric?

| Dataset | persistence | ridge | **model** | model/persist | model/ridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| sleep_edf | 3.885 | 3.355 | **2.336** | **0.60** | 0.70 |
| chbmit | 36.05 | 30.21 | **20.66** | **0.57** | 0.68 |
| seed_iv_raw | 120.31 | 205.76 | **52.72** | **0.44** | 0.26 |
| bci_iv_2a | 62.02 | 45.91 | **21.64** | **0.35** | 0.47 |
| seed_iv (smoothed) | **0.237** | 9.999 | 6.569 | 27.76 | 0.66 |

**The pretraining works.** On 4 of 5 datasets the model beats both a zero-parameter copy
baseline and a linear predictor, cutting persistence MSE by 40–65%. It is genuinely learning
predictive temporal structure. (The exception is smoothed emotion, where the features are so
autocorrelated that persistence is near-optimal at 0.237 and nothing beats it — the
degenerate-pretext case F1 already identified.)

**But pretext skill is ANTI-CORRELATED with downstream benefit:**

| Task | pretext skill (model/persist, lower=better) | downstream gain |
| --- | ---: | ---: |
| Motor imagery | **0.35 (best forecaster)** | **−1.8 (worst)** |
| Emotion un-smoothed | 0.44 | +11.0 |
| Seizure | 0.57 | +8.1 frozen / −1.6 fine-tuned |
| Sleep | **0.60 (worst of the four)** | **+14.5 frozen / +2.2 (best)** |

The model that forecasts *best* transfers *worst*, and vice versa.

**This is the root cause: OBJECTIVE MISALIGNMENT.** The failure is not that pretraining fails
(it does not), nor that there is no temporal structure (there is, and it is learned). It is
that **the structure that makes DE predictable is not the structure that makes it
class-discriminative.** Forecasting rewards modelling the smooth, autocorrelated, slowly
varying component of DE — precisely the part that carries little class information. On MI,
DE is highly autocorrelated (easy to forecast) but that autocorrelation says nothing about
which limb was imagined, so a model that excels at the pretext learns nothing useful. On
sleep, what is predictable (sleep architecture — stage sequences and transitions) *is* the
label, so pretext skill converts into downstream skill.

**The rule:** *predictive-coding pretraining helps only when what is predictable is also what
is discriminative.* That is a property of the task, not of the model, and it cannot be fixed
by scaling, more seeds, or better tuning.

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
