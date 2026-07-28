# PhysioFM — Results section (draft)

*Draft results prose for the paper. Numbers trace to the experiment log
(`docs/experiments/EXP-*`) and the figures in `results/figures/`. Framing per
`docs/STRATEGY.md`.*

---

## Setup (one paragraph)

Every model is a decoder-only, causal predictive-coding transformer over structured
`(channel × band)` differential-entropy (DE) patches ("PhysioFM-S"), pretrained by
predicting the next DE window(s) and evaluated frozen through a single fixed harness
(StandardScaler + balanced logistic regression) so every comparison uses the identical
readout. Across four EEG tasks we compare three feature sets under matched conditions:
the **PC-pretrained** encoder, a **matched random-init** encoder (identical architecture,
no pretraining — the pretraining control), and **raw DE** fed straight to the classifier
(the linear ceiling). Emotion uses subject-dependent folds; sleep and seizure use
subject/patient-disjoint folds; motor imagery uses the canonical session-holdout —
all leakage-free.

## 1. Structured patching recovers a signal univariate forecasting destroys

Feeding each channel–band DE trace to TimesFM as an independent univariate series is at
chance (20–28%; Phase 1), because per-series instance normalization removes the absolute
band-power level — the discriminative part of DE — and flattening erases channel–band
identity. Replacing the scalar patch with a structured `(C×B)` patch and removing instance
normalization recovers the emotion signal (chance → ~46–61% zero-shot; **C1 confirmed**).
A per-series-instance-norm control collapses even raw DE to below chance (SEED-V 18.0%,
SEED-IV 26.8%), confirming the mechanism.

## 2. On smoothed emotion DE, predictive-coding pretraining adds nothing — and why

On the public SEED DE, the PC-pretrained encoder ties matched random-init and both sit at
the raw-DE linear ceiling (~51% SEED-V / ~63% SEED-IV). Controlled ablations localize the
cause to the *features*, not the method: a matched neural-network readout head does not beat
the linear probe (F4); a frozen random transformer of TimesFM's exact shape matches the
frozen pretrained TimesFM stack (F3); swapping forecasting for masked reconstruction gives
the same null (F9); and no nonlinear classifier beats a linear one on the per-window DE→emotion
map (F10, linear saturation). The public SEED DE is LDS-smoothed, leaving only **0.1%** of the
per-(C,B) variance within-trial — the temporal dynamics a predictive objective needs are
absent. On **un-smoothed** DE (17.2% within-trial variance) the null flips: PC beats
random-init by **+10–13 points**, the gain grows with temporal context (up to +18) and model
scale is stable, time-order shuffling destroys it, and PC beats raw-DE by ~6 points at 10%
labels (F1, F2, F5, F6, F7). Emotion DE is *static spectral structure*; smoothing removed the
little temporal structure a time-series objective could use.

## 3. The benchmark we were chasing was mostly leakage

The published PC-SSL emotion SOTA (SEED-IV 84.5%, SEED-V 92.4%) rests on ~80% temporal-neighbor
leakage: the authors split individual DE windows at random, placing near-duplicate adjacent
windows in both train and test. Holding their implementation fixed and using a clean
trial-disjoint split collapses accuracy to 40–45% — at or below the raw-DE linear ceiling and
within the PhysioFM-S band (F12). The honest emotion target is far lower than reported.

## 4. On genuinely dynamic tasks, predictive-coding pretraining pays off

**Sleep staging (Sleep-EDF, 78 subjects, 195k epochs; subject-disjoint LOPO).** PC-pretrained
reaches 73.0 ± 0.4% accuracy over three seeds, beating matched random-init by **+9.8 points**
(paired p<10⁻⁴, 5/5 folds) and the raw-DE ceiling by +5.2 (paired p=0.0008, 5/5 folds). A
pre-registered **order-shuffle control** scrambles epoch order before encoding: PC collapses
from 72.6 to 67.4 — exactly the raw-DE level — so its *entire* advantage over the linear
ceiling is temporal (Figure 2A). The pretraining gain widens as labels shrink (+9.8 at 100%
labels → +12.7 at 1%), and PC with 1% of labels (70.9%) beats both baselines at 100% labels
(Figure 1A).

**Seizure detection (CHB-MIT, 24 patients, 1.76M epochs, 0.32% seizure; leave-one-patient-out,
imbalance-aware metrics).** PC-pretrained significantly beats matched random-init — AUC
0.822 vs 0.740, paired p=0.006 (17/24 patients), widening to +0.105 AUC at 1% labels (p=0.0002,
22/24). PC is nearly label-insensitive (AUC 0.822→0.797 from 100% to 1% labels) while raw-DE
degrades three times as fast; **PC with 5% of labels (AUC 0.809) matches raw-DE trained on 100%
(0.806)** — ≈20× label efficiency (Figure 1B). PC does *not* significantly beat raw-DE at full
labels (paired p=0.46, 11/24 — a tie), but does at 1% labels (p=0.017, 18/24) (Figure 2C).

## 5. The mechanism: temporal structure, graded

Evaluated under one matched protocol (same architecture, objective, harness, and
single-dataset pretraining throughout), the pretraining gain forms a **graded spectrum**
rather than a binary split (Figure 1C): **sleep +14.5**, **un-smoothed emotion +11.0**,
**seizure +8.1**, **smoothed emotion +2.4**, **motor imagery −1.3** accuracy points over
matched random-init. The gain tracks how much of the discriminative signal lives in the
*evolution* of the feature sequence rather than in any single window.

Two interventions support this causally. The **smoothing flip** (Figure 2B) varies temporal
structure while holding trials, labels, folds, and architecture fixed: LDS smoothing leaves
0.08% of the per-(channel,band) variance within-trial and the gain is +2.4; the un-smoothed
features of the *same recordings* carry 17.6% and the gain is +11.0. The **order-shuffle
control** (Figure 2A) scrambles window order before encoding: on sleep the PC advantage
collapses to the raw-DE level (72.6 → 67.4).

*Scope of the shuffle control.* It is interpretable only where labels are **per-epoch**
(sleep, seizure). Where labels are **trial-constant** (emotion, motor imagery) every window
shares one label, so shuffling lets each causal position aggregate a random sample of the
whole trial — an effective denoising that can *raise* accuracy (+7.4 on un-smoothed emotion).
We therefore restrict the shuffle argument to per-epoch-label tasks.

A caution for future work: simple data-only measures of temporal structure do **not**
predict the gain. Motor imagery has the highest k-step predictability of all five settings
(τ = 0.27) yet gains nothing, while sleep has the lowest τ among the positive tasks and gains
the most. Predictability that a *linear* model already captures appears not to be the kind
predictive-coding pretraining can convert into linear-probe-accessible information.

## 6. Positioning

PhysioFM-S is *not* a model that beats a strong linear baseline when labels are abundant (on
seizure it ties raw-DE at full labels). It is a lightweight, interpretable, non-contrastive
predictive-coding foundation model on DE features whose value is **cross-task transfer and label
efficiency** — winning specifically where labels are scarce and temporal structure exists, which
is where foundation models are supposed to help.

## Honest limitations

- Two positive tasks (sleep, seizure), single-seed for seizure (per-patient variance ±0.20 AUC
  dwarfs seed variance); emotion un-smoothed leg is SEED-IV-only (sole un-smoothed feature key).
- The clean PC-SSL absolute number comes from a faithful re-implementation; the controlled,
  implementation-invariant claim is the leaky-vs-clean delta on identical code.
- With large inter-patient variance, only paired per-subject tests are trustworthy — unpaired
  means overstated a seizure raw-DE "win" that did not survive pairing.
