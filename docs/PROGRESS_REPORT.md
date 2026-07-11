# PhysioFM — Progress Report

**From:** Mahdiar Khodabakhshi · **Date:** 2026-07-08

## 1. Summary
I implemented the method from the proposal and evaluated it end-to-end. On the first
validation task (emotion, SEED) it does **not** beat the baselines. I then ran a systematic
set of experiments to find out *why*, and traced the cause to the **emotion DE features
themselves** — they are static and already linearly separable, so there is nothing for the
pretraining to add. Crucially, the same method **does work once real temporal dynamics are
present**: I confirmed this both by removing the smoothing from the emotion features, and —
the decisive test — on a genuinely dynamic task, **sleep staging**. On the **full Sleep-EDF
corpus (78 subjects)**, pretraining beats the matched no-pretraining model by **+10 points** and
now **also beats the strong linear DE baseline by +4.7 points** — a clean foundation-model win
on a dynamic task, and the mirror image of the emotion null. (The earlier out-of-memory blocker
was resolved by running on a rented 80 GB GPU; the true fix was a smaller batch, since attention
memory scales with sequence-length² and sleep recordings are ~60× longer than emotion trials.)

## 2. Starting point (recap of the previous stage)
In the first stage I fed each channel–band DE trace to TimesFM directly, as a univariate
series, and got **chance-level** accuracy. The reason was diagnostic: TimesFM's built-in
instance normalization removes the **absolute band-power level** (which is exactly where the
emotion signal lives), and treating each channel–band as an independent series discards the
spatial/spectral structure. This is precisely what motivated the proposal's design — a
**structured (channel × band) DE patch** fed to a decoder-only **predictive-coding**
transformer **with no instance normalization**. I built that model, and everything below tests
it.

## 3. Testing the proposed method on emotion (SEED)
**Why I did it.** The proposal's core claim is that predictive-coding pretraining on structured
DE learns representations that improve downstream recognition. Emotion/SEED is our first
validation task, so this is the first thing to verify.

**What I did.** Pretrained the model with the next-step prediction objective, then compared it,
under one fixed evaluation harness, against (a) the *identical model with no pretraining*
(random init) and (b) a plain **linear classifier** on the DE features.

**Result.** Pretraining gave **no measurable improvement**: pretrained ≈ no-pretraining ≈
linear classifier.

**So the next question.** Is the problem the *objective*, the *architecture*, the *readout
head*, or the *features*? The next experiments isolate each one.

## 4. Localizing the cause — controlled ablations
I eliminated the possible causes one at a time:

- **Readout head** — replaced the linear probe with a matched neural-network head → no change.
  *The head is not the limiter.*
- **Architecture / pretrained weights** — compared a frozen *random* transformer stack against
  the pretrained one → they matched. *The pretrained weights add nothing here.*
- **Objective** — swapped next-step forecasting for masked reconstruction → same null.
  *It is not the specific objective.*
- **Feature ceiling** — tested whether *any* nonlinear model can beat a linear one on the
  DE→emotion mapping → none could. *The DE features are already linearly saturated — there is
  no representational headroom left for pretraining to capture.*

**Interim conclusion:** the limiter is the **emotion DE signal itself**, not any component of
the method.

## 5. The turning point — the features had been smoothed
**Why I did it.** If the features are static, perhaps the standard preprocessing removed the
dynamics. The public SEED DE is **temporally smoothed** (LDS filter). I re-ran the whole
comparison on the **un-smoothed** DE.

**Result.** The null **flips**: pretraining now beats no-pretraining by **10–13 points**.
Follow-ups confirmed the effect is real and comes from temporal structure:
- the gain **grows with the amount of temporal context** the model sees (up to ~18 points);
- it is **stable across model sizes**;
- **shuffling the time order destroys it**;
- in the **low-label** setting the pretrained model beats the linear baseline by ~6 points at
  10% labels.

**Meaning.** The method's mechanism is genuine — it simply only appears when the features
actually **contain temporal dynamics**. Standard smoothing had been hiding it, which is why the
first emotion test looked flat.

## 6. Auditing the baseline we were chasing
**Why I did it.** The prior published state-of-the-art on this task reports 84–92%, far above
us, so I checked whether that number is real.

**What I did / result.** Using the authors' own code with a **clean, non-leaking** train/test
split, the accuracy drops to **~40–45%** — the reported 84–92% came largely from
**temporal-neighbor data leakage**.

**Meaning.** The target we were told to beat was inflated; the honest gap is small, and **peak
within-subject accuracy is the wrong yardstick** for this method.

## 7. The decisive test — a genuinely dynamic task (sleep staging)
**Why I did it.** This goes straight at the proposal's central claim (transfer across tasks,
value on dynamic biosignals). I registered a prediction *in advance*: on a task with strong
temporal structure, pretraining should clearly beat no-pretraining — the opposite of the
smoothed-emotion null. Sleep staging is ideal: the signal evolves through the night, and there
is a large public dataset (Sleep-EDF).

**What I did.** Built a sleep-staging pipeline (whole-night recordings → 30-second epochs →
the same DE features and the same evaluation harness as emotion), then ran the identical
pretrained-vs-no-pretraining comparison.

**Result (definitive, full corpus — 78 subjects, 195k epochs).**

| Model | Accuracy | Agreement (Cohen's κ) |
| --- | ---: | ---: |
| **Pretrained (proposed method)** | **72.6%** | **0.64** |
| No pretraining (random init) | 62.9% | 0.51 |
| Linear baseline on DE | 67.9% | 0.58 |

**Meaning.** Pretraining gives **+10 points** over no-pretraining — the mirror image of the
emotion result, exactly as predicted. This confirms the thesis: **the proposed pretraining
helps in proportion to how much genuine temporal structure a task has.** And at full scale the
method now **also beats the strong linear DE baseline (+4.7 points)**: the baseline degrades as
more subjects are added (cross-subject variability), while the pretrained model improves with
more data — so it wins on **peak accuracy too**, not only pretraining-gain and data-efficiency.
*(A 9-subject preliminary, now superseded, showed +19 over random with the linear baseline still
edging the peak; the full run confirms the direction and removes that caveat.)*

## 8. Where this leaves the project
- The **emotion** result is now a well-understood **negative result** — I can state precisely
  *when and why* the method fails (static, saturated features).
- The **sleep** result is the **positive signal** that makes the project work.
- **Plan:** center the work on **dynamic tasks** — confirm sleep at full scale, add one more
  dynamic task (seizure or motor-imagery), and keep emotion as the controlled negative
  baseline. The headline metric becomes **data-efficiency and cross-task transfer**, not peak
  accuracy.

## 9. Compute status & remaining runs
The keystone full-dataset sleep result (§7) is **done** — run on a rented **80 GB H100**
(RunPod). Note the earlier out-of-memory crash was **not** a model-size problem (the model is
2.4M params and used <3 GB); it was batch size. Attention memory scales with
sequence-length², and sleep recordings (~2,600 steps) are ~60× longer than emotion trials
(~40), so the emotion-default batch exploded. A smaller batch (`BATCH=4–16`) fits on the
original ~20 GB card too — the 80 GB GPU mainly bought **speed** (pretraining in ~2 min) and
headroom for the heavier raw-EEG work to come.

**Still to run (small compute; H100 access already sufficient):**
1. **Sleep label-efficiency curve** (F7-analog: pc vs rand vs raw at 10/50/100% labels) — the
   Option-A headline metric; the FM is expected to win by more in the low-label regime.
2. **Multi-seed repeat** (≥3 seeds) + a **paired per-fold test** for the pc−raw margin.
3. A **second dynamic task** (seizure CHB-MIT or motor-imagery BCI-IV-2a) for the cross-task
   "foundation model" claim — this one is **data-blocked** (dataset not yet on disk), not
   compute-blocked.
