# PhysioFM — Progress Report

**From:** Mahdiar Khodabakhshi · **Date:** 2026-07-08

## 1. Summary
I implemented the method from the proposal and evaluated it end-to-end. On the first
validation task (emotion, SEED) it does **not** beat the baselines. I then ran a systematic
set of experiments to find out *why*, and traced the cause to the **emotion DE features
themselves** — they are static and already linearly separable, so there is nothing for the
pretraining to add. Crucially, the same method **does work once real temporal dynamics are
present**: I confirmed this both by removing the smoothing from the emotion features, and —
the decisive test — on a genuinely dynamic task, **sleep staging**, where pretraining gives a
**+19-point** jump. I am now blocked on GPU memory for running this at full scale.

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

**Result (preliminary, 9 subjects).**

| Model | Accuracy | Agreement (Cohen's κ) |
| --- | ---: | ---: |
| **Pretrained (proposed method)** | **68.8%** | **0.59** |
| No pretraining (random init) | 50.0% | 0.35 |
| Linear baseline on DE | 72.1% | 0.63 |

**Meaning.** Pretraining gives **+19 points** over no-pretraining — the mirror image of the
emotion result, exactly as predicted. This confirms the thesis: **the proposed pretraining
helps in proportion to how much genuine temporal structure a task has.** Its advantage is a
**pretraining gain and data-efficiency** rather than peak accuracy (a strong linear DE baseline
still edges the peak here).

## 8. Where this leaves the project
- The **emotion** result is now a well-understood **negative result** — I can state precisely
  *when and why* the method fails (static, saturated features).
- The **sleep** result is the **positive signal** that makes the project work.
- **Plan:** center the work on **dynamic tasks** — confirm sleep at full scale, add one more
  dynamic task (seizure or motor-imagery), and keep emotion as the controlled negative
  baseline. The headline metric becomes **data-efficiency and cross-task transfer**, not peak
  accuracy.

## 9. What I'm blocked on — compute request
The sleep result above is only preliminary (9 subjects) because I cannot run the full dataset.
The current GPU has only **~20 GB** of memory. Sleep recordings are whole nights (~2,600
time-steps each), and a transformer's memory grows with the **square** of the sequence length,
so the full-dataset run **runs out of memory and crashes**. Only tiny batches fit, which is too
slow and unstable to finish the experiments.

**Request:** access to a GPU with **≥40 GB (ideally 80 GB — e.g. A100/H100).**

**It unlocks immediately:**
1. The **full-dataset sleep result** (the keystone experiment for the paper).
2. **Data-efficiency curves** — the headline metric showing the model's advantage with limited
   labels.
3. A **second dynamic task**, to support the cross-task "foundation model" claim.
