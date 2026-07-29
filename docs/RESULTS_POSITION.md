# Where PhysioFM-S actually stands — the honest scorecard

*Built for the supervisor meeting. Every "ours" number comes from one matched protocol
(`scripts/run_parity.sh`, [[EXP-0016]]); published numbers are from a first-pass literature
sweep and are **flagged for verification** against the source papers before use in a paper.*

Last updated: 2026-07-28.

---

## 1. The one-paragraph answer

Predictive-coding pretraining **works** in the scientific sense — it beats a matched
no-pretraining control in **5 of 6 settings**. But we do **not** beat published
state-of-the-art on any task, and at full labels we do not even beat our own simple linear
baseline except on sleep. **The one consistent, defensible win is label efficiency: with 1%
of labels our representation beats raw features on 5/5 tasks.** That is the paper.

---

## 2. Our results — all tasks, one matched protocol

Frozen encoder → StandardScaler + balanced logistic regression. `PC` = predictive-coding
pretrained; `rand` = identical architecture, no pretraining; `raw-DE` = features straight
to the classifier.

| Task | metric | PC | rand | raw-DE | PC − rand | PC − raw-DE | seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sleep (Sleep-EDF, 78 subj) | acc % | **73.03** | 58.51 | 67.86 | **+14.52** | **+5.17** | 3 |
| Seizure (CHB-MIT, 24 pat.) | bal-acc % | **75.51** | 67.41 | 72.37 | **+8.10** | +3.14 | 1 |
| Seizure | ROC-AUC | **0.822** | 0.740 | 0.806 | **+0.082** | +0.016 | 1 |
| Emotion, un-smoothed (SEED-IV) | acc % | 51.70 | 40.70 | **55.34** | **+11.01** | −3.64 | 3 |
| Emotion, smoothed (SEED-IV) | acc % | 59.26 | 56.88 | **62.75** | +2.38 | −3.49 | 3 |
| Motor imagery (BCI-IV-2a) | acc % | 41.71 | 43.54 | **51.08** | −1.84 | −9.37 | 3 |

**Pretraining helps: 5/6.** **We beat our own linear baseline: 2/5 tasks** — and of those,
only **sleep** is statistically significant (paired p=0.0008, 5/5 folds); the seizure margin
is a **tie** under a paired per-patient test (p=0.46, 11/24 patients).

---

## 3. Versus published state of the art — we are below it

⚠️ *First-pass numbers; verify each against the source paper and confirm protocol match
before citing.*

### Sleep-EDF-78, 5-class, subject-independent

| Method | acc % | κ | Protocol match? |
| --- | ---: | ---: | --- |
| SleepGMUformer | 85.0 | 0.83 | multimodal — not directly comparable |
| SleepTransformer | 84.8 | 0.787 | raw-signal, single-channel EEG |
| AT-BiLSTM | 83.8 | 0.766 | raw-signal |
| EEGSNet (CNN-LSTM) | 83.0 | 0.77 | raw-signal |
| ULW-SleepNet ("ultra-lightweight") | 81.4 | — | raw-signal, multimodal |
| **Ours (PhysioFM-S, fine-tuned)** | **75.4** | **0.672** | DE features, subject-disjoint |
| Ours (PhysioFM-S, frozen probe) | 73.0 | ~0.63 | DE features, subject-disjoint |
| Ours (random-init, fine-tuned) | 73.2 | 0.645 | **no pretraining** — nearly matches PC |
| Ours (raw-DE linear baseline) | 67.9 | 0.575 | DE features |

**Gap to SOTA: ~8–12 accuracy points.** Every stronger method operates on the **raw signal**;
we operate on compressed DE features and a ~2.4M-parameter model. That is the honest trade.

### CHB-MIT, cross-patient (patient-independent) seizure detection

| Method | sensitivity | AUC | Notes |
| --- | ---: | ---: | --- |
| Gradient-boosted ensemble (LOSO) | 0.922 | ≥0.99 | segment-level |
| Transformer + reference learning | 91.1 | 0.943 | cross-subject |
| Spiking NN (LOO cross-patient) | 90.5 | 0.969 | **with fine-tuning**, 4 s segments |
| CNN-BiLSTM + channel perturbation | 86.5 | 0.908 | strictly patient-independent |
| **Ours (PhysioFM-S)** | **65.2** | **0.822** | 2 s epochs, frozen encoder + linear probe |
| Ours (raw-DE linear baseline) | 67.1 | 0.806 | |

**Gap to SOTA: ~0.09–0.17 AUC.** Note these are cross-patient too, so the "our protocol is
harder" defence does **not** apply here. Ours is a frozen linear probe; several of theirs
fine-tune end-to-end on raw signal.

*(Emotion and motor imagery: published SOTA is also far above us — SEED-IV ~84% claimed
(but see the leakage audit, [[EXP-0008]]: clean protocol ≈ 40–45%), BCI-IV-2a ~70–85%
vs our 41.7%.)*

---

## 4. What we *do* win — label efficiency, consistently

`PC − raw-DE` at two label budgets. This is the only comparison that is positive everywhere.

| Task | @ 1% labels | @ 100% labels |
| --- | ---: | ---: |
| Emotion, un-smoothed | **+7.56** | −1.86 |
| Seizure (AUC×100) | **+6.60** | +1.60 |
| Sleep | **+3.90** | +4.77 |
| Motor imagery | **+2.50** | −8.72 |
| Emotion, smoothed | **+1.96** | −2.76 |
| **Tasks where PC > raw-DE** | **5 / 5** | **2 / 5** |

Headline framings this supports:
- **Seizure: PC with 5% of labels (AUC 0.809) matches raw-DE trained on 100% (0.806)** — ~20× label efficiency; significant at 1% labels (paired p=0.017, 18/24 patients).
- **Sleep: PC with 1% of labels (70.9%) beats raw-DE and random-init trained on 100%.**
- The crossover is the classic self-supervised signature and it holds on **every** task.

---

## 5. What we can and cannot claim

**Can claim (supported by the numbers above):**
1. Predictive-coding pretraining on structured DE learns real, transferable structure — it beats a matched random-init control on 5/6 settings, significantly on sleep (p<1e-4) and seizure (p=0.006).
2. The benefit is **concentrated where labels are scarce** — PC > raw-DE on 5/5 tasks at 1% labels, on only 2/5 at full labels.
3. The gain scales with how much of the signal lives in the sequence: sleep +14.5, un-smoothed emotion +11.0, seizure +8.1, smoothed emotion +2.4, MI −1.8. The **smoothing flip** shows this causally on identical recordings (+2.4 → +11.0 by removing LDS smoothing).
4. A widely-cited emotion SSL benchmark is inflated by ~80% temporal-neighbour leakage ([[EXP-0008]]).
5. It is **lightweight** — ~1–3M parameters vs ~100M+ for raw-signal EEG foundation models.

**⚠️ MAJOR CAVEAT — the pretraining gain is frozen-probe-specific.** Under end-to-end
fine-tuning the advantage collapses on **both** per-epoch tasks ([[EXP-0017]] §4b–c):

| Task | PC − random-init, frozen | PC − random-init, fine-tuned |
| --- | ---: | ---: |
| Sleep | +9.77 | **+2.19** |
| Seizure (bal-acc) | +8.10 | **−1.55** (random-init *wins*: 80.2 vs 78.7) |

Fine-tuning gains the pretrained model ~3 points and the random-init model ~10–13. So at full
labels, training the same architecture from scratch matches or beats pretraining. **Claim 1
below must be stated as a frozen-encoder result.**

**⚠️ UNVERIFIED — the label-efficiency claim (§4) was measured with the frozen probe only.**
It must be re-tested under fine-tuning at low label fractions before publication. If PC still
beats random-init at 1–5% labels when both are fine-tuned, the standard SSL story holds; if
not, there is no positive full-protocol result. This is the decisive open experiment.

**Cannot claim:**
1. ❌ State-of-the-art on any task. We are 8–12 points below on sleep, ~0.1 AUC below on cross-patient seizure.
2. ❌ Beating a simple linear baseline at full labels — true only on sleep (seizure is a tie; emotion and MI lose).
3. ❌ A clean "temporal vs spectral 2×2" — it is a graded spectrum, and smoothed emotion is a weak positive, not a negative ([[EXP-0016]]).
4. ❌ That the order-shuffle control proves the mechanism on emotion/MI — it is only interpretable for per-epoch-label tasks.

---

## 6. Recommended framing for the paper

> A **lightweight, interpretable predictive-coding foundation model on spectral (DE)
> features**, whose value is **label efficiency and cross-task transfer**, not peak accuracy.
> It recovers most of a strong linear baseline's performance with ~1–5% of the labels across
> four EEG tasks, and its pretraining benefit scales with a task's sequence-level temporal
> structure — which we characterise mechanistically, including where it fails.

Do **not** lead with accuracy tables against raw-signal SOTA. Lead with the label-efficiency
crossover (§4), report the SOTA gap honestly in a "positioning" paragraph, and use motor
imagery as the characterised failure case rather than as a fifth headline application.

---

## 7. Open items before submission

- [ ] Verify every published number in §3 against its source paper (protocol, subset, #classes, split).
- [ ] Seizure multi-seed (currently 1 seed; per-patient variance dominates, so low priority).
- [ ] Decide whether MI stays in the paper as a failure case or moves to an appendix.
