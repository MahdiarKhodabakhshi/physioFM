# PhysioFM — Final Report

**Predictive-coding pretraining on EEG spectral features: what we built, what we found, and why it does not work**

Mahdiar Khodabakhshi · 2026-07-29

---

## TL;DR

We built the proposed model and evaluated it on **5 EEG tasks / 7 settings** under one fixed protocol.

- ✅ The **architecture** works — structured `(channel × band)` patching + a causal transformer beats differential-entropy (DE) linear baselines on several tasks.
- ❌ The **pretraining** — the actual contribution — does not. Its apparent benefit is largely an artefact of freezing the encoder, and it disappears or reverses under fair evaluation.
- 🔬 We diagnosed *why*, proposed a mechanism, made a falsifiable prediction from it — and **the prediction failed**, which we report rather than hide.

**One-line verdict:** *a randomly-initialised structured transformer is already a strong EEG encoder; predictive-coding pretraining on DE adds ~2 points on sleep and nothing measurable anywhere else.*

---

## 0. Naming key (read this before the tables)

Every row below is **our own model**; only the weight initialisation differs.

| Label | Meaning |
|---|---|
| **PhysioFM (pretrained)** | our model **with** predictive-coding pretraining — the full proposed pipeline (`physiofm_pc` in the code) |
| **PhysioFM (no pretrain)** | the **identical architecture**, randomly initialised — the control that isolates what pretraining contributes (`physiofm_rand`) |
| **raw-DE (no model)** | no transformer at all: DE features straight into logistic regression — the baseline both must beat |

⚠️ Do not confuse **PC-SSL** (the *prior work* we build on) with **PC** in our code, which abbreviates
*predictive coding* and refers to **our own pretrained model**.

---

## 1. The idea

Combine two published results:

```
   PC-SSL (ICASSP'26)                  TimesFM (ICML'24)
   predictive coding on                decoder-only transformer,
   differential-entropy (DE)           pretrained on time series
            \                                    /
             \                                  /
              +------------  PhysioFM  --------+
        predict the next DE window with a decoder-only
        causal transformer  →  transferable EEG representations
```

**The bet:** learning to *predict* EEG's future teaches the model what EEG *means*, giving representations that transfer across tasks.

### The model (PhysioFM-S)

```
 raw EEG ──► differential entropy ──► token = (channels × bands) vector
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │  causal decoder transformer  │   ← TimesFM-2.5 layers
                              │  (d=256, 6 layers, RoPE)     │      ~1–3M params
                              └──────────────────────────────┘
                                             │
                   ┌─────────────────────────┴────────────────────────┐
                   ▼                                                  ▼
       PRETRAIN: predict next 16 DE windows              DOWNSTREAM: freeze,
       (MSE, no labels)                                  classify with logistic regression
```

Key design choices, all deliberate: **structured patch** (keep channel×band together), **no instance-norm** (it destroys absolute band power), **frozen linear probe** for evaluation.

---

## 2. Step-by-step: what we did and what happened

### Step 1 — Phase 1: univariate TimesFM → **at chance**

Fed each channel–band DE trace to TimesFM as its own series.

| Result | 20–28% (chance) |
|---|---|

**Cause:** TimesFM's instance normalisation strips absolute band power (where the signal is), and treating channels independently destroys spatial structure.
**→ Therefore:** redesign the input.

### Step 2 — Phase 2: structured patching → **signal recovered**

Made each token the whole `(62 × 5)` DE matrix; removed instance-norm.

| Result | chance → **46–61%** ✅ |
|---|---|

**This is our strongest confirmed claim.** But then:

### Step 3 — Does the *pretraining* help? → **no**

Compared pretrained vs the **identical architecture with random weights**.

| Result | PC ≈ random-init |
|---|---|

**→ Therefore:** find out why. We ablated the head, architecture, objective, and nonlinearity — all null.

### Step 4 — The features had been smoothed → **the null flips**

Public SEED DE is LDS-smoothed: only **0.08%** of variance is within-trial.

| DE variant | within-trial variance | PC − random-init |
|---|---:|---:|
| smoothed (public) | 0.08% | +2.4 |
| un-smoothed | 17.6% | **+11.0** |

**→ Therefore:** the effect is real but depends on temporal dynamics. Test on genuinely dynamic tasks.

### Step 5 — Audit of the benchmark we were chasing → **it was leakage**

The published emotion SOTA (84–92%) uses random *window* splits, placing near-duplicate adjacent windows in train and test.

| Protocol | Accuracy |
|---|---:|
| As published (leaky) | 84–92% |
| **Same code, clean trial-disjoint split** | **40–45%** |

**→ Therefore:** the target we were told to beat was inflated ~2×.

### Step 6 — Scale to 4 tasks → **looked like a clean story**

| Task | PC − random-init (frozen) |
|---|---:|
| Sleep | **+14.5** ✅ |
| Emotion (un-smoothed) | **+11.0** ✅ |
| Seizure detection | **+8.1** ✅ |
| Emotion (smoothed) | +2.4 |
| Motor imagery | −1.8 ❌ |

At this point the story appeared to be *"pretraining helps in proportion to temporal structure."*
**→ Then we ran the controls.**

---

## 3. The controls that broke the story

### Control A — dimension-matched baseline

Our encoder outputs 256-d; raw DE is 10–310-d. So we compared against **raw DE pushed through a *random* 256-d projection** — same dimensionality, no learning.

| Task | raw-DE | random 256-d proj | **PhysioFM (pretrained)** | **benefit over control** |
|---|---:|---:|---:|---:|
| Sleep | 67.86 | 69.30 | 72.63 | **+3.33** ✅ |
| Emotion | 62.75 | 60.17 | 59.99 | **−0.18** ❌ |
| Motor imagery | 51.08 | 49.92 | 40.59 | **−9.33** ❌ |

> **On emotion, our pretrained transformer performs identically to random weights.** Concatenating it with raw DE adds nothing (−0.30), so it holds **no information beyond the current window**. Part of the sleep "win" was just extra dimensions (+1.4 from a random projection alone).

### Control B — fine-tuning (what every baseline does)

We froze the encoder; all published competitors fine-tune end-to-end.

| Task | model | frozen | fine-tuned | Δ |
|---|---|---:|---:|---:|
| Sleep | PhysioFM (pretrained) | 72.63 | 75.37 | +2.74 |
| Sleep | PhysioFM (no pretrain) | 62.86 | 73.18 | **+10.32** |
| Seizure | PhysioFM (pretrained) | 75.51 | 78.66 | +3.15 |
| Seizure | PhysioFM (no pretrain) | 67.41 | **80.21** | **+12.80** |
| Seizure (3 seeds) | pretrained vs no-pretrain | — | 78.19 vs 79.11 | **Δ ≈ 0** |

```
 PRETRAINING BENEFIT (PhysioFM pretrained − PhysioFM random-init)

 Sleep     frozen  +9.8  ████████████████████
           fine-t  +2.2  ████

 Seizure   frozen  +8.1  ████████████████
           fine-t  ~0.0  ·    (3 seeds: bal-acc −0.9, AUC +0.007 — no effect)
```

> **Freezing handicapped the baseline far more than us.** Fine-tuning gains random-init +10 to +13 points and PC only ~+3.

### Control C — does label efficiency survive? (the decisive test)

Our last positive claim was *"the advantage grows when labels are scarce."*

| labels | frozen gap | fine-tuned gap |
|---:|---:|---:|
| 1% | **+12.74** | +1.90 |
| 10% | +10.75 | +2.44 |
| 100% | +9.77 | +2.19 |

> Frozen, the gap **widens** as labels shrink — the classic SSL signature. Fine-tuned it is **flat**. The label-efficiency story was a frozen-probe artefact.

---

## 4. Root-cause analysis

### Is the pretraining even working?

We compared the model against **persistence** (copy the last window, zero parameters) and **ridge** (linear map) at its *own* pretext task.

| Dataset | persistence | ridge | **model** | model/persistence |
|---|---:|---:|---:|---:|
| Sleep | 3.885 | 3.355 | **2.336** | 0.60 ✅ |
| Seizure | 36.05 | 30.21 | **20.66** | 0.57 ✅ |
| Emotion (raw) | 120.3 | 205.8 | **52.72** | 0.44 ✅ |
| Motor imagery | 62.02 | 45.91 | **21.64** | 0.35 ✅ |
| Emotion (smoothed) | **0.237** | 9.999 | 6.569 | 27.8 ❌ |

**Yes — the pretraining works**, cutting forecast error 40–65% below both baselines on 4/5 datasets.

### But pretext skill *anti-correlates* with transfer

```
   forecasting skill  ──────────────────────────►  downstream benefit
   (model/persistence, lower = better)

   Motor imagery   0.35  (BEST forecaster)   →   −1.8   (WORST transfer)
   Emotion (raw)   0.44                      →  +11.0
   Seizure         0.57                      →   +8.1
   Sleep           0.60  (WORST forecaster)  →  +14.5   (BEST transfer)
```

> **The better the model forecasts, the worse it transfers.**

**Proposed mechanism:** forecasting rewards modelling the *smooth, autocorrelated* part of DE — which is not the *class-discriminative* part. Predictive-coding pretraining should therefore help only when what is predictable is also what is discriminative.

---

## 5. We tested the mechanism — and it failed

The mechanism makes a falsifiable prediction: **seizure *prediction*** is the one EEG task whose downstream objective *is* forecasting, so pretraining should finally pay off.

| | bal-acc | AUC |
|---|---:|---:|
| raw-DE (no model) | 65.15 | 0.710 |
| **PhysioFM (pretrained)** | 66.79 | 0.769 |
| **PhysioFM (no pretrain)** | **71.34** | **0.793** |

> ❌ **Prediction refuted.** Random-init beat the pretrained model, even in the frozen regime that normally flatters pretraining.

*(A first attempt used cross-patient splits and put every arm at chance including the baseline — an uninformative setup. Corrected to the standard patient-specific protocol above.)*

---

## 6. Final results table — all settings, one protocol

| Task | metric | PhysioFM<br>(pretrained) | PhysioFM<br>(no pretrain) | raw-DE<br>(no model) | pretrain<br>benefit | vs raw-DE |
|---|---|---:|---:|---:|---:|---:|
| Sleep | acc % | **73.03** | 58.51 | 67.86 | +14.52 | **+5.17** |
| Seizure detection | bal-acc % | **75.51** | 67.41 | 72.37 | +8.10 | +3.14 (n.s.) |
| Seizure prediction | AUC | 0.769 | **0.793** | 0.710 | **−0.024** | +0.059 |
| Emotion (un-smoothed) | acc % | 51.70 | 40.70 | **55.34** | +11.01 | −3.64 |
| Emotion (smoothed) | acc % | 59.26 | 56.88 | **62.75** | +2.38 | −3.49 |
| Motor imagery | acc % | 41.71 | 43.54 | **51.08** | −1.84 | −9.37 |
| **Sleep (fine-tuned)** | acc % | **75.37** | 73.18 | — | **+2.19** | — |
| **Seizure (fine-tuned, 3 seeds)** | bal-acc % | 78.19 | 79.11 | — | **≈ 0** | — |

**Versus published SOTA** (first-pass, needs verification):

| Task | ours | published SOTA | gap |
|---|---:|---:|---:|
| Sleep-EDF (5-class, subject-indep.) | 75.4 (κ 0.672) | 81–85 (κ 0.77–0.83) | **−6 to −10** |
| CHB-MIT (cross-patient) | AUC 0.867 | AUC 0.91–0.99 | **−0.05 to −0.12** |

---

## 7. Conclusions

### What works ✅
1. **Structured `(C×B)` patching + no instance-norm** — recovers signal univariate models destroy (chance → 60%). Our strongest, best-supported claim.
2. **The architecture as a classifier** — sleep 75.4% (κ 0.672), seizure AUC 0.867, both above DE linear baselines.
3. **The pretext task itself is learned** — 40–65% below persistence and ridge.

### What does not ❌
1. **Predictive-coding pretraining** — +2.2 on sleep fine-tuned; null or negative on all six other settings.
2. **The label-efficiency claim** — an artefact of frozen-encoder evaluation.
3. **Our proposed mechanism** — made a falsifiable prediction; the prediction failed.

### Methodological findings (of value beyond this project) 🔬
1. **Frozen-probe evaluation inflates SSL gains ~5×** (+9.8 → +2.2 on sleep; +8.1 → −1.6 on seizure). Widely used in EEG-SSL papers.
2. **A published EEG-SSL benchmark is inflated ~2× by temporal-neighbour leakage** (84–92% → 40–45%).
3. **Dimension-matched controls are essential** — our encoder equalled a *random projection* on emotion.
4. **Pretext skill can anti-correlate with transfer** — optimising the SSL objective harder can make representations worse.

### Why it does not work — the honest account

```
  Layer 1  DE deletes ~600× of the signal before the model sees it
           (30 s × 100 Hz × 2ch = 6000 samples → 10 numbers)
                          ↓
  Layer 2  What remains is linearly saturated — a linear probe already
           reads DE optimally, so an encoder can only reorganise it
                          ↓
  Layer 3  Forecasting optimises the smooth, predictable component,
           which is not the discriminative one
                          ↓
  Layer 4  Frozen evaluation hides all of the above by handicapping
           the baseline more than the model
```

No single layer is fatal. Together they explain why a method that *sounds* like it should work delivers ~2 points.

---

## 8. What we would do differently

| Change | Rationale |
|---|---|
| Replace DE with **raw EEG** or rich time–frequency | DE deletes morphology (sleep spindles, K-complexes, spike-waves) — the actual class signatures |
| Predict in **latent space** (JEPA-style), not input space | Removes the "learn the smooth part" failure mode |
| **Always report fine-tuned** numbers | Frozen probes systematically inflate SSL gains |
| Run **dimension-matched controls** from day one | Would have caught the emotion result immediately |

⚠️ **Novelty caution:** BrainGPT/EEGPT (arXiv 2410.19779) already published causal decoder-only autoregressive pretraining on raw EEG at ~1B tokens, and Laya already applies JEPA to EEG. Swapping DE→raw EEG alone reproduces published work.

---

## 9. Recommended framing

> **Not:** *"We built an EEG foundation model."*
>
> **But:** *"We built a structured-patch causal transformer for EEG, and rigorously characterised when predictive-coding pretraining does and does not help — finding that most reported benefit in this setting is an artefact of frozen-probe evaluation."*

The negative result is well-controlled, reproducible, and methodologically useful. The controls we ran (dimension-matched baseline, pretext-vs-transfer correlation, frozen-vs-fine-tuned) are not standard practice in EEG-SSL papers, and applying them to *other* models would likely revise their reported gains too.

---

*Every number traces to `docs/experiments/EXP-*` and `results/`; figures regenerate via `scripts/make_figure_*.py`.*
