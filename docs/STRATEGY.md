# PhysioFM — Publication Strategy (Option A) & Consolidated Data Analysis

*Living strategy doc. Source of truth for "what is the paper, why did emotion fail,
and how do we make it work." Pairs with the experiment log in `docs/experiments/`.*
Last updated: 2026-07-06.

---

## 1. The contribution the proposal actually promises

Per `URE/PhysioFM_Proposal.docx.pdf`, PhysioFM is **a cross-task physiological-signal
foundation model**: a decoder-only transformer, pretrained with a **predictive-coding**
objective on **structured (C×B) DE patches**, whose value is **transferable
representations across multiple EEG tasks** (emotion, **sleep, seizure, motor imagery**).

Claimed differentiators:
- vs **PC-SSL**: variable context length, longer temporal dependencies, multi-horizon.
- vs **TimesFM**: channel/band spectral structure awareness.

**Emotion (SEED) is ONE validation task** — the one used to isolate the structured-patch
ablation. The payoff is **Phase 3: multi-dataset pretraining + cross-task evaluation.**

## 2. The reframe (why we were measuring the wrong thing)

Every differentiator above — variable context, long dependencies, cross-task transfer —
is **irrelevant on emotion DE**, because emotion DE is *static-spectral and linearly
saturated*. We proved this exhaustively. So the FM's strengths can only appear on tasks
with **genuine temporal dynamics**, and the honest headline metric is **label-efficiency
+ transfer**, not peak within-subject accuracy.

## 3. Precise diagnosis — why the emotion leg failed (with numbers)

| Finding | Evidence | Number |
| --- | --- | ---: |
| Emotion DE is **linearly saturated** — nonlinear ≤ linear, no headroom | F10 / [[EXP-0010]] | rbf/mlp ≤ logreg on all subj-dep |
| **LDS-smoothed** DE has ~no learnable dynamics → SSL pretext degenerate | F1 / [[EXP-0001]] | within-trial var 0.1%; persistence MSE 1e-5 |
| Not the **objective** (masked ≡ forecasting; both die on smoothed, both help on raw) | F9 / [[EXP-0011]] | +~9 on raw, ≤ random on smoothed |
| The **PC-SSL 84–92% SOTA was ~80% leakage**; clean ≈ 40–45% | F12 / [[EXP-0008]] | leaky 66–70 → clean 40–45 |
| **Subject-invariance on DE** doesn't rescue it; CORAL *collapses* (removes abs. level) | F14 / [[EXP-0012]] | dann_adv ≈ dann_l0; coral −20 pts |
| Not the **architecture** or **head** | F3/F4 | random mixer ≈ pretrained; MLP ≤ linear probe |
| The **one positive on emotion**: label-efficiency on *raw* DE | F7 / [[EXP-0007]] | PC > raw-DE +6 at 10% labels |

**One line:** emotion-DE failure is *task/feature-induced* (static, saturated, smoothed,
illusory SOTA) — **not** a flaw in the method, architecture, head, or objective.

## 4. The positive pivot — where it DOES work (the keystone)

**F13 / [[EXP-0009]] — sleep staging (dynamic task), DEFINITIVE (78 subj, full
Sleep-EDF Cassette, 195k epochs), subject-disjoint 5-fold, logreg:**

| Features | acc % | macro-F1 % | κ |
| --- | ---: | ---: | ---: |
| **physiofm_pc** | **72.6** | **67.6** | **0.635** |
| raw_de (logreg) | 67.9 | 61.9 | 0.575 |
| **physiofm_rand** | **62.9** | **56.1** | **0.509** |

**PC-pretrained beats matched random-init by +9.8 pts / +0.13 κ** — the mirror image
of the emotion null. *Confirms the thesis: PC pretraining helps ∝ a task's temporal
dynamics.* And at full scale the FM now **also beats the raw-DE linear ceiling by
+4.8 pts** (raw-DE degrades with more subjects; the FM improves with more pretraining
data) — so it wins on **peak accuracy**, not just pretraining-gain, reversing the
preliminary's caveat. Remaining: single-seed → repeat ≥3 seeds; paired per-fold test
for the pc−raw margin; label-efficiency curve (the expected larger low-label win).
*(9-subj preliminary, now superseded: pc 68.8 / raw 72.1 / rand 50.0.)*

## 5. Thesis for Option A (the paper)

> **PhysioFM: a lightweight, interpretable predictive-coding foundation model on
> structured spectral (DE) features that transfers across physiological tasks.** Its
> pretraining value scales with a task's temporal dynamics — negligible on static-spectral
> emotion (which we characterize mechanistically), substantial on dynamic tasks (sleep;
> +19 pts over random), and concentrated in the **low-label / transfer** regime where
> foundation models are supposed to help. We also correct an inflated emotion SSL benchmark
> (PC-SSL, ~80% leakage) that the sub-field has been chasing.

This makes the emotion null a **feature** (rigorous "when/why it works"), not a failure.

## 6. How to make Option A work — the plan

1. **Task portfolio:** lead with **dynamic** tasks (sleep ✅, then seizure CHB-MIT and/or
   motor-imagery BCI-IV-2a). **Emotion = the static-spectral negative control.**
2. **Feature:** never LDS-smooth (F1). Use un-smoothed DE; seriously consider **raw-EEG**
   tokenizer ([[EXP-0013]]/F15) — DE is a saturated bottleneck for emotion (per-task differs).
3. **Headline metric:** **label-efficiency curves (10/50/100%)** + cross-task transfer +
   pretraining-gain over random. Report κ for sleep. NOT peak within-subject accuracy.
4. **Baselines (honest):** drop the 84–92 PC-SSL target; benchmark vs **clean-protocol
   PC-SSL**, **raw-DE linear**, task-specific SSL (TS-TCC/BENDR/DeepSleepNet), and **≥1
   modern EEG-FM** (EEGPT or CBraMod) for credibility. Strict inductive / subject-disjoint
   splits everywhere; beware transductive-DA inflation.
5. **Positioning:** the cheap, interpretable, non-contrastive-PC, DE-based alternative to
   heavy raw-signal contrastive/masked FMs (LaBraM/EEGPT/CBraMod) — competitive in low-data
   cross-task transfer.
6. **Multi-task pretraining infra:** channel-agnostic / zero-padded patch so 2-ch sleep,
   62-ch emotion, 22/23-ch MI/seizure can be jointly pretrained (proposal Phase 3).
7. **Venue realism:** JBHI / ICASSP / EMBC (NeurIPS/ICML unrealistic vs LaBraM-class work).

## 7. Concrete next experiments

- **F13 definitive** — ✅ done (78 subj, full corpus): pc 72.6 > raw 67.9 > rand 62.9.
- **Sleep label-efficiency curve** (F7-analog) — ✅ done ([[EXP-0009]] §4c): pc−rand gain
  widens +9.8→+12.7 as labels drop to 1%; pc@1% beats both baselines @100%. The headline.
- **Multi-seed (3) + paired test** — in progress (`run_f13_multiseed.sh`).
- **+1 dynamic task** (seizure or MI) for the "multi-task" claim (data-blocked).
- **Multi-task joint pretraining** (channel-agnostic) once ≥2 tasks' DE are on disk.
- **(Optional) raw-EEG leg** (F15) if the DE bottleneck caps the dynamic tasks too.

## 8. Honest risks

- A DE-based FM likely won't beat strong raw-signal baselines (DeepSleepNet, EEG-Conformer)
  on **peak** accuracy — must win on transfer/label-efficiency/interpretability/cost.
- Sleep preliminary is 9 subjects / single seed — needs the full-corpus, multi-seed run.
- Cross-task "foundation model" claims need ≥2–3 tasks; one task = "SSL helps sleep", not a
  foundation model.

*Refs (see also each EXP §8): EEG-FM-Bench (arXiv:2508.17742); CBraMod (ICLR 2025); EEGPT
(NeurIPS 2024); EEG-FM review (arXiv:2507.11783); SSL label-efficient sleep (arXiv:2510.07960);
CLISA (arXiv:2109.09559).*
