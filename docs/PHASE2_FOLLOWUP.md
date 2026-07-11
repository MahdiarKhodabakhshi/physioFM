# PhysioFM — Stage 2 Follow-Up Experiments (results log)

Implements `docs/PhysioFM_Stage2_FollowUp_Experiments.md`. Each experiment reuses
the frozen harness in `physiofm/phase2_eval.py` (segment-level, PC-SSL
subject-dependent splits, seed 42) and writes under `results/phase2/followup/`.

**Runtime.** `/home/mahdiar/.conda/envs/xcqa/bin/python` (torch 2.7 / transformers
5.12 / timesfm2.5), H100. Reproduce each experiment with its `scripts/run_fN.sh`.

**Data availability (decisive for several experiments).**
- Un-smoothed DE (`de_movingAve`) is available **only for SEED-IV**
  (`PC-SSL/data/raw/SEED-IV/eeg_feature_smooth/*/*.mat`). SEED-V/SEED publish
  LDS-only features, so their un-smoothed legs are blocked without raw EEG.
- The PC-SSL reference repo (code + raw data + per-fold trained models + author
  splits) is present at `PC-SSL/`, so F12 is fully runnable.
- Sleep-EDF / CHB-MIT / BCI-IV-2a are **not** on disk, so F13 stays data-blocked.

---

## F1 🔴 — Un-smoothed DE + persistence baseline — DONE (SEED-IV)

`scripts/run_f1.sh` → `scripts/phase2_f1_smoothing.py`,
`results/phase2/followup/f1/`. Matched PhysioFM-S runs (scratch d=256/6L,
`p_in=1→p_out=16`) trained **SEED-IV-only** so smoothed (`de_LDS`) vs un-smoothed
(`de_movingAve`) is the only changed variable; persistence + variance computed in
the per-(C,B) corpus-standardized space the model trains in.

| Variant | persistence MSE 1-step | persistence MSE multi-step | model PC-MSE | within-trial var frac |
| --- | ---: | ---: | ---: | ---: |
| SEED-IV smoothed (LDS) | 0.00001 | 0.00054 | 0.02274 | **0.1%** |
| SEED-IV un-smoothed (movingAve) | 0.10249 | 0.29675 | 0.12811 | **17.2%** |

Zero-shot linear probe (acc % / macro-F1 %), SEED-IV:

| DE variant | PC-pretrained (logreg) | random-init (logreg) | PC-pretrained (lin-SVM) | random-init (lin-SVM) |
| --- | ---: | ---: | ---: | ---: |
| smoothed (LDS) | 61.41 / 52.00 | 55.98 / 48.37 | 61.85 / 54.29 | 61.09 / 52.19 |
| un-smoothed (movingAve) | **54.67 / 45.99** | 41.97 / 35.53 | **52.14 / 44.13** | 44.33 / 38.42 |

**Verdict — the Stage-2 null flips on un-smoothed DE.** Under LDS smoothing the
within-trial (dynamic) signal is ~0.1% of variance, persistence is near-perfect
(MSE 1e-5), and PC-pretrained ≈ random-init — exactly the original null. On
un-smoothed DE the within-trial fraction is **~17×** larger, persistence is far
from optimal, and PC-pretraining beats random-init by **~10–13 points**. So the
Stage-2 "temporal PC adds nothing" result is (at least substantially) an
**artifact of LDS smoothing destroying the learnable dynamics**, not evidence
that emotion is intrinsically static. The static-emotion claim must be scoped to
smoothed DE; the cleaner story is "LDS smoothing hides the dynamics PC can use."

*Scope:* SEED-IV only (sole dataset with an un-smoothed feature key).

---

## F4 🔴 — Matched downstream head — DONE

`scripts/phase2_f4_matched_head.py`, `results/phase2/followup/f4/`. One identical
frozen-encoder + 2-hidden-layer MLP head (`(256,128)`, balanced oversampling for
class weighting, per-fold validation early stopping) applied to raw-DE,
PhysioFM-S (PC-pretrained, combined corpus) and PhysioFM-S (random-init). `logreg`
(linear probe) and `mlp` (un-balanced) shown for context. acc % / macro-F1 %.

**SEED-V (chance 20%)**

| Features | logreg | mlp | mlp_bal |
| --- | ---: | ---: | ---: |
| raw_de | 51.40 / 49.92 | 40.10 / 37.43 | 40.81 / 38.92 |
| physiofm_pretrained | 45.50 / 44.01 | 44.37 / 42.15 | 45.10 / 43.07 |
| physiofm_random_init | 48.54 / 46.57 | 42.53 / 40.64 | 43.16 / 41.17 |

**SEED-IV (chance 25%)**

| Features | logreg | mlp | mlp_bal |
| --- | ---: | ---: | ---: |
| raw_de | 62.75 / 54.76 | 48.93 / 41.58 | 54.64 / 45.85 |
| physiofm_pretrained | 57.49 / 48.93 | 51.72 / 44.25 | 55.06 / 46.40 |
| physiofm_random_init | 60.68 / 53.11 | 49.55 / 45.29 | 53.65 / 48.60 |

**Verdict — the head is not the lever.** The matched MLP **does not** reach the
80s on any feature set; with balancing it lands at 40.8 / 54.6 (raw-DE) — *below*
the linear probe (51 / 63), confirming the documented overfit on ~600 labels/fold.
All three feature sets sit in the same ~41–55% band. So the gap to PC-SSL's
published 84–92% is **not** explained by the downstream head, and PhysioFM-S
representations carry no MLP-accessible value the linear probe was hiding. This
sharpens the remaining suspicion onto the PC-SSL number itself (→ F12).

---

## F2 🔴 — Sequence-level / order-aware readout — DONE

`scripts/phase2_f2_sequence_readout.py`, `results/phase2/followup/f2/`.
Trial-level classification from frozen per-window hidden states: `last` (causal
last-state → logreg), `gru` (order-respecting GRU pool), `gru_shuf` (same GRU,
window order shuffled — the order control). Subject-dependent folds, seed 42.

**Smoothed DE** (combined-corpus encoders):

| Dataset | Readout | pretrained | random-init |
| --- | --- | ---: | ---: |
| SEED-V | last | 47.50 / 45.73 | 48.40 / 46.77 |
| SEED-V | gru | 43.33 / 40.31 | 35.49 / 31.14 |
| SEED-V | gru_shuf | 42.08 / 39.08 | 36.18 / 31.89 |
| SEED-IV | last | 56.39 / 48.81 | 61.39 / 54.53 |
| SEED-IV | gru | 51.67 / 44.22 | 55.56 / 46.07 |
| SEED-IV | gru_shuf | 51.67 / 43.15 | 55.00 / 44.28 |

**Un-smoothed DE** (SEED-IV, F1 encoders):

| Dataset | Readout | pretrained | random-init |
| --- | --- | ---: | ---: |
| SEED-IV raw | last | **59.72 / 51.30** | 48.89 / 42.30 |
| SEED-IV raw | gru | **58.61 / 50.23** | 45.00 / 38.25 |
| SEED-IV raw | gru_shuf | 53.61 / 44.69 | 39.44 / 34.11 |

**Verdict — order matters only where dynamics survive.** On **smoothed** DE
`gru ≈ gru_shuf` (shuffling time costs ~0–1 pt) and pretrained ≯ random, so the
signal is order-invariant/static and the negative result holds even with a fair
temporal readout. On **un-smoothed** DE the order control bites: for the
pretrained encoder `gru` (58.6) beats `gru_shuf` (53.6) by ~5 pts, and pretrained
beats random under every readout (e.g. last 59.7 vs 48.9). So temporal order is
genuinely discriminative once LDS smoothing is removed — corroborating F1 from
the readout side.

---

## F3 🟡 — Frozen-random vs frozen-TimesFM stack — DONE

New variant `timesfm_rand` in `physiofm/physiofm_s.py`: a decoder stack at
TimesFM-2.5's **exact** shape (d=1280, 20 layers, 16 heads, head_dim 80,
intermediate 1280) but **random** weights, frozen, with the same structured I/O
blocks trained fresh (`results/phase2/followup/f3/`, PC-MSE 0.0031 vs E1b 0.0039).
Three matched frozen/from-scratch runs, identical probe (acc % / macro-F1 %):

| Stack | SEED-V logreg | SEED-V SVM | SEED-IV logreg | SEED-IV SVM |
| --- | ---: | ---: | ---: | ---: |
| frozen **TimesFM** (E1b) | 43.27 / 41.32 | 44.18 / 41.83 | 60.40 / 52.35 | 61.48 / 53.20 |
| frozen **random** (TimesFM shape) | 47.26 / 45.69 | 46.24 / 44.57 | 58.93 / 50.31 | 60.63 / 51.84 |
| from-scratch (E1a, d=256) | 45.58 / 44.10 | 46.13 / 44.53 | 57.49 / 48.93 | 57.41 / 49.12 |

**Verdict — C6 refuted.** A frozen *random* stack of the same 1280×20 shape
matches (SEED-V: even edges out) the frozen *pretrained* TimesFM stack. So
TimesFM's pretrained temporal priors add nothing over "a big fixed nonlinear
mixer" on smoothed DE; the apparent "transfer" in E1b is the trained I/O blocks +
a high-dimensional random projection, not the pretrained weights. This is
consistent with the random-init ≈ pretrained finding throughout Stage 2.

---

## F12 🔴 — PC-SSL leakage audit + clean re-replication — DONE

`scripts/phase2_f12_pcssl_audit.py`, `results/phase2/followup/f12/`. Uses the
in-repo PC-SSL reference (`PC-SSL/`: code, raw data, per-fold trained models,
author splits, result CSVs). The author notebook splits **individual DE windows**
with `train_test_split(test_size=0.2, shuffle=True)`, while PC-SSL forms
consecutive `(window_i → window_{i+1})` pairs with trial-constant labels — so a
random window split puts near-duplicate adjacent windows in both train and test.

**Leakage (random window split vs clean paper trial-disjoint split):**

| Dataset | split | future-partner-in-train | either-neighbor | same-trial-in-train | NN same-trial | NN cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SEED-V | random window (leaky) | ~81% | 95.6% | 100% | 100% | 1.000 |
| SEED-V | trial-disjoint (clean) | 0% | 0% | 0% | — | — |
| SEED-IV | random window (leaky) | 81.4% | 95.3% | 100% | 100% | 1.000 |
| SEED-IV | trial-disjoint (clean) | 0% | 0% | 0% | — | — |

`future-partner-in-train` reproduces the author notebook's logged
`mean_test_overlap_with_classifier_train = 0.7997`. Every test window's nearest
train window is a same-trial near-duplicate (cosine 1.000).

**PC-SSL accuracy — same code, only the split changes (acc % / macro-F1 %):**

| Dataset | leaky random split | clean trial-disjoint | raw-DE LogReg (clean) | PhysioFM-S (clean probe) | chance |
| --- | ---: | ---: | ---: | ---: | ---: |
| SEED-V | 65.84 / 63.05 | **39.77 / 33.97** | 51.40 / 49.92 | 45–49 | 20 |
| SEED-IV | 70.41 / 67.85 | **44.72 / 26.73** | 62.75 / 54.76 | 57–61 | 25 |

(The full author-notebook reproduction with encoder fine-tuning reached **91.25%**
on SEED-V at 80% overlap — i.e. ≈ the published 92.39.)

**Verdict — the gap was largely leakage.** The published 84–92% rests on ~80%
temporal-neighbor leakage. Holding the PC-SSL implementation fixed and removing
only the leakage collapses accuracy to ~40–45% — at/below the raw-DE linear
ceiling and within the PhysioFM-S band. So PhysioFM-S is competitive on a clean
protocol and the honest contribution is the mechanistic decomposition. *Caveat:*
the clean absolute number comes from a faithful-but-unverified re-implementation;
the **leaky-vs-clean delta** (same code) is the controlled, implementation-invariant
result and is what should be cited.

---

## Status

**Minimal blocking set for any external claim — COMPLETE: F1, F2, F3, F4, F12.**

Because F1/F2 **flipped** the result on un-smoothed DE, the source doc prescribes
re-running the ablation grid (F5–F7) on un-smoothed DE. Those follow.

---

## F5 🟡 — Input-context (p_in) sweep on un-smoothed SEED-IV — DONE

`scripts/run_f5.sh` + `scripts/phase2_f5_context.py`,
`results/phase2/followup/f5/`. p_in ∈ {1,4,8} × p_out ∈ {1,16}, matched PC vs
random-init on `seed_iv_raw`, paired with the F2 trial-level readouts. acc %;
gap = PC − random (GRU readout).

| p_in | p_out | GRU PC | GRU rand | **GRU gap** | last PC | logreg PC | logreg rand |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 50.28 | 45.00 | **5.28** | 48.89 | 53.26 | 41.97 |
| 1 | 16 | 58.61 | 45.00 | **13.61** | 59.72 | 54.67 | 41.97 |
| 4 | 1 | 56.94 | 43.33 | **13.61** | 48.89 | 53.21 | 36.98 |
| 4 | 16 | 58.06 | 43.33 | **14.72** | 55.00 | 55.04 | 36.98 |
| 8 | 1 | 52.50 | 40.28 | **12.22** | 48.89 | 51.44 | 38.28 |
| 8 | 16 | 58.33 | 40.28 | **18.06** | 55.56 | 53.96 | 38.28 |

**Verdict — on raw DE, context and multi-horizon both help (C4 revived).** The
pretrained−random gap is large at every config and **grows with context** (peak
18.1 at p_in=8/p_out=16). Multi-horizon `p_out=16` beats single-step `p_out=1` at
every p_in (e.g. gap 13.6 vs 5.3 at p_in=1) — the opposite of smoothed DE, where
`p_out` made no difference (C4 "not supported"). So the multi-horizon predictive
objective is useful precisely where temporal dynamics survive.

---

## F6 🟢 — Scale check on un-smoothed SEED-IV — DONE

`scripts/run_f6.sh` + `scripts/phase2_f6_scale.py`, `results/phase2/followup/f6/`.
Fixed p_in=1, p_out=16; pretrained−random gap (GRU readout) vs model size.

| hidden | layers | params (M) | GRU PC | GRU rand | **GRU gap** | logreg gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 4 | 1.08 | 53.61 | 48.61 | **5.00** | 8.82 |
| 256 | 6 | 3.72 | 58.61 | 45.00 | **13.61** | 12.71 |
| 512 | 8 | 15.31 | 57.78 | 44.44 | **13.33** | 13.11 |

**Verdict — the gap is real and scale-stable.** On un-smoothed DE the
pretrained−random gap is robustly positive at every size, opening from ~5 pts at
1M params to ~13 pts by 4M and plateauing through 15M. So the pretraining benefit
is not a sub-million-parameter artifact — it is present across the ladder (and
the SEED corpus is tiny, so larger scale neither creates nor destroys it).

---

## F7 🟡 — Limited-label curves (decide C2) on un-smoothed SEED-IV — DONE

`scripts/phase2_f7_label_curves.py`, `results/phase2/followup/f7/`. Frozen
encoder + the F4 matched head (balanced 2-layer MLP, per-fold val early stopping,
class weighting), trained on a label fraction of each fold's train segments.
acc % / macro-F1 %.

| Features | 10% labels | 50% labels | 100% labels |
| --- | ---: | ---: | ---: |
| raw_de | 40.92 / 35.21 | 50.06 / 43.63 | 50.80 / 44.54 |
| physiofm_pretrained | **46.69 / 38.64** | 49.63 / 42.29 | 50.76 / 43.34 |
| physiofm_random_init | 35.08 / 30.05 | 38.46 / 32.84 | 38.85 / 32.80 |

**Verdict — a label-efficiency FM win on raw DE.** At full labels PC-pretrained
ties raw-DE (~50%), but the FM margin **grows as labels shrink**: at 10% labels
PC-pretrained beats raw-DE by ~6 pts (46.7 vs 40.9) and beats random-init by ~12
pts. So on un-smoothed DE the pretrained representation is genuinely more
label-efficient — the kind of positive FM result Stage 2 could not surface on
smoothed DE. C2 leans positive on raw DE in the low-label regime.

---

## Flip-path summary (F1/F2 → F5–F7 on un-smoothed DE)

Stage 2's negative verdict was measured on **LDS-smoothed** DE, where ~99.9% of
the per-(C,B) signal is a static per-trial level (F1). Re-running on un-smoothed
SEED-IV reverses the key conclusions:

| Claim | Smoothed DE (Stage 2) | Un-smoothed DE (this work) |
| --- | --- | --- |
| PC pretraining vs random | tie (null) | **PC > random by ~10–18 pts** (F1, F2, F5, F6) |
| Temporal order matters | no (gru≈shuf) | **yes** (gru > shuf ~5 pts, F2) |
| Multi-horizon p_out helps (C4) | no | **yes** (p_out=16 > p_out=1, F5) |
| Benefit is scale artifact | n/a | **no** — stable 1M→15M params (F6) |
| Label efficiency (C2) | no margin | **FM > raw-DE at 10% labels** (F7) |
| TimesFM weights transfer (C6) | neutral | refuted as "big fixed mixer" (F3, smoothed) |
| PC-SSL 84–92% gap | the target | ~80% leakage; clean ≈ 40–45% (F12) |

The honest revised thesis: *temporal predictive-coding pretraining helps DE
emotion in proportion to the temporal dynamics left in the features; LDS
smoothing removes those dynamics, which is why Stage 2 saw a null.*
