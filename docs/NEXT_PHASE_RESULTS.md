# Next-phase plan — results (Gates 0–3)

*Executes `docs/NEXT_PHASE_PLAN.md` end to end, 2026-08-18/19. Every number traces to
`results/phase4/` and the lab-notebook entries [[EXP-0020]]–[[EXP-0023]]. Naming as in
`docs/FINAL_REPORT.md` §0: **PhysioFM (pretrained)** = our model with input-space
predictive-coding pretraining (`physiofm_pc`), **PhysioFM (latent)** = the same model with the
new latent-target objective (`physiofm_latent`), **PhysioFM (no pretrain)** = identical
architecture, random init (`physiofm_rand`), **raw features** = features straight into logistic
regression (no model). "Fine-tuned" = end-to-end fine-tuning on the identical subject-/
patient-disjoint folds used everywhere in this project — the fair protocol; frozen-probe numbers
are reported only to show (again) how much they inflate pretraining.*

Compute: local H100-20C (20 GB) + a RunPod H100 80 GB for the raw-EEG stage; SDPA attention
(numerically identical to the original eager kernel, verified to 3e-6). Reproduction check before
anything new: sleep DE PC pretraining gives the pod's July best PC-MSE `0.22886` exactly, and the
frozen / fine-tuned sleep and seizure numbers reproduce July's (72.62 / 75.37 / 73.18; 67.47 /
0.741, 80.21 / 0.874).

---

## 0. One-table summary

| Gate | Question (from the plan) | Verdict |
|---|---|---|
| **0** | Does *less compression* — 64 log-spaced spectral bins instead of 5 DE bands — restore representational headroom (R1)? | **Partly, sleep only, and it helps the architecture, not the pretraining.** Sleep: linear ceiling 67.9 → 72.8; the whole fine-tuned model 75.4 → **77.9 ± 0.4 (κ .71, 4 seeds)**; nonlinear headroom +2.3 (DE: +1.7); pretraining benefit shrinks +2.5 → **+0.85**. Seizure: nothing (linear 72.4 = 72.4; best nonlinear 72.6 vs 74.0). |
| **1** | Does predicting in *latent* space (JEPA/data2vec-style targets) fix the objective (R2)? | **No.** Fine-tuned, latent-PC ≤ input-space PC on both per-epoch tasks (sleep 73.7 vs 75.5 over 4 seeds; seizure 79.7 vs 78.4, within ±12 per-patient noise) and ≈ random-init (73.1 / 80.2). Five objective variants: 72.7–74.0. Mechanism: on DE the latent predictor never beats trivial baselines (skill −0.2 to −0.6) — the target drifts to a smooth trajectory; on raw/short-trial data it does learn (skill 0.4–0.6) but that skill still anti-correlates with transfer. |
| **2** | Do raw-EEG structured tokens escape the DE bottleneck, and does structured spatial patching beat per-electrode decomposition? | **Parity, not escape; and the ablation goes the other way on 2-channel sleep.** Raw 200 ms tokens (29 M tokens, 10 pretraining epochs, 3 fine-tuning epochs): input-PC 75.5 · latent 74.7 · random 74.2 (= the DE pipeline's 75.4, below tf64's 77.9). Frozen inflation is the largest yet (+14.7 frozen → +1.2 fine-tuned). Per-electrode tokens: 76.4 / 75.5 / 74.6 — ≥ structured on every arm. |
| **3** | Under a streaming constraint, does the causal decoder legitimately beat its bidirectional twin? | **Yes.** Same stack, bidirectional attention: 74.1 offline → **70.4** when only the past is visible; causal 73.2 (random-init) / 75.4 (pretrained) in both settings, at 1 token per decision (with a KV cache) vs 190. Causal beats bidirectional online by **+2.8 / +5.0**. |

**Bottom line for the plan.** Neither fix (R1 richer input, R2 latent targets) rescues
predictive-coding pretraining: under fair evaluation it is worth +2.5 (DE), +0.85 (tf64), +1.2
(raw, 1 seed) points on sleep and nothing on seizure, whatever the target space. What the gates
*did* deliver: (i) the architecture claim got stronger — a 2.4 M-parameter structured causal
transformer reaches **77.9 % / κ 0.71** on Sleep-EDF-78 subject-disjoint (SOTA raw-signal models:
81–85), (ii) the causal design has a measurable, defensible advantage in streaming use, and
(iii) the frozen-probe inflation finding is reconfirmed on every new representation.

---

## 1. Gate 0 — rich time–frequency tokens (tf64) · [[EXP-0020]]

**Setup.** Same recordings, epoching, wake-trimming and label companions as the DE archives
(asserted identical at build time); per-epoch feature = Welch log-power in 64 log-spaced bins
0.5–49 Hz per channel (`physiofm/spectral.py`, `scripts/build_tf_dataset.py`). Sleep 128-d,
seizure 1152-d tokens. Saturation heads: logistic regression (linear) vs balanced 2-layer MLP vs
gradient-boosted trees (HGB); folds identical to the model evaluators (sleep subject-disjoint
5-fold seed 42; seizure LOPO with a 60 k stratified train cap — the cap reproduces the uncapped
raw-DE number, 72.41 / 0.807). `scripts/gate0_saturation.py`.

### 1a. Linear-saturation test

| task | features | linear (logreg) | MLP | HGB | headroom = best nonlinear − linear |
|---|---|---:|---:|---:|---:|
| Sleep (acc %) | DE (10-d) | 67.86 | 67.20 | **69.60** | +1.7 |
| Sleep (acc %) | **tf64 (128-d)** | **72.83** | 71.85 | **75.17** | **+2.3** |
| Seizure (bal-acc / AUC) | DE (90-d) | 72.41 / .807 | 69.54 / .824 | **74.02 / .851** | +1.6 / +.044 |
| Seizure (bal-acc / AUC) | tf64 (1152-d) | 72.41 / .809 | 69.92 / .824 | 72.63 / .849 | +0.2 / +.040 |

Pre-registered rule (headroom(tf64) ≥ headroom(DE) + 2 **and** ≥ 2 absolute): **fails** on both
tasks. But on sleep tf64 raises the *linear* ceiling by five points — DE was discarding linearly
useful spectral detail — and a plain HGB on tf64 (75.2) already equals the fine-tuned DE-PhysioFM
(75.4).

### 1b. PC ladder on sleep tf64 (`results/phase4/gate0/sleep_edf_tf64/`)

| arm | frozen probe (seed 42) | fine-tuned, seed 42 | fine-tuned, 4 seeds (42,1,2,3) |
|---|---:|---:|---:|
| raw tf64 → logreg | 72.83 | — | — |
| raw tf64 → random 256-d projection (dimension-matched control) | 73.30 | — | — |
| PhysioFM (pretrained, input-space) | **76.75** (κ .690) | 77.43 (κ .701) | **77.88 ± 0.37** |
| PhysioFM (latent) | 71.02 | 76.87 | 77.31 ± 0.36 |
| PhysioFM (no pretrain) | 65.51 | 77.11 (κ .694) | **77.03 ± 0.19** |
| concat [raw tf64 ‖ PC] | 77.41 | — | — |

Frozen, pretraining still looks large (+11 over random-init, +3.5 over the dimension-matched
projection); fine-tuned, **PC − random-init = +0.85** (sign consistent in 4/4 seeds; was +2.5
on DE). Richer input lifts the *architecture* by +4 (random-init 73.1 → 77.0) and removes most
of what was left of the pretraining benefit. Seizure tf64 ladder: not run (skipped after the
saturation test showed zero headroom; the plan's own stop rule).

---

## 2. Gate 1 — latent-target predictive coding · [[EXP-0021]]

**Setup.** `scripts/phase2_pretrain.py --objective latent`: EMA target encoder (0.996 → 1,
cosine schedule), stop-gradient, MLP predictor h_j → ẑ_{j+1..j+16}, targets instance-normalised
over time within each sequence (data2vec-style anti-collapse), MSE; matched arms per task, seed
42, 60 epochs, p_in 1 / p_out 16 (8 on MI); the EMA target and predictor are stored in the
checkpoint. Frozen + fine-tuned evaluators as before. Multi-seed on sleep (pretraining seeds
42, 1, 2, 3; fine-tuning seed fixed).

### 2a. Sleep (Sleep-EDF, 78 subj, subject-disjoint 5-fold), fine-tuned end-to-end

| arm | seed 42 | seed 1 | seed 2 | seed 3 | **mean ± sd** | κ (s42) |
|---|---:|---:|---:|---:|---:|---:|
| PhysioFM (pretrained, input-space) | 75.37 | 75.09 | 75.92 | 75.76 | **75.54 ± 0.32** | .672 |
| PhysioFM (latent) | 74.12 | 73.89 | 72.93 | 73.82 | **73.69 ± 0.45** | .656 |
| PhysioFM (no pretrain) | 73.18 | 72.55 | 73.83 | 72.76 | **73.08 ± 0.49** | .645 |

Input-space PC keeps a consistent **+2.5** over random-init (4/4 seeds); latent-PC gives
**+0.6** and is **−1.9 below input-space PC**. Frozen (seed 42): input-PC 72.62 · latent 66.08 ·
random 62.86 · raw-DE 67.86 — the latent representation is *less* linearly readable than raw DE.

**Objective variants (sleep, seed 42, fine-tuned):** delta targets (predict z_{j+k} − z_j) 73.87 ·
cosine loss without time-normalisation 72.74 · within-sequence variance term 73.20 · EMA 0.99
74.00 · p_out 4 74.02. None reaches input-space PC (75.37); all sit at the random-init level.

### 2b. Seizure (CHB-MIT, 24 patients, LOPO), seed 42

| arm | frozen bal-acc / AUC | fine-tuned bal-acc / AUC |
|---|---:|---:|
| raw-DE → logreg | 72.37 / .806 | — |
| PhysioFM (pretrained, input-space) | 77.42 / .852 | 78.36 / .863 |
| PhysioFM (latent) | 71.36 / .777 | 79.74 / .879 |
| PhysioFM (no pretrain) | 67.47 / .741 | **80.21 / .874** |

Per-patient sd is ±11–16 bal-acc, so the three fine-tuned arms are indistinguishable (July's
3-seed result: pretraining Δ ≈ 0). Latent targets do not change that.

### 2c. Mechanism (P2): latent pretext skill vs downstream gain

`scripts/diagnose_pretext_latent.py`: predictor error in the training loss's own normalised
latent space vs three trivial predictors — constant (per-sequence mean; MSE ≈ 1 by construction),
persistence (copy z_j) and per-lag AR-shrinkage (ρ_k z_j) — **skill = 1 − model / best trivial**.

| dataset (arch) | skill | frozen gain latent − random | fine-tuned gain |
|---|---:|---:|---:|
| Sleep DE (3 seeds) | −0.16 … −0.21 | +3.2 | +0.6 |
| Seizure DE | −0.59 | +3.9 | −0.5 |
| Sleep tf64 | −0.01 | +5.5 | +0.3 |
| Sleep raw tokens (structured / per-electrode) | **+0.36 / +0.37** | +0.8 / +2.2 | +0.4 / +0.9 |
| Emotion un-smoothed | **+0.62** | **−5.3** (36.7 vs 42.0) | — |
| Emotion smoothed | +0.39 | −10.2 (45.7 vs 56.0) | — |
| Motor imagery | +0.59 | −4.2 (39.4 vs 43.6) | — |

On the long, smooth per-epoch corpora the latent objective **degenerates**: the EMA target
becomes temporally smooth and the predictor never beats copy-last / AR-shrinkage (skill ≤ 0);
without time-normalised targets it collapses outright (cosine variant: persistence distance
0.002). Where the pretext *is* learned (short trials: emotion, MI; raw tokens mid-training) the
frozen gain is negative. So the sign of the pretext-skill ↔ transfer relation is unchanged from
EXP-0017 §4e — **latent targets do not realign the objective**, and under fine-tuning every arm
converges to the random-init level anyway.

---

## 3. Gate 2 — raw-EEG structured tokens · [[EXP-0022]]

**Setup.** `physiofm/raw_eeg.py`, `scripts/build_raw_dataset.py`: 0.3 Hz high-pass, token = all
channels × 200 ms (sleep: 2 ch × 20 samples @100 Hz = 40-d; 150 tokens per 30 s epoch; 29.3 M
tokens; label companions asserted identical to DE). Per-electrode variant: each channel its own
sequence of 1 × 20 tokens (58.6 M tokens); evaluators average the two channel sequences per epoch
(`--merge_every 2`). Same PhysioFM-S (d = 256, 6 layers) with n_cb = 40 / 20; sequences chunked
to 20 epochs = 3000 tokens; per-epoch readout = mean of the epoch's 150 token states. Pretraining
10 epochs (H100 80 GB, batch 32 / 16, fp32), fine-tuning 3 epochs, subject-disjoint 5-fold, seed
42 (single pretraining seed — compute-bound).

| tokens | arm | frozen probe | fine-tuned (3 ep) |
|---|---|---:|---:|
| structured 2 ch × 200 ms | PhysioFM (pretrained, input-space) | **70.84** (κ .615) | **75.46** (κ .676) |
| structured | PhysioFM (latent) | 56.90 | 74.65 |
| structured | PhysioFM (no pretrain) | 56.15 | 74.23 |
| per-electrode 1 × 200 ms | PhysioFM (pretrained, input-space) | **72.68** (κ .638) | **76.36** (κ .684) |
| per-electrode | PhysioFM (latent) | 62.81 | 75.45 |
| per-electrode | PhysioFM (no pretrain) | 60.65 | 74.56 |
| *reference: DE pipeline* | pretrained / no pretrain | 72.62 / 62.86 | 75.37 / 73.18 |
| *reference: tf64 pipeline* | pretrained / no pretrain | 76.75 / 65.51 | 77.88 / 77.03 |

Readings:
- **P1 (escape the DE bottleneck)** — not with this budget: raw tokens reach parity with the DE
  pipeline (75.5 vs 75.4 fine-tuned) and stay 2.4 below tf64. Ten pretraining epochs and three
  fine-tuning epochs over 29 M tokens is a small budget for raw EEG; a longer schedule, bf16 and
  a larger model are the obvious next levers, but the comparison at equal recipe is what it is.
- **P2 (pretraining finally matters on raw)** — the frozen gap is the largest in the project
  (**+14.7**: a random-init causal transformer over raw tokens is a poor frozen feature extractor),
  and it collapses to **+1.2** under fine-tuning. Frozen-probe inflation is *worse* on raw EEG,
  not better. Latent targets: +0.4.
- **P3 (structured > per-electrode)** — **not supported on sleep**: per-electrode decomposition
  is ≥ structured on every arm (+1.8 frozen, +0.9 fine-tuned for input-PC). With only two
  channels there is almost no spatial structure to exploit and the per-electrode model sees twice
  as many sequences; the ablation the plan wanted needs a many-channel corpus (CHB-MIT raw was
  not run — 43 GB / 16 B token values, out of this budget). Interestingly the raw pretext is
  slightly *easier* per electrode (best MSE 0.788 vs 0.802).

---

## 4. Gate 3 — streaming / causal evaluation · [[EXP-0023]]

**Setup.** `scripts/gate3_streaming_eval.py`, sleep DE, same 5 folds and fine-tuning recipe.
Bidirectional twin = identical stack with full attention (`PhysioFMS(causal=False)`; a code-review
found and fixed an SDPA fallback that would have made it silently causal on unpadded batches —
verified: alone-vs-padded identical to 3e-6, future perturbations change position 0). Offline =
whole 400-epoch chunk visible; online = only epochs ≤ t visible when scoring epoch t (the
bidirectional model is re-run on every prefix; the causal model's offline pass already satisfies
the constraint — checked: identical numbers).

| arm | attention | offline acc / κ | **online acc / κ** | tokens per decision |
|---|---|---:|---:|---:|
| PhysioFM (no pretrain) | causal | 73.18 / .645 | **73.18 / .645** | 1 (KV cache) — 190 without |
| PhysioFM (no pretrain), twin | bidirectional | **74.10 / .655** | 70.40 / .607 | 190 |
| PhysioFM (latent) | causal | 74.12 / .656 | 74.12 / .656 | 1 |
| PhysioFM (pretrained, input-space) | causal | 75.37 / .672 | **75.37 / .672** | 1 |

Offline the bidirectional twin is +0.9 better (it sees the future); under a streaming
constraint the causal decoder wins by **+2.8** (random-init) and **+5.0** (pretrained) at 1/190
of the compute per decision. This is the plan's "defensible claim", and it holds.

---

## 5. What this means for the plan

1. **The two fixes were the right hypotheses to test and both are now answered.** R1 (spectral
   resolution) helps the architecture on sleep (+2.3 fine-tuned, +4 for random-init) but leaves
   pretraining ≈ +0.85; R2 (latent targets) does not beat input-space PC anywhere and degenerates
   on exactly the long smooth corpora where predictive coding was supposed to shine.
2. **Raw EEG at this scale does not change the story**: parity with DE, below tf64, and the
   largest frozen-vs-fine-tuned inflation yet. BrainGPT/LaBraM-scale pretraining (10³–10⁴× the
   tokens, 100× the parameters) is a different regime we did not enter.
3. **What survives, and is stronger than before:** the structured causal transformer as an
   EEG encoder — **77.9 % / κ 0.71 on Sleep-EDF-78 with tf64 tokens** (was 75.4 with DE), plus
   a measured streaming advantage of the causal design (+2.8 to +5.0 online), plus the
   methodological result that frozen-probe evaluation inflates SSL gains by 3–12× on every
   representation tried (DE +9.8→+2.5, tf64 +11.2→+0.85, raw +14.7→+1.2).
4. **What to write.** Not "predictive-coding pretraining works with the right input/target" —
   it does not, in this regime. Rather: a compact structured causal EEG transformer with the
   spectral-token front end, a streaming-evaluation result, and a rigorous negative on
   forecasting-SSL with the diagnostics that explain it.

Caveats: raw-token and per-electrode results are single-pretraining-seed with a short recipe;
seizure raw not run; the per-electrode ablation is weak on a 2-channel montage; SOTA numbers are
the unverified July web figures.
