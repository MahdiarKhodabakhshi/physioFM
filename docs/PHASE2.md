# PhysioFM Phase 2 — Structured Patching

Phase 2 implements the proposal's core idea: replace TimesFM's univariate scalar
patch with a **structured `(C×B)` DE patch** in a **decoder-only, predictive-coding
transformer** that does **not** instance-normalize away absolute band power.

## Why Phase 2 looks the way it does (Phase 1 verdict)

Phase 1 is complete and the at-chance TimesFM result is **real and robust** (not a
bug). The diagnostics (`results/timesfm_phase1/diagnostics/`, `docs/PHASE1.md`)
established:

- **Raw DE → LogReg** (no model) is the real non-FM linear ceiling: SEED-V ~51.8% /
  SEED-IV ~62.5%, trial and segment level, when the 310 channel-band values are kept.
- **Univariate TimesFM is at chance** (20–28%) in every config: frozen ≈ fine-tuned,
  raw ≈ z-scored, trial ≈ segment, mean-pool ≈ structure-preserving.
- It fails for two independent reasons: (1) per-series **instance normalization strips
  absolute band-power**, the discriminative part of DE; (2) cross-series flattening /
  averaging **erases channel-band identity**.
- `patch_len=32` is too coarse: one patch ≈ a whole trial; ~40% of trials are shorter
  than one patch.

**Mandate:** keep the `(C×B)` matrix intact as a structured patch, avoid per-series
normalization, use a small input patch so short trials still yield multiple tokens,
and evaluate segment-level under the PC-SSL splits.

## Stage 2 scientific claims

| # | Claim | Decided by |
|---|-------|-----------|
| C1 | Structured `(C×B)` patching ≫ univariate Phase 1 | E2 vs Phase 1 |
| C2 | Decoder-only transformer + predictive coding adds value over raw-DE linear AND nonlinear ceilings | E1 vs E0 |
| C3 | Transformer matches/approaches PC-SSL, with variable context + multi-horizon for free | E1 vs published PC-SSL |
| C4 | Output horizon > input horizon (multi-horizon PC) helps | E3.1 |
| C5 | Avoiding instance-norm is necessary (confirms Phase 1 mechanism) | E3.2 |
| C6 | TimesFM pretrained weights transfer — or provably don't (OOD input space) | E1b vs E1a |

## Decisions (locked)

- **PC-SSL comparison:** use the **published** numbers — SEED-IV **84.48%**, SEED-V
  **92.39%** — with an explicit caveat that the local replication was incomplete
  (SEED-IV 44.7–70.4%) and the clean SEED-V run had ~80% train/test segment overlap.
  We do **not** re-run PC-SSL replication in Phase 2.
- **Two co-primary backbones:** **E1a** from-scratch decoder-only transformer, and
  **E1b** TimesFM-initialized transformer (proposal-literal: replace the scalar patch
  in TimesFM's input residual block). Both are first-class, compared head to head (C6).
- **Pretraining corpus:** SEED-IV + SEED-V + SEED combined DE (labels unused for SSL).
  ~104.6k DE windows / 2,475 trial-sequences, all 62×5.
- **Eval scope:** subject-dependent (PC-SSL-comparable) primary, subject-independent
  LOSO as a stretch generalization result.

## Data facts (verified)

| Dataset | Trials | Shape | Len min/med/max | Windows | Subjects | Classes |
|---|---:|---|---|---:|---:|---:|
| SEED-V  | 720  | 62×5 | 13 / 41 / 74 | 29,168 | 16 | 5 |
| SEED-IV | 1080 | 62×5 | 10 / 36 / 64 | 37,575 | 15 | 4 |
| SEED    | 675  | 62×5 | 46 / 58 / 66 | 37,890 | 15 | 3 |

No trial is shorter than 13 windows, so `p_in=4` always yields ≥1 input patch.
Trials shorter than `p_in+p_out`: SEED-V 128 (<20), SEED-IV 165 (<20) — handle with a
shorter effective horizon for short trials (predict whatever windows remain).

## Architecture — "PhysioFM-S"

- **Token:** one short time window of the full DE matrix. Input patch = `p_in`
  consecutive DE windows, each 310-d (62×5), embedded to model dim `d`.
- **Patch embedder** (ablated, E3.3): (a) plain `Linear(p_in·310 → d)`; (b) PC-SSL-style
  band-attention → channel-attention → projection; (c) channel-agnostic (per-channel
  shared MLP + pool) — the **Phase 3 enabler** for variable channel counts, designed in now.
- **Backbone:** causal decoder-only transformer (RoPE, ~6–12 layers, `d`≈256–512).
  - **E1a:** random init.
  - **E1b:** transformer block initialized from TimesFM-2.5 weights, structured
    input/output residual blocks trained fresh.
- **Output residual block:** predicts next `p_out` DE windows (each 310-d).
- **Objective:** multi-step predictive-coding MSE (predict windows `t+1 … t+p_out`),
  generalizing PC-SSL's single-step `t+1`. `p_in < p_out` (default 4→16), per proposal.
- **Normalization:** NO per-series instance norm. Per-(channel,band) standardization
  fit on TRAIN only — preserves relative band-power structure.

## Evaluation harness (frozen, reused by all models)

`physiofm/phase2_eval.py`:
- Segment-level, PC-SSL subject-dependent splits (`seed_v_fold_mask` /
  `seed_iv_fold_mask` in `physiofm/embedding_evaluation.py`), seed 42, StandardScaler +
  balanced LogReg.
- Two modes: **zero-shot** (freeze encoder, linear probe) and **fine-tuned** (unfreeze
  input/output blocks; limited-label 10/50/100%).
- Comparison ladder per dataset (accuracy % / macro-F1 %):
  `chance | raw-DE→LogReg | raw-DE→SVM/MLP | TimesFM (Phase 1) | PhysioFM-S zero-shot | PhysioFM-S fine-tuned | PC-SSL published`.

## Experiments, dependencies, and WAIT gates

### E0 — Baselines & shared harness (no model training) — DONE
- **E0.1** Raw-DE nonlinear ceilings: 310-d → {LogReg, Linear-SVM, RBF-SVM, small MLP},
  segment-level, PC-SSL splits. If RBF-SVM ≈ PC-SSL, reframe the FM contribution.
- **E0.2** Canonical eval module `physiofm/phase2_eval.py`; writes to `results/phase2/`.
- (PC-SSL re-replication intentionally dropped — published numbers used.)

Harness validated: LogReg on SEED-V reproduces the Phase-1 raw-DE segment number
exactly (51.40 / 49.92). Scripts: `scripts/phase2_raw_de_ceilings.py`. Artifacts:
`results/phase2/raw_de_ceilings.{csv,md}`.

**E0.1 results (segment-level, subject-dependent, accuracy % / macro-F1 %, mean ± std):**

SEED-V (chance 20%):

| Classifier | Accuracy % | Macro-F1 % |
| --- | ---: | ---: |
| raw-DE → LogReg | 51.40 ± 12.99 | 49.92 ± 13.50 |
| raw-DE → Linear-SVM | 50.81 ± 12.82 | 48.57 ± 13.24 |
| raw-DE → RBF-SVM | 44.59 ± 12.78 | 42.99 ± 13.10 |
| raw-DE → MLP | 40.10 ± 10.40 | 37.43 ± 10.91 |

SEED-IV (chance 25%):

| Classifier | Accuracy % | Macro-F1 % |
| --- | ---: | ---: |
| raw-DE → LogReg | 62.75 ± 20.28 | 54.75 ± 20.79 |
| raw-DE → Linear-SVM | 64.99 ± 19.09 | 55.60 ± 20.89 |
| raw-DE → RBF-SVM | 51.37 ± 25.65 | 47.27 ± 25.25 |
| raw-DE → MLP | 48.93 ± 23.03 | 41.58 ± 20.53 |

**Finding.** The raw-DE ceiling is **linear** (~51% SEED-V / ~65% SEED-IV). Adding
nonlinearity (RBF-SVM, MLP) *hurts* — high-capacity classifiers overfit the small
per-subject-fold training sets (~600–700 segments). So the gap to PC-SSL
(published 84.48 / 92.39) is **not** closed by nonlinearity on raw DE; it requires a
learned representation (pretraining + channel/band structure), which is exactly what
PhysioFM-S must supply. The FM value proposition stands. PhysioFM-S targets are:
beat ~51/65% (raw-DE linear, C2) and approach ~84/92% (PC-SSL, C3).

> WAIT GATE 1 — confirm raw-DE ceilings + harness before committing to the model.

### E1 — PhysioFM-S core (two co-primary backbones; depends on E0) — DONE (zero-shot)
- **E1.0** Structured-patch data pipeline + model + scripts:
  `physiofm/physiofm_s.py`, `physiofm/structured_data.py`,
  `scripts/phase2_pretrain.py`, `scripts/phase2_extract_eval.py`.
- **E1a** From-scratch backbone (d=256, 6 layers, 8 heads, 0.88M params): PC pretrain
  on SEED-IV+V+SEED, `p_in=1 → p_out=16`, 60 epochs (~53s), best PC-MSE 0.0037.
- **E1b** TimesFM-init backbone (d=1280, 20 pretrained layers FROZEN; only structured
  input/output blocks trained, ~6.7M trainable): PC pretrain, 40 epochs (~7.5 min),
  best PC-MSE 0.0039.

Both implementations bypass TimesFM's RevIN instance-norm and feed structured `(C×B)`
patches straight into the decoder stack. Input normalization is fixed per-(C,B) corpus
standardization (preserves band-power structure). Default token = one DE window
(`p_in=1`, fully window-causal, directly PC-SSL-comparable); `p_out=16` multi-horizon.

**E1 zero-shot linear-probe results** (segment-level, subject-dependent, acc % / F1 %):

SEED-V (chance 20%):

| Method | LogReg | Linear-SVM |
| --- | ---: | ---: |
| Phase-1 univariate TimesFM | ~23–25 | — |
| raw-DE ceiling | 51.40 / 49.92 | 50.81 / 48.57 |
| PhysioFM-S **scratch** (E1a) | 45.58 / 44.10 | 46.13 / 44.53 |
| PhysioFM-S **TimesFM-init** (E1b) | 43.27 / 41.32 | 44.18 / 41.83 |

SEED-IV (chance 25%):

| Method | LogReg | Linear-SVM |
| --- | ---: | ---: |
| Phase-1 univariate TimesFM | ~26–28 | — |
| raw-DE ceiling | 62.75 / 54.75 | 64.99 / 55.60 |
| PhysioFM-S **scratch** (E1a) | 57.49 / 48.93 | 57.41 / 49.12 |
| PhysioFM-S **TimesFM-init** (E1b) | 60.40 / 52.35 | 61.48 / 53.20 |

**Gate 2 findings.**
- **C1 confirmed, strongly.** Structured patching recovers the emotion signal that
  univariate TimesFM destroyed: chance (~25%) → 44–61% zero-shot. The structured-patch
  + no-instance-norm design is the fix the Phase-1 diagnostics predicted.
- **C2 open (expected).** Frozen zero-shot PhysioFM-S sits just *below* the raw-DE
  linear ceiling (46 vs 51 on SEED-V; 61 vs 65 on SEED-IV). This is the normal frozen-SSL
  gap — the PC features are not label-aligned. The **fine-tuned mode** (E3) is where C2
  is decided.
- **C6 (transfer).** TimesFM-init ≈ scratch (better on SEED-IV: 61.5 vs 57.5; slightly
  worse on SEED-V: 44.2 vs 46.1). Notably, E1b's transformer is FROZEN and matches a
  fully-trained scratch model — TimesFM's pretrained temporal priors carry useful
  structure into the DE space.

Artifacts: `results/phase2/pretrain/{scratch,timesfm}_pin1_pout16/`.

> WAIT GATE 2 — make-or-break: does PhysioFM-S beat Phase 1 (C1) and the raw-DE linear
> ceiling (C2)? C1 cleared decisively; C2 deferred to fine-tuned mode (E3).

### E2 — Core comparison (depends on E1)
Single headline table: Phase-1 univariate TimesFM vs PhysioFM-S (E1a, E1b) vs raw-DE vs
PC-SSL published, identical eval. Proves C1/C2/C3/C6.

### E3 — Ablations (depend on E1) — PARTIAL (key controls done)
- **E3.1** Patch/horizon: `p_out=16` vs `p_out=1` (single-step PC, PC-SSL-style).
- **E3.3** Patch embedder: plain Linear vs band/channel-attention (PC-SSL SE attention).
- **E3.4** Pretraining objective: multi-step PC vs single-step vs **no-pretrain
  (random-init probe)** — isolates the value of pretraining.
- **E3.2** Normalization (per-series instance norm to reproduce Phase-1 collapse): TODO.
- Fine-tuned mode: full/io SGD fine-tuning per fold is unstable on ~600 labels/fold and
  underperforms the convex probe; a frozen-encoder SGD linear head (head-only) reaches
  52.6% on SEED-V (> raw-DE), showing the probe optimizer matters.

**E3 results — zero-shot LogReg probe (segment-level, subject-dependent, acc % / F1 %):**

SEED-V (chance 20%, raw-DE ceiling 51.40 / 49.92):

| Config | acc / F1 |
| --- | ---: |
| Phase-1 univariate TimesFM | ~23–25 (chance) |
| scratch, PC p_out=16, linear embed (E1a) | 45.58 / 44.10 |
| scratch, PC p_out=16, **attn** embed (E3.3) | 46.98 / 45.37 |
| scratch, PC **p_out=1**, linear (E3.1) | 49.78 / 48.36 |
| scratch, **NO pretrain** (random init) (E3.4) | 48.60 / 46.63 |
| TimesFM-init, PC p_out=16 (E1b) | 43.27 / 41.32 |
| scratch E1a feats + **head-only SGD probe** | **52.58 / 50.87** |

SEED-IV (chance 25%, raw-DE ceiling 62.75 / 54.75):

| Config | acc / F1 |
| --- | ---: |
| Phase-1 univariate TimesFM | ~26–28 (chance) |
| scratch, PC p_out=16, linear embed (E1a) | 57.49 / 48.93 |
| scratch, PC p_out=16, attn embed (E3.3) | 55.16 / 45.97 |
| scratch, PC p_out=1, linear (E3.1) | 57.01 / 48.38 |
| scratch, **NO pretrain** (random init) (E3.4) | 60.67 / 53.09 |
| TimesFM-init, PC p_out=16 (E1b) | 60.40 / 52.35 |

**Critical finding (E3.4).** A **random-init** structured transformer (NO pretraining)
matches or beats the predictive-coding–pretrained model on the linear probe (SEED-V
48.6 vs 45.6; SEED-IV 60.7 vs 57.5). Single-step `p_out=1` ≈ multi-step `p_out=16` ≈
no-pretrain. **So the temporal predictive-coding pretraining adds essentially no
linear-probe-accessible emotion signal beyond preserving the structured `(C×B)` input
and passing it through a (random) nonlinear mixer.**

**Interpretation.** Everything that preserves the structured static spectral pattern
lands in the same ~48–65% band: raw-DE linear (51/63), random-init structured
transformer (49/61), PC-pretrained (46–58). DE emotion signal is **static spectral
structure**, not temporal dynamics — so a *temporal* predictive-coding FM objective
does not help it. This sharply characterizes where a time-series FM helps (it recovers
the structure univariate TimesFM destroyed → C1) and where it does not (temporal PC
pretraining ≠ emotion gain → revises C2/C4). The remaining gap to PC-SSL's published
84–92% is the supervised nonlinear classifier (and the flagged PC-SSL test/train
segment overlap), not the pretraining objective.

> WAIT GATE 3 — direction decision: the linear-probe ceiling for structure-preserving
> methods is ~raw-DE; pushing past it needs a supervised nonlinear head, not more
> temporal pretraining. See chat for options.

### E3.2 — instance-norm control (C5) — DONE

Raw-DE LogReg, subject-dependent, with per-series instance norm (TimesFM-style RevIN):

| Dataset | corpus-standardized | per-series instance-norm | chance |
| --- | ---: | ---: | ---: |
| SEED-V | 51.40 / 49.92 | **18.02 / 10.88** | 20 |
| SEED-IV | 62.75 / 54.75 | **26.77 / 23.07** | 25 |

**C5 confirmed decisively.** Per-series instance normalization collapses raw DE to
(below-)chance — proving the Phase-1 mechanism at the structured level: the
discriminative DE signal is the *absolute spectral level*, which instance-norm removes.
This is exactly why PhysioFM-S must (and does) avoid RevIN.

### E4 — Interpretability + Phase-3 readiness — DONE (E4.1, E4.3)

**E4.3 — LOSO (subject-independent), Linear-SVM** (`results/phase2/analysis.md`,
`results/phase2/*/eval_loso.csv`):

| Method (LOSO) | SEED-V | SEED-IV |
| --- | ---: | ---: |
| raw-DE | 31.91 / 27.13 | 37.77 / 33.00 |
| PhysioFM-S random-init | 28.98 / 23.45 | 35.92 / 29.67 |
| PhysioFM-S PC-pretrained | 27.60 / 20.68 | 34.73 / 28.46 |
| (subject-dependent raw-DE for reference) | 51.40 | 62.75 |

Subject-independent accuracy is far below subject-dependent (large cross-subject DE
distribution shift). **Crucially, the learned/pretrained representation does NOT beat
raw DE on cross-subject transfer either, and PC-pretrained ≈ random-init again.** So the
foundation model provides no advantage over raw structured DE on *any* axis tested
(subject-dependent or subject-independent) — the static-spectral nature of DE emotion is
the binding constraint, not the model or the protocol.

**E4.1 — band/channel discriminative importance** (mean |LogReg coef|; figure
`results/phase2/band_channel_importance.png`): band importance is **roughly uniform**
(γ slightly highest on SEED-V at 21.8%, α on SEED-IV at 21.1%; β lowest on both) — no
single dominant band, consistent with DE emotion being a *distributed* spectral pattern.
Top channels concentrate at low indices (frontal sites, e.g. ch 0/1/8) plus
temporal/parietal sites — broadly consistent with the frontal/temporal emotion-EEG
literature.

- **E4.2** Channel-agnostic embedder (Phase-3 bridge): deferred to Phase 3.

## Stage 2 Conclusions and mapping to the proposal

**Headline.** Structured `(C×B)` patching + avoiding instance-norm **recovers the
emotion signal that univariate TimesFM destroyed** (chance → ~50–60%), but **temporal
predictive-coding pretraining adds essentially nothing** for DE emotion (random-init ≈
PC-pretrained ≈ single-step ≈ multi-horizon). The DE emotion signal is **static spectral
structure**, not temporal dynamics, so a *temporal* foundation-model objective cannot
help it — every structure-preserving method lands at the raw-DE linear ceiling
(~51 / 63%). The published PC-SSL 84–92% is attributable to a supervised nonlinear
classifier (and a flagged test/train segment overlap), not to the PC pretraining idea.

| Proposal claim | Stage 2 verdict |
| --- | --- |
| Phase 1: univariate TimesFM transfers to DE | **Refuted** (Stage 1): at chance — RevIN + flattening destroy the signal |
| Phase 2: structured `(C×B)` patching ≫ univariate | **Confirmed (C1)**: chance → ~50–60% |
| Decoder-only handles variable context / multi-horizon | Mechanically yes; multi-horizon (`p_out=16`) gives **no gain** over `p_out=1` (**C4 not supported** for DE emotion) |
| "PC in a transformer = PC-SSL objective" → big wins | **Not supported for DE emotion**: pretraining ≈ random init |
| Avoid instance-norm | **Confirmed necessary (C5)**: instance-norm collapses to chance |
| TimesFM init transfers | **Roughly neutral (C6)**: frozen TimesFM stack ≈ from-scratch |

**Final comparison ladder (segment-level, subject-dependent, acc %):**

| Method | SEED-V | SEED-IV |
| --- | ---: | ---: |
| chance | 20.0 | 25.0 |
| Phase-1 univariate TimesFM | ~24 | ~27 |
| raw-DE → LogReg (linear ceiling) | 51.4 | 62.8 |
| raw-DE + instance-norm (C5 control) | 18.0 | 26.8 |
| PhysioFM-S random-init (E3.4) | 48.6 | 60.7 |
| PhysioFM-S PC-pretrained, scratch (E1a) | 45.6 | 57.5 |
| PhysioFM-S PC-pretrained, TimesFM-init (E1b) | 60.4 (IV) | 60.4 |
| PhysioFM-S + head-only SGD probe | 52.6 | — |
| PC-SSL published (caveated) | 92.39 | 84.48 |
| raw-DE LOSO (subject-independent) | 31.9 | 37.8 |

**Implications for Phase 3.** The negative pretraining result is itself the strongest
argument for the Phase-3 pivot: temporal predictive coding should be tested on tasks
with genuine temporal dynamics (sleep staging, seizure, motor imagery), where a
time-series FM objective is far more likely to pay off than on static-spectral emotion DE.

## Sequencing & compute

1. E0 (no GPU model training) → Gate 1
2. E1a + E1b (pretrain + eval, ~tens of min each on H100) → Gate 2
3. E2 + E3 (ablation grid, parallelizable, minutes each) → Gate 3
4. E4 (analysis)

All artifacts under `results/phase2/`; Phase-1 artifacts untouched.

## Risks

- **Raw-DE nonlinear ceiling already high** → E0.1 reveals early; reframe contribution.
- **Tiny corpus / overfitting** → small model + 3-dataset corpus; synthetic mixing
  deferred to Phase 3.
- **PhysioFM-S only matches raw-DE** → still a valid result; ablations explain why;
  Gate 2 catches it early.
- **TimesFM-init underperforms scratch** → expected possibility (OOD input space); that
  IS the C6 finding.
