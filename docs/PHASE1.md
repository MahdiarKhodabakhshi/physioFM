# PhysioFM Phase 1

Phase 1 follows the proposal:

1. Replicate PC-SSL on SEED-IV and SEED-V.
2. Set up TimesFM.
3. Convert EEG datasets into a common DE representation.
4. Fine-tune TimesFM on univariate DE series.
5. Evaluate TimesFM embeddings on SEED-IV/V emotion classification.

## TimesFM Setup

Phase 1 uses the HuggingFace/Transformers TimesFM checkpoint directly:

```text
google/timesfm-2.5-200m-transformers
```

Run the scripts from a Python environment with `torch`, `transformers`, `peft`,
`numpy`, `scipy`, and `scikit-learn` installed:

```bash
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__)"
```

The local `timesfm/` clone is not required by the PhysioFM scripts and should
stay ignored unless it is intentionally used as a reference checkout.

## Build Canonical DE Archives

Run from the workspace root:

```bash
python scripts/build_de_dataset.py --dataset seed_v
python scripts/build_de_dataset.py --dataset seed_iv
python scripts/build_de_dataset.py --dataset seed
```

Outputs:

```text
data/physiofm/de_features/seed_v_de_LDS.npz
data/physiofm/de_features/seed_iv_de_LDS.npz
data/physiofm/de_features/seed_de_LDS.npz
```

Each archive stores variable-length trials as `time x channels x bands`.

Built local archive summary:

| Dataset | Trials | Subjects | Labels | Length range | Univariate series |
| --- | ---: | ---: | --- | --- | ---: |
| SEED-V | 720 | 16 | 0, 1, 2, 3, 4 | 13-74 | 223200 |
| SEED-IV | 1080 | 15 | 0, 1, 2, 3 | 10-64 | 334800 |
| SEED | 675 | 15 | -1, 0, 1 | 46-66 | 209250 |

## TimesFM Phase 1 Commands

Small smoke fine-tune:

```bash
python scripts/finetune_timesfm_de.py \
  --archive data/physiofm/de_features/seed_v_de_LDS.npz \
  --archive data/physiofm/de_features/seed_iv_de_LDS.npz \
  --output_dir results/phase1_timesfm_de/seed_iv_v_lora_smoke \
  --epochs 1 \
  --num_samples 128 \
  --max_series 2048 \
  --batch_size 8
```

Completed smoke outputs:

```text
results/phase1_timesfm_de/seed_iv_v_lora_smoke/
results/phase1_timesfm_de/seed_v_subject1_embeddings_smoke.npz
results/phase1_timesfm_de/seed_iv_subject1_embeddings_smoke.npz
results/phase1_timesfm_de/seed_v_subject1_trial_classification_smoke.csv
results/phase1_timesfm_de/seed_iv_subject1_trial_classification_smoke.csv
```

Smoke classification is intentionally not a result number; it uses one subject
and only 8 random fine-tuning windows to validate code paths. It produced:

| Dataset smoke | Runs | Mean accuracy | Mean macro F1 |
| --- | ---: | ---: | ---: |
| SEED-V subject 1 | 3 | 15.56% | 14.92% |
| SEED-IV subject 1 | 3 | 20.83% | 20.56% |

Full Phase 1 fine-tune:

```bash
python scripts/finetune_timesfm_de.py \
  --archive data/physiofm/de_features/seed_v_de_LDS.npz \
  --archive data/physiofm/de_features/seed_iv_de_LDS.npz \
  --output_dir results/phase1_timesfm_de/seed_iv_v_lora \
  --epochs 10 \
  --num_samples 50000 \
  --batch_size 32
```

Completed one-epoch pilot:

```text
results/phase1_timesfm_de/seed_iv_v_lora_e1/
results/phase1_timesfm_de/seed_v_embeddings_e1.npz
results/phase1_timesfm_de/seed_iv_embeddings_e1.npz
results/phase1_timesfm_de/seed_v_trial_classification_e1.csv
results/phase1_timesfm_de/seed_iv_trial_classification_e1.csv
```

The one-epoch pilot used 2048 random windows from SEED-IV/V, then embedded every
trial and ran the subject-dependent trial-level classifier:

| Dataset | Runs | Embeddings | Mean accuracy | Mean macro F1 |
| --- | ---: | ---: | ---: | ---: |
| SEED-V | 48 | 720 x 1280 | 22.85% | 21.27% |
| SEED-IV | 45 | 1080 x 1280 | 27.22% | 25.69% |

These are pilot numbers, not final Phase 1 results. The classifier is currently
trial-level because Phase 1 aggregates TimesFM channel-band embeddings per
trial. It is therefore not directly comparable to PC-SSL's segment-level paper
accuracies.

## Final Phase 1 Results

The full fine-tune in the "Full Phase 1 fine-tune" recipe above was completed
(LoRA r=4, all-linear targets, 10 epochs, 50,000 random `context_len=32,
horizon_len=1` windows drawn from 328,910 valid SEED-IV+SEED-V univariate DE
series, batch size 32, lr 1e-4). Train MSE decreased monotonically 0.0232 →
0.0179 across the 10 epochs. The fine-tuned adapter is at
`results/phase1_timesfm_de/seed_iv_v_lora/`.

Every trial was then embedded with the fine-tuned adapter (last hidden state,
mean over patches per univariate series, then mean over all 310 channel-band
series → one 1280-dim vector per trial) and classified with the
subject-dependent protocol (SEED-V 3-fold MSLTE split, SEED-IV GMSS 16/8 split),
standardized logistic regression, class-balanced.

| Dataset | Trials × dim | Runs | Accuracy | Macro F1 | Chance |
| --- | --- | ---: | ---: | ---: | ---: |
| SEED-V | 720 × 1280 | 48 | 20.49% | 18.78% | ~20% (5-class) |
| SEED-IV | 1080 × 1280 | 45 | 23.33% | 20.75% | 25% (4-class) |

Artifacts:

```text
results/phase1_timesfm_de/seed_iv_v_lora/            # fine-tuned LoRA adapter
results/phase1_timesfm_de/seed_v_embeddings.npz      # 720 x 1280
results/phase1_timesfm_de/seed_iv_embeddings.npz     # 1080 x 1280
results/phase1_timesfm_de/seed_v_trial_classification.csv
results/phase1_timesfm_de/seed_iv_trial_classification.csv
results/phase1_timesfm_de/logs/finetune_full.log
results/phase1_timesfm_de/logs/embed_classify.log
```

> **Superseded interpretation (kept for the record).** An earlier draft called
> the at-chance number "the expected Phase 1 baseline." The diagnostics below
> (STEP 1–4) show that explanation was only half right, so the corrected
> interpretation is in **Corrected Evaluation** below. The trial-level numbers
> above are real and reproducible, but they are not the whole story.

## Corrected Evaluation (Diagnostics STEP 1–4)

The at-chance result was treated as **possibly a bug**, not a confirmed
baseline, and investigated. All checks use fixed seeds (42), the identical
subject-dependent splits, and the identical classifier. Full report and
artifacts: `results/phase1_timesfm_de/diagnostics/REPORT.md` and
`results/phase1_timesfm_de/diagnostics/segment/`.

**Key controls.**

- **Raw DE → same classifier/splits, no TimesFM** is far above chance when the
  310 channel-band values are kept (SEED-V 51.8%, SEED-IV 62.5% trial-level;
  51.4% / 62.8% segment-level). So the trial-level protocol and the
  ~15-train-trials/fold are **not** the cause of the chance result.
- The same raw DE collapses to ~chance (SEED-V 26.3%, SEED-IV 33.9%) the moment
  it is mean-pooled across the 310 series — i.e. **the cross-channel/band
  averaging is what destroys the signal.**
- The fine-tuned trial embedding is near-degenerate: 79% (SEED-V) / 68%
  (SEED-IV) of total variance lies in a single dimension; cosine-1NN is at
  chance.
- **frozen ≈ fine-tuned**, and **raw input ≈ z-scored input** — the LoRA adapter
  and input normalization are both irrelevant. TimesFM applies its own per-series
  instance normalization internally, which removes the absolute band-power level.
- Moving to **segment-level with channel-band structure preserved** (per-window
  TimesFM context embedding, random-projected and concatenated over the 310
  channel-bands) is **still at chance** (24.7–26.6%), far below raw DE.

### STEP 4 — Final comparison (accuracy % / macro-F1 %)

SEED-V (chance ≈ 20%):

| Method | Trial-level | Segment-level |
| --- | --- | --- |
| Raw DE → LogReg (control) | 51.81 / 50.29 | 51.40 / 49.92 |
| TimesFM frozen | 23.19 / 21.85 | 25.04 / 24.22 |
| TimesFM fine-tuned | 20.49 / 18.78 | 26.62 / 25.27 |
| PC-SSL replicated (local) | — | 89.71 / 89.34 |
| PC-SSL published | — | 92.39 / — |

SEED-IV (chance = 25%):

| Method | Trial-level | Segment-level |
| --- | --- | --- |
| Raw DE → LogReg (control) | 62.50 / 54.89 | 62.75 / 54.75 |
| TimesFM frozen | 27.78 / 25.31 | 26.37 / 24.53 |
| TimesFM fine-tuned | 23.33 / 20.75 | 24.73 / 22.90 |
| PC-SSL replicated (local) | — | 44.72–70.41 (incomplete) |
| PC-SSL published | — | 84.48 / — |

(TimesFM segment-level = structure-preserving `310×8` feature; per-series
mean-pooled `310` features are even lower, ~20–22%. PC-SSL segment-level
SEED-IV replication is incomplete — the paper-protocol run reached only 44.7%
and an alternative config 70.4% vs the published 84.48%; SEED-V replicates
cleanly. PC-SSL's local SEED-V run also carries an ~80% test/train segment
overlap flagged in its own `final_metrics.json`, so the ~90% should be read with
that caveat.)

**Corrected interpretation.** The chance result is **real and robust**, but the
mechanism is sharper than "trial-level pooling." Two things destroy the signal,
both rooted in TimesFM being a *univariate* forecaster:

1. **TimesFM strips absolute band-power.** Emotion-discriminative DE information
   lives in the absolute spectral-power level of each channel-band. TimesFM
   normalizes every series internally (RevIN-style), so its embedding encodes
   temporal *shape*, not level — which is why z-scoring the input changes
   nothing and why even a structure-preserving segment-level feature stays at
   chance.
2. **Cross-series averaging removes channel-band identity.** Independently of
   (1), mean-pooling the 310 series collapses them onto one global component
   (proven by the raw-DE control dropping 51.8%→26.3% under the same pooling).

Net: univariate TimesFM is **architecturally mismatched** to DE-based emotion
recognition, and a plain raw-DE linear classifier (51–63%) beats every TimesFM
variant by a wide margin. This is a much stronger motivation for **Phase 2**
than the original framing: Phase 2 must keep the multichannel (C×B) band-power
matrix intact as a structured patch and avoid per-series normalization, rather
than treating channel-bands as 310 independent normalized series. The remaining
gap to PC-SSL (raw-DE 51–63% vs PC-SSL ~84–92%) is the nonlinear
encoder + band/channel attention, which Phase 2's transformer is meant to supply.

Reproduce the diagnostics:

```bash
python scripts/diagnose_timesfm_trial_embeddings.py              # (b)+(e)
python scripts/evaluate_timesfm_segment_embeddings.py \
  --datasets seed_v seed_iv --models frozen finetuned            # STEP 3 + 4
```

Embed SEED-V and classify:

```bash
python scripts/extract_timesfm_trial_embeddings.py \
  --archive data/physiofm/de_features/seed_v_de_LDS.npz \
  --adapter_dir results/phase1_timesfm_de/seed_iv_v_lora \
  --output results/phase1_timesfm_de/seed_v_embeddings.npz

python scripts/evaluate_timesfm_trial_embeddings.py \
  --embeddings results/phase1_timesfm_de/seed_v_embeddings.npz \
  --dataset seed_v \
  --output_csv results/phase1_timesfm_de/seed_v_trial_classification.csv
```

Embed SEED-IV and classify:

```bash
python scripts/extract_timesfm_trial_embeddings.py \
  --archive data/physiofm/de_features/seed_iv_de_LDS.npz \
  --adapter_dir results/phase1_timesfm_de/seed_iv_v_lora \
  --output results/phase1_timesfm_de/seed_iv_embeddings.npz

python scripts/evaluate_timesfm_trial_embeddings.py \
  --embeddings results/phase1_timesfm_de/seed_iv_embeddings.npz \
  --dataset seed_iv \
  --output_csv results/phase1_timesfm_de/seed_iv_trial_classification.csv
```

## Dataset Scope

The proposal text lists SEED-IV, SEED-V, Sleep-EDF Expanded, CHB-MIT, and BCI
Competition IV-2a, while the timeline says "all four datasets." Locally
available data currently covers SEED, SEED-IV, and SEED-V. The DE extraction
core supports raw EEG arrays, and the next adapters should use MNE for EDF/GDF
datasets when Sleep-EDF, CHB-MIT, and BCI IV-2a are uploaded.
