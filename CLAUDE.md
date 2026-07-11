# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project (**PhysioFM**) investigating whether time-series / foundation-model
machinery can do EEG emotion recognition on the SEED datasets, using **differential
entropy (DE)** features. It is organized as a sequence of experimental phases, each
documented in `docs/` with a results log, and is not a deployed application — the
"product" is the experiments and their write-ups.

- **Phase 1** (`docs/PHASE1.md`, complete): feed each channel-band DE trace to
  **TimesFM** as a univariate series. Result is **at chance** and that null is real
  (instance-norm strips absolute band power; flattening erases channel-band identity).
- **Phase 2** (`docs/PHASE2.md`): **PhysioFM-S** — replace TimesFM's scalar patch with
  a structured `(C×B)`=310-d DE patch in a decoder-only, predictive-coding transformer,
  with NO per-series instance norm.
- **Stage-2 follow-ups** (`docs/PHASE2_FOLLOWUP.md`, spec in
  `docs/PhysioFM_Stage2_FollowUp_Experiments.md`): F1, F2, F4–F7, F12 ablations. Key
  finding: the Phase-2 "temporal PC adds nothing" null is largely an artifact of LDS
  smoothing destroying learnable dynamics; it flips on un-smoothed DE.

## Repo layout (what is and isn't committed)

- `physiofm/` — **the committed Python package** (the real source).
- `scripts/` — **committed** experiment entry points (argparse CLIs + `run_fN.sh` drivers).
- `docs/` — **committed** phase write-ups; treat these as the source of truth for
  experimental decisions, locked claims, and results. Update the relevant `docs/*.md`
  when an experiment's outcome changes.
- `timesfm/` and `PC-SSL/` — **external reference clones, gitignored.** PhysioFM scripts
  do NOT import from `timesfm/` (TimesFM is used via the HuggingFace `transformers`
  checkpoint `google/timesfm-2.5-200m-transformers`). `PC-SSL/` holds the reference repo +
  raw data + author splits/models, used only by the F12 audit.
- `datasets/`, `data/`, `results/` — **gitignored.** Raw `.mat` datasets, derived DE
  archives, and all experiment outputs respectively.

## Environment & running

There is no build step and **no committed test suite** for the `physiofm` package
(the only `test_*.py` files live in the gitignored `timesfm/` clone). Work is done by
running scripts directly.

- Interpreter: `/home/mahdiar/.conda/envs/xcqa/bin/python` (conda env `xcqa`: torch 2.7,
  transformers 5.12, timesfm2.5; runs on H100). Scripts insert the repo root on
  `sys.path`, so run them from anywhere with that interpreter; `run_fN.sh` honor a `$PY` override.
- Quick sanity check: `python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__)"`

Typical Phase-2 flow:

```bash
PY=/home/mahdiar/.conda/envs/xcqa/bin/python
# 1. Build canonical DE archives from raw .mat -> data/physiofm/de_features/<ds>_<key>.npz
$PY scripts/build_de_dataset.py --dataset seed_iv          # also: seed_v, seed
# 2. Self-supervised predictive-coding pretraining (writes <out>/<tag>/, tag = variant_pinN_poutN_embedder)
$PY scripts/phase2_pretrain.py --variant scratch --datasets seed_iv seed_v seed \
    --p_in 1 --p_out 16 --epochs 40 --output_dir results/phase2/pretrain
# 3a. Zero-shot linear probe of the frozen encoder
$PY scripts/phase2_extract_eval.py --model_dir <out>/<tag> --datasets seed_iv \
    --classifiers logreg linear_svm
# 3b. Fine-tuned eval (full | io | head; io = train only structured in/out blocks)
$PY scripts/phase2_finetune_eval.py --model_dir <out>/<tag> --mode io --label_fracs 0.1 0.5 1.0
# Baselines (no model):
$PY scripts/phase2_raw_de_ceilings.py --datasets seed_v seed_iv
```

Reproduce a whole follow-up experiment with its driver, e.g. `bash scripts/run_f1.sh`
(set `EPOCHS=...` to shorten). A pretrain run with `--epochs 0` is the matched
**random-init** control that every follow-up pairs against the PC-pretrained model.

## Architecture / key abstractions

- **DE is the universal currency** (`physiofm/de.py`). Canonical `DETrial` shape is
  `time × channels(62) × bands(5)`. `load_de_archive` / `save_de_archive` read/write the
  `.npz` corpora; hard-coded `SEED_*_LABELS` map trials to emotion classes.
- **Structured-patch pipeline** (`physiofm/structured_data.py`). A trial becomes a
  sequence of T tokens, each a flattened 310-d DE window. Normalization is **fixed
  per-(channel,band) standardization fit on the corpus** (NOT per-series instance norm) —
  this is the deliberate fix for the Phase-1 failure mode, so preserve it. `ARCH` maps
  dataset keys to `.npz` paths (note `seed_iv_raw` = un-smoothed `de_movingAve`, available
  for SEED-IV only).
- **PhysioFM-S model** (`physiofm/physiofm_s.py`). Structured input embedder
  (`linear` or PC-SSL-style `attn` band/channel attention) → causal decoder stack →
  output block predicting the next `p_out` DE windows (multi-step PC-MSE). `--variant`:
  `scratch` (random init), `timesfm` (real TimesFm2_5 decoder layers with pretrained
  weights), `timesfm_rand` (same architecture, random init). It feeds embeddings straight
  into the decoder layers — it deliberately bypasses TimesFM's `model.forward` to avoid
  the RevIN instance norm.
- **Frozen evaluation harness** (`physiofm/phase2_eval.py`). EVERY Phase-2 model is
  scored through this so the comparison ladder is identical: segment-level, PC-SSL
  subject-dependent fold masks (`seed_v_fold_mask` / `seed_iv_fold_mask` in
  `physiofm/embedding_evaluation.py`), seed 42, StandardScaler + balanced classifier.
  Reuse it rather than writing new eval logic. `base_dataset()` resolves variant keys
  (e.g. `seed_iv_raw` → `seed_iv`) to the right split/label family.

## Conventions

- Phase-2 pretrain runs write to `<output_dir>/<tag>` where
  `tag = "{variant}_pin{p_in}_pout{p_out}_{embedder}"`. Downstream scripts take that
  directory as `--model_dir`.
- Results land under `results/phase2/...` (gitignored) as paired `.csv` + `.md`. When an
  experiment finishes, also fold its verdict into the matching `docs/*.md` table.
- Phase-1 (`timesfm_*` modules/scripts) is frozen; new work is Phase 2 / follow-ups.

## Experiment log (lab notebook)

Every experiment is journaled in `docs/experiments/` — one file per experiment with
why/setup/results(dated)/interpretation, a verification slot reserved for the user,
and the related commits. **Create the entry when you start an experiment and update
it when it finishes.** Use the `experiment-log` skill (`/experiment`); the full
convention (shared with Cursor/Codex) is in `AGENTS.md` and
`docs/experiments/README.md`. The phase docs stay the curated results; this log is
the chronological journal. Never fill §6 — it's the user's verification.

## graphify knowledge graph

A knowledge graph lives in `graphify-out/`. Per `AGENTS.md`: for codebase questions,
prefer `graphify query "<question>"` / `graphify path "<A>" "<B>"` /
`graphify explain "<concept>"` when `graphify-out/graph.json` exists; use
`graphify-out/wiki/index.md` for navigation and `GRAPH_REPORT.md` for broad architecture.
After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
