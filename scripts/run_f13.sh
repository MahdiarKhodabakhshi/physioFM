#!/usr/bin/env bash
# F13 driver: pre-registered temporal-PC test on SLEEP STAGING (Sleep-EDF).
# Pipeline: build the per-epoch DE corpus -> pretrain a PC model and a matched
# random-init model (epochs 0) on sleep_edf -> evaluate both + the raw-DE ceiling
# under a subject-disjoint k-fold. Pre-registered prediction (EXP-0009):
# PC-pretrained > random-init here, unlike static emotion DE.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="results/phase3/f13"
EPOCHS="${EPOCHS:-60}"
PIN="${PIN:-1}"
POUT="${POUT:-16}"
# Sleep "trials" are whole nights (~1000-2600 tokens), ~30x longer than SEED
# emotion trials, and attention is O(L^2). batch=64 (the emotion default) OOMs a
# 20GB GPU; keep it small here. Override with BATCH=... if you have more VRAM.
# (eager attention stores B*H*L*L weights per layer for backward, so memory is
# dominated by the longest night in a batch; ~2GB/sample at L~2600.)
BATCH="${BATCH:-4}"
DS=sleep_edf

# 1. Build the DE corpus (idempotent; skip if archive exists).
if [ ! -f "data/physiofm/de_features/sleep_edf_de.npz" ]; then
  "$PY" scripts/build_sleep_dataset.py
fi

# 2. Matched PC vs random-init pretraining on the sleep corpus.
"$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
  --p_in "$PIN" --p_out "$POUT" --epochs "$EPOCHS" --batch "$BATCH" --output_dir "$ROOT/pc"
"$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
  --p_in "$PIN" --p_out "$POUT" --epochs 0 --batch "$BATCH" --output_dir "$ROOT/rand"

# 3. Evaluate both encoders + raw-DE ceiling, subject-disjoint k-fold.
TAG="scratch_pin${PIN}_pout${POUT}_linear"
"$PY" scripts/phase2_f13_sleep.py \
  --pc_dir   "$ROOT/pc/$TAG" \
  --rand_dir "$ROOT/rand/$TAG" \
  --raw --classifiers logreg mlp_bal --k 5 --out_dir "$ROOT"

echo "F13 DONE -> $ROOT/f13_sleep.csv"
