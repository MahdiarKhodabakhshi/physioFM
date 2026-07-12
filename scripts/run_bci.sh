#!/usr/bin/env bash
# F16 driver: motor-imagery (BCI-IV-2a) — the 2nd dynamic task for the temporal-PC
# thesis. Builds the DE archive, PC-pretrains + matched random-init, evaluates both
# + the raw-DE ceiling under leakage-free session-holdout (train T, test E, per subj).
# Needs a GPU for pretraining (the eval is CPU + fast). Same ladder as sleep (F13).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
ROOT="results/phase3/f16"
EPOCHS="${EPOCHS:-60}"
PIN="${PIN:-1}"
# MI trials are short (13 DE windows), so a shorter multi-step horizon than sleep's 16.
POUT="${POUT:-8}"
BATCH="${BATCH:-64}"
DS=bci_iv_2a
TAG="scratch_pin${PIN}_pout${POUT}_linear"

# 1. Build DE archive if missing (CPU; datasets/BCI-IV-2a/A0{1..9}{T,E}.mat).
[ -f data/physiofm/de_features/bci_iv_2a_de.npz ] || "$PY" scripts/build_bci_dataset.py

# 2. PC pretraining + matched random-init (labels unused; SSL over all sessions).
"$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
  --p_in "$PIN" --p_out "$POUT" --epochs "$EPOCHS" --batch "$BATCH" --output_dir "$ROOT/pc"
"$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
  --p_in "$PIN" --p_out "$POUT" --epochs 0 --batch "$BATCH" --output_dir "$ROOT/rand"

# 3. Session-holdout eval (train T -> test E, per subject), pc vs rand vs raw-DE.
"$PY" scripts/phase2_bci_eval.py \
  --pc_dir   "$ROOT/pc/$TAG" \
  --rand_dir "$ROOT/rand/$TAG" \
  --raw --classifiers logreg --out_dir "$ROOT"

echo "F16 DONE -> $ROOT/f16_bci.csv"
