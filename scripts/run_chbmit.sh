#!/usr/bin/env bash
# F17 driver: CHB-MIT seizure detection — the truer 2nd dynamic task. Builds the DE
# archive, PC-pretrains + matched random-init, evaluates both + raw-DE under
# leave-one-patient-out with imbalance-aware metrics. Needs a GPU for pretraining.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
ROOT="results/phase3/f17"
EPOCHS="${EPOCHS:-60}"
PIN="${PIN:-1}"
POUT="${POUT:-16}"   # long seizure recordings -> multi-horizon like sleep
BATCH="${BATCH:-16}"
DS=chbmit
TAG="scratch_pin${PIN}_pout${POUT}_linear"

# 1. Build DE archive if missing (CPU; needs datasets/CHB-MIT/chb*/ downloaded).
[ -f data/physiofm/de_features/chbmit_de.npz ] || "$PY" scripts/build_chbmit_dataset.py

# 2. PC pretraining + matched random-init (labels unused).
"$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
  --p_in "$PIN" --p_out "$POUT" --epochs "$EPOCHS" --batch "$BATCH" --output_dir "$ROOT/pc"
"$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
  --p_in "$PIN" --p_out "$POUT" --epochs 0 --batch "$BATCH" --output_dir "$ROOT/rand"

# 3. Leave-one-patient-out eval, pc vs rand vs raw-DE, imbalance-aware metrics.
"$PY" scripts/phase2_chbmit_eval.py \
  --pc_dir   "$ROOT/pc/$TAG" \
  --rand_dir "$ROOT/rand/$TAG" \
  --raw --classifiers logreg --out_dir "$ROOT"

echo "F17 DONE -> $ROOT/f17_chbmit.csv"
