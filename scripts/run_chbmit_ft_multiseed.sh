#!/usr/bin/env bash
# Seizure fine-tuned multi-seed: is the -1.55 pretraining "penalty" real, or seed noise?
#
# The single-seed fine-tuned result was PC 78.66 vs random-init 80.21 bal-acc. Against a
# +/-15.7 per-patient spread that is NOT established as significantly negative, so the
# defensible reading is "no measurable benefit" rather than "random-init wins". This runs
# 3 seeds so the claim can be stated properly.
#
# Each seed: fresh PC pretraining + a fresh matched random-init, then the SAME leave-one-
# patient-out fine-tuned evaluation. Only --epochs differs between the two arms.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
ROOT="results/phase3/f17/multiseed"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-4}"          # 7213-epoch recordings; O(L^2) attention needs a small batch
FT_EPOCHS="${FT_EPOCHS:-4}"
SEEDS="${SEEDS:-1 2 3}"
TAG="scratch_pin1_pout16_linear"
mkdir -p "$ROOT"

for SEED in $SEEDS; do
  echo "=== SEIZURE seed $SEED : pretrain PC ==="
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets chbmit \
    --p_in 1 --p_out 16 --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" \
    --output_dir "$ROOT/seed${SEED}/pc"
  echo "=== SEIZURE seed $SEED : matched random-init ==="
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets chbmit \
    --p_in 1 --p_out 16 --epochs 0 --batch "$BATCH" --seed "$SEED" \
    --output_dir "$ROOT/seed${SEED}/rand"
  echo "=== SEIZURE seed $SEED : fine-tuned LOPO eval ==="
  "$PY" scripts/phase2_chbmit_finetune.py \
    --pc_dir   "$ROOT/seed${SEED}/pc/$TAG" \
    --rand_dir "$ROOT/seed${SEED}/rand/$TAG" \
    --epochs "$FT_EPOCHS" \
    --out_csv "$ROOT/f17_ft_seed${SEED}.csv"
done

echo "SEIZURE FT MULTISEED DONE -> $ROOT/f17_ft_seed*.csv"
