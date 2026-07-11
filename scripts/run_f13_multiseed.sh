#!/usr/bin/env bash
# F13 multi-seed robustness: repeat the definitive sleep run over several seeds so
# the PC>random>raw ordering (EXP-0009) is not a single-seed artifact. Each seed
# re-pretrains a fresh PC model and a matched random-init model, then evaluates
# logreg under the same subject-disjoint folds, writing per-seed + per-fold CSVs.
# Aggregate + paired test done offline (scripts/phase2_f13_multiseed_summary.py).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
ROOT="results/phase3/f13/multiseed"
EPOCHS="${EPOCHS:-60}"
PIN=1; POUT=16
BATCH="${BATCH:-16}"
DS=sleep_edf
TAG="scratch_pin${PIN}_pout${POUT}_linear"

for SEED in ${SEEDS:-1 2 3}; do
  echo "=== SEED $SEED : pretrain PC ==="
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
    --p_in $PIN --p_out $POUT --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" \
    --output_dir "$ROOT/seed${SEED}/pc"
  echo "=== SEED $SEED : matched random-init ==="
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
    --p_in $PIN --p_out $POUT --epochs 0 --batch "$BATCH" --seed "$SEED" \
    --output_dir "$ROOT/seed${SEED}/rand"
  echo "=== SEED $SEED : eval ==="
  "$PY" scripts/phase2_f13_sleep.py \
    --pc_dir   "$ROOT/seed${SEED}/pc/$TAG" \
    --rand_dir "$ROOT/seed${SEED}/rand/$TAG" \
    --raw --classifiers logreg --k 5 --out_dir "$ROOT" --tag "_seed${SEED}"
done

echo "F13 MULTISEED DONE -> $ROOT/f13_sleep_seed*.csv"
