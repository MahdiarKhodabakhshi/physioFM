#!/usr/bin/env bash
# F17 multi-seed robustness: repeat the seizure run over seeds so the PC>>random
# result (EXP-0015) is not single-seed. Fresh PC + matched random-init each seed;
# leave-one-patient-out eval with imbalance-aware metrics, tagged per seed.
# Aggregate offline with phase2_chbmit_multiseed_summary.py.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
ROOT="results/phase3/f17/multiseed"
EPOCHS="${EPOCHS:-60}"
PIN=1; POUT=16
BATCH="${BATCH:-16}"
DS=chbmit
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
  "$PY" scripts/phase2_chbmit_eval.py \
    --pc_dir   "$ROOT/seed${SEED}/pc/$TAG" \
    --rand_dir "$ROOT/seed${SEED}/rand/$TAG" \
    --raw --classifiers logreg --out_dir "$ROOT" --tag "_seed${SEED}"
done

echo "F17 MULTISEED DONE -> $ROOT/f17_chbmit_seed*.csv"
