#!/usr/bin/env bash
# MI multi-seed: bring motor imagery to the same seed coverage as sleep and emotion,
# so every task in the cross-task comparison reports a multi-seed mean. MI trials are
# short (13 DE windows) so p_out=8; pretraining is a couple of minutes per seed.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
ROOT="results/phase3/parity"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-64}"
SEEDS="${SEEDS:-1 2 3}"
TAG="scratch_pin1_pout8_linear"
mkdir -p "$ROOT"

for SEED in $SEEDS; do
  echo "=== MI seed $SEED: pretrain PC + matched random-init ==="
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets bci_iv_2a \
    --p_in 1 --p_out 8 --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" \
    --output_dir "$ROOT/mi_seed${SEED}/pc"
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets bci_iv_2a \
    --p_in 1 --p_out 8 --epochs 0 --batch "$BATCH" --seed "$SEED" \
    --output_dir "$ROOT/mi_seed${SEED}/rand"
  echo "=== MI seed $SEED: eval ==="
  "$PY" scripts/phase2_bci_eval.py \
    --pc_dir "$ROOT/mi_seed${SEED}/pc/$TAG" \
    --rand_dir "$ROOT/mi_seed${SEED}/rand/$TAG" \
    --raw --label_fracs 1.0 --out_dir "$ROOT" --tag "_mi_seed${SEED}"
done

echo "MI MULTISEED DONE -> $ROOT/f16_bci_mi_seed*.csv"
