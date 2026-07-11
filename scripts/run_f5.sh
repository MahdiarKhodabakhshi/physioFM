#!/usr/bin/env bash
# F5 driver: input-context (p_in) x horizon (p_out) sweep on UN-SMOOTHED SEED-IV
# (where F1 showed real dynamics survive). For each config we pretrain a PC model
# and a matched random-init model; F5 analysis pairs them with the sequence-level
# readout (the extra context can only surface through a temporal readout).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="results/phase2/followup/f5"
EPOCHS="${EPOCHS:-60}"
DS=seed_iv_raw

for pin in 1 4 8; do
  for pout in 1 16; do
    "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
      --p_in "$pin" --p_out "$pout" --epochs "$EPOCHS" --output_dir "$ROOT/pc"
    "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
      --p_in "$pin" --p_out "$pout" --epochs 0 --output_dir "$ROOT/rand"
  done
done
echo "F5 pretrain DONE"
