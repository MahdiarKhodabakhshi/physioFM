#!/usr/bin/env bash
# F6 driver: parameter-scale ladder on UN-SMOOTHED SEED-IV, tracking the
# pretrained-minus-random gap as model size grows (p_in=1, p_out=16 fixed).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="results/phase2/followup/f6"
EPOCHS="${EPOCHS:-60}"
DS=seed_iv_raw

# hidden:layers:heads scale points
for cfg in 128:4:4 256:6:8 512:8:8; do
  IFS=: read -r h l hd <<< "$cfg"
  out="$ROOT/h${h}_l${l}"
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
    --p_in 1 --p_out 16 --hidden "$h" --layers "$l" --heads "$hd" \
    --epochs "$EPOCHS" --output_dir "$out/pc"
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
    --p_in 1 --p_out 16 --hidden "$h" --layers "$l" --heads "$hd" \
    --epochs 0 --output_dir "$out/rand"
done
echo "F6 pretrain DONE"
