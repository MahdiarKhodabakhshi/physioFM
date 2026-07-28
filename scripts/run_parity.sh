#!/usr/bin/env bash
# Protocol-parity driver: bring EMOTION and MOTOR IMAGERY up to the same analyses
# sleep (EXP-0009) and seizure (EXP-0015) already have, so the 4-task comparison is
# apples-to-apples: multi-seed headline, label-efficiency curves, order-shuffle
# control, and per-fold/per-subject outputs for paired tests.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
ROOT="results/phase3/parity"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-64}"
SEEDS="${SEEDS:-1 2 3}"
FRACS="${FRACS:-0.01 0.05 0.1 0.25 0.5 1.0}"
mkdir -p "$ROOT"

# ---------------------------------------------------------------- EMOTION
# Two variants on the SAME trials/labels/folds: smoothed (de_LDS, the public
# benchmark) and un-smoothed (de_movingAve) — the two temporal-structure levels.
for DS in seed_iv seed_iv_raw; do
  for SEED in $SEEDS; do
    echo "=== EMOTION $DS seed $SEED: pretrain PC + matched random-init ==="
    "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
      --p_in 1 --p_out 16 --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" \
      --output_dir "$ROOT/emotion_${DS}_seed${SEED}/pc"
    "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" \
      --p_in 1 --p_out 16 --epochs 0 --batch "$BATCH" --seed "$SEED" \
      --output_dir "$ROOT/emotion_${DS}_seed${SEED}/rand"
  done

  TAG=scratch_pin1_pout16_linear
  # headline, multi-seed (full labels)
  for SEED in $SEEDS; do
    "$PY" scripts/phase2_emotion_parity.py \
      --pc_dir "$ROOT/emotion_${DS}_seed${SEED}/pc/$TAG" \
      --rand_dir "$ROOT/emotion_${DS}_seed${SEED}/rand/$TAG" \
      --raw --datasets "$DS" --label_fracs 1.0 \
      --out_dir "$ROOT" --tag "_${DS}_seed${SEED}"
  done
  # label-efficiency curve + order-shuffle control (seed 1, matching sleep/seizure)
  S1=$(echo $SEEDS | awk '{print $1}')
  "$PY" scripts/phase2_emotion_parity.py \
    --pc_dir "$ROOT/emotion_${DS}_seed${S1}/pc/$TAG" \
    --rand_dir "$ROOT/emotion_${DS}_seed${S1}/rand/$TAG" \
    --raw --datasets "$DS" --label_fracs $FRACS \
    --out_dir "$ROOT" --tag "_${DS}_labelcurve"
  "$PY" scripts/phase2_emotion_parity.py \
    --pc_dir "$ROOT/emotion_${DS}_seed${S1}/pc/$TAG" \
    --rand_dir "$ROOT/emotion_${DS}_seed${S1}/rand/$TAG" \
    --raw --datasets "$DS" --label_fracs 1.0 --shuffle_time \
    --out_dir "$ROOT" --tag "_${DS}_shuffle"
done

# ---------------------------------------------------------- MOTOR IMAGERY
# Reuses the existing F16 models (p_out=8, short MI trials).
MITAG=scratch_pin1_pout8_linear
if [ -f "results/phase3/f16/pc/$MITAG/model.pt" ]; then
  echo "=== MI: label-efficiency curve ==="
  "$PY" scripts/phase2_bci_eval.py \
    --pc_dir "results/phase3/f16/pc/$MITAG" --rand_dir "results/phase3/f16/rand/$MITAG" \
    --raw --label_fracs $FRACS --out_dir "$ROOT" --tag "_mi_labelcurve"
  echo "=== MI: order-shuffle control ==="
  "$PY" scripts/phase2_bci_eval.py \
    --pc_dir "results/phase3/f16/pc/$MITAG" --rand_dir "results/phase3/f16/rand/$MITAG" \
    --raw --label_fracs 1.0 --shuffle_time --out_dir "$ROOT" --tag "_mi_shuffle"
else
  echo "!! MI models missing at results/phase3/f16 — run scripts/run_bci.sh first"
fi

echo "PARITY DONE -> $ROOT/"
