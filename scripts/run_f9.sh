#!/usr/bin/env bash
# F9 driver: does a static-structure SSL (masked-DE reconstruction) help where
# temporal forecasting did not? Trains three matched encoders — masked-recon,
# forecasting-PC, random-init — and probes each with the identical frozen probe,
# on smoothed (combined corpus) and un-smoothed (seed_iv_raw) DE.
# Pre-registered fork in docs/experiments/EXP-0011.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="results/phase2/followup/f9"
EPOCHS="${EPOCHS:-40}"
MASK_RATIO="${MASK_RATIO:-0.5}"
MASK_MODE="${MASK_MODE:-random}"
MASK_TAG="masked_${MASK_MODE}$(python3 -c "print(int($MASK_RATIO*100))")_pin1_pout1_linear"
FC_TAG="scratch_pin1_pout16_linear"

run_group () {  # $1=group dir  $2..=pretrain datasets ; probe datasets via $PROBE_DS
  local gdir="$1"; shift
  local dss=("$@")
  # 1. masked-DE reconstruction
  "$PY" scripts/phase2_pretrain_masked.py --datasets "${dss[@]}" \
    --mask_ratio "$MASK_RATIO" --mask_mode "$MASK_MODE" --epochs "$EPOCHS" \
    --output_dir "$gdir/masked"
  # 2. forecasting PC
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "${dss[@]}" \
    --p_in 1 --p_out 16 --epochs "$EPOCHS" --output_dir "$gdir/pc"
  # 3. random-init control
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "${dss[@]}" \
    --p_in 1 --p_out 16 --epochs 0 --output_dir "$gdir/rand"
  # probe all three identically
  "$PY" scripts/phase2_extract_eval.py --model_dir "$gdir/masked/$MASK_TAG" \
    --datasets "${PROBE_DS[@]}" --classifiers logreg linear_svm --out_csv "$gdir/probe_masked.csv"
  "$PY" scripts/phase2_extract_eval.py --model_dir "$gdir/pc/$FC_TAG" \
    --datasets "${PROBE_DS[@]}" --classifiers logreg linear_svm --out_csv "$gdir/probe_pc.csv"
  "$PY" scripts/phase2_extract_eval.py --model_dir "$gdir/rand/$FC_TAG" \
    --datasets "${PROBE_DS[@]}" --classifiers logreg linear_svm --out_csv "$gdir/probe_rand.csv"
}

# Smoothed DE: combined corpus, probe SEED-V + SEED-IV.
PROBE_DS=(seed_v seed_iv)
run_group "$ROOT/smoothed" seed_v seed_iv seed

# Un-smoothed DE: SEED-IV only.
PROBE_DS=(seed_iv_raw)
run_group "$ROOT/raw" seed_iv_raw

echo "F9 DONE -> $ROOT/{smoothed,raw}/probe_*.csv"
