#!/usr/bin/env bash
# F1 driver: matched PhysioFM-S runs on smoothed (de_LDS) vs un-smoothed
# (de_movingAve) SEED-IV, isolating LDS smoothing as the only changed variable.
# Each run is SEED-IV-only so smoothed and raw corpora are otherwise identical.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="results/phase2/followup/f1"
EPOCHS="${EPOCHS:-60}"

pretrain () {  # $1=datasets-key $2=outdir $3=epochs
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$1" \
    --p_in 1 --p_out 16 --epochs "$3" --output_dir "$2"
}
probe () {     # $1=model_dir $2=dataset-key
  "$PY" scripts/phase2_extract_eval.py --model_dir "$1" --datasets "$2" \
    --classifiers logreg linear_svm --out_csv "$1/eval_zeroshot.csv"
}

TAG="scratch_pin1_pout16_linear"

# --- smoothed SEED-IV ---
pretrain seed_iv      "$ROOT/smoothed_pc"   "$EPOCHS"
pretrain seed_iv      "$ROOT/smoothed_rand" 0
probe "$ROOT/smoothed_pc/$TAG"   seed_iv
probe "$ROOT/smoothed_rand/$TAG" seed_iv

# --- un-smoothed SEED-IV ---
pretrain seed_iv_raw  "$ROOT/raw_pc"   "$EPOCHS"
pretrain seed_iv_raw  "$ROOT/raw_rand" 0
probe "$ROOT/raw_pc/$TAG"   seed_iv_raw
probe "$ROOT/raw_rand/$TAG" seed_iv_raw

# --- persistence baseline + variance decomposition + model PC-MSE ---
"$PY" scripts/phase2_f1_smoothing.py --p_out 16 \
  --smoothed_model_dir "$ROOT/smoothed_pc/$TAG" \
  --raw_model_dir      "$ROOT/raw_pc/$TAG" \
  --probe_root "$ROOT" --probe_tag "$TAG"

echo "F1 DONE"
