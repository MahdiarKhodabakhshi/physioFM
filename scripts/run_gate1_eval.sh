#!/usr/bin/env bash
# Gate 1 evaluation ladder (docs/NEXT_PHASE_PLAN.md; EXP-0021).
# For each task: frozen probe (raw features + pc + latent + rand) and END-TO-END FINE-TUNING
# (pc + latent + rand) on the identical subject-/patient-disjoint folds used everywhere else.
# Usage: ARCH=sleep_edf|chbmit|sleep_edf_tf64|chbmit_tf64 SEED=42 bash scripts/run_gate1_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="${ROOT:-results/phase4/gate1}"
ARCH="${ARCH:-sleep_edf}"
SEED="${SEED:-42}"
DO_FROZEN="${DO_FROZEN:-1}"
DO_FT="${DO_FT:-1}"
M="$ROOT/$ARCH/seed$SEED"
OUT="$ROOT/$ARCH"
mkdir -p "$OUT" "$ROOT/logs"
ARMS="--pc_dir $M/pc --latent_dir $M/latent --rand_dir $M/rand"
TAG="_seed$SEED"

case "$ARCH" in
  sleep_edf*)
    if [ "$DO_FROZEN" = 1 ]; then
      "$PY" scripts/phase2_f13_sleep.py --arch_key "$ARCH" $ARMS --raw --classifiers logreg --k 5 \
        --out_dir "$OUT" --tag "_frozen$TAG" 2>&1 | tee "$ROOT/logs/eval_${ARCH}_frozen$TAG.log" | grep -E "RESULT|wrote|corpus"
    fi
    if [ "$DO_FT" = 1 ]; then
      "$PY" scripts/phase2_sleep_finetune.py --arch_key "$ARCH" $ARMS --mode full --epochs 8 --k 5 \
        --out_csv "$OUT/finetune.csv" --tag "$TAG" 2>&1 | tee "$ROOT/logs/eval_${ARCH}_ft$TAG.log" | grep -E "RESULT|wrote"
    fi ;;
  chbmit*)
    if [ "$DO_FROZEN" = 1 ]; then
      "$PY" scripts/phase2_chbmit_eval.py --arch_key "$ARCH" $ARMS --raw --classifiers logreg \
        --out_dir "$OUT" --tag "_frozen$TAG" 2>&1 | tee "$ROOT/logs/eval_${ARCH}_frozen$TAG.log" | grep -E "RESULT|wrote|CHB-MIT"
    fi
    if [ "$DO_FT" = 1 ]; then
      "$PY" scripts/phase2_chbmit_finetune.py --arch_key "$ARCH" $ARMS --epochs 4 \
        --out_csv "$OUT/finetune.csv" --tag "$TAG" 2>&1 | tee "$ROOT/logs/eval_${ARCH}_ft$TAG.log" | grep -E "RESULT|wrote|CHB-MIT"
    fi ;;
esac
echo "GATE1 EVAL DONE ($ARCH seed$SEED)"
