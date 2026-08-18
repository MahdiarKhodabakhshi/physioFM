#!/usr/bin/env bash
# Next-phase plan, Gate 1 (docs/NEXT_PHASE_PLAN.md): the OBJECTIVE fix.
# Pretrain, on the same DE corpora and with the same seed, three matched arms per task:
#   pc      input-space predictive coding (Phase-2 objective; the reference)
#   latent  latent-target predictive coding (JEPA/BYOL-style, EMA target, stop-grad)
#   rand    matched random-init control (--epochs 0)
# The pretraining protocol mirrors the F13 / F17 drivers (60 epochs, p_in=1, p_out=16).
# All arms use SDPA attention (numerically identical to eager) so whole recordings fit
# a 20 GB card without chunking. Downstream: scripts/run_gate1_eval.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="${ROOT:-results/phase4/gate1}"
EPOCHS="${EPOCHS:-60}"
SEED="${SEED:-42}"
TASKS="${TASKS:-sleep_edf chbmit seed_iv_raw seed_iv bci_iv_2a}"
mkdir -p "$ROOT/logs"

for DS in $TASKS; do
  case "$DS" in
    sleep_edf*) BATCH=16; POUT=16 ;;
    chbmit*)    BATCH=4;  POUT=16 ;;
    bci_iv_2a)  BATCH=64; POUT=8 ;;   # F16 used p_out=8 (13-window trials)
    *)          BATCH=64; POUT=16 ;;
  esac
  OUT="$ROOT/$DS/seed$SEED"
  for ARM in pc latent rand; do
    case "$ARM" in
      pc)     EXTRA="--objective input  --epochs $EPOCHS" ;;
      latent) EXTRA="--objective latent --epochs $EPOCHS" ;;
      rand)   EXTRA="--objective input  --epochs 0" ;;
    esac
    if [ -f "$OUT/$ARM/DONE" ]; then echo "skip $DS/$ARM (exists)"; continue; fi
    echo "=== $DS $ARM ==="
    "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" --p_in 1 --p_out "$POUT" \
      --batch "$BATCH" --seed "$SEED" $EXTRA --output_dir "$OUT" --tag "$ARM" \
      2>&1 | tee "$ROOT/logs/${DS}_seed${SEED}_${ARM}.log" | grep -E "epoch (1|10|20|30|40|50|60)/|DONE|corpus|params"
  done
done
echo "GATE1 PRETRAIN DONE"
