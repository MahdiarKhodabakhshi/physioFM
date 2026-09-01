#!/usr/bin/env bash
# EXP-0028: our causal sequence recipe stacked on frozen REVE-base features (HMC).
# Ladder per seed: PC pretrain (60 ep) + rand control on train+val subjects only,
# then fixed-split fine-tune (e20, val-kappa selection) — identical protocol to the
# tf64 ladder, so rows are directly comparable to both our 73.8 and REVE's 74.0.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="results/phase4/reve_stack"
SEEDS="${SEEDS:-42 1 2}"
mkdir -p "$ROOT/logs"
stamp() { date +%H:%M:%S; }

[ -f data/physiofm/reve_features/hmc_reve_pretrain.npz ] || { echo "FATAL: run build_hmc_reve.py first"; exit 1; }

for SEED in $SEEDS; do
  OUT="$ROOT/pretrain/seed$SEED"
  for ARM in pc rand; do
    EP=60; [ "$ARM" = rand ] && EP=0
    if [ -f "$OUT/$ARM/DONE" ]; then echo "skip pretrain seed$SEED/$ARM"; continue; fi
    echo "[$(stamp)] pretrain seed$SEED $ARM"
    "$PY" scripts/phase2_pretrain.py --variant scratch --datasets hmc_reve_pretrain \
      --p_in 1 --p_out 16 --batch "${BATCH:-4}" --seed "$SEED" --objective input --epochs "$EP" \
      --output_dir "$OUT" --tag "$ARM" \
      2>&1 | tee "$ROOT/logs/pretrain_seed${SEED}_${ARM}.log" \
           | { grep -E "epoch (1|30|60)/|corpus|params" || true; }
    [ -f "$OUT/$ARM/DONE" ] || { echo "FATAL: pretrain seed$SEED/$ARM no DONE"; exit 1; }
  done
  if [ -f "$OUT/pc/DONE" ] && [ -f "$OUT/rand/DONE" ] \
     && ! grep -q "_seed$SEED," "$ROOT/finetune.csv" 2>/dev/null; then
    echo "[$(stamp)] fine-tune seed$SEED (e20)"
    "$PY" scripts/phase2_hmc_finetune.py --arch_key hmc_reve \
      --labels data/physiofm/reve_features/hmc_reve_labels.npz \
      --pc_dir "$OUT/pc" --rand_dir "$OUT/rand" \
      --mode full --epochs 20 --ft_seed "$SEED" --tag "_seed$SEED" --out_csv "$ROOT/finetune.csv" \
      2>&1 | tee "$ROOT/logs/finetune_seed$SEED.log" | { grep -E "RESULT|best epoch|split:" || true; }
    grep -q "_seed$SEED," "$ROOT/finetune.csv" || { echo "FATAL: FT seed$SEED wrote no rows"; exit 1; }
  fi
done

# Control: REVE frozen, NO sequence context — balanced logistic probe on per-epoch
# features, fixed split (shows the stack's gain comes from OUR sequence model).
if [ ! -f "$ROOT/probe.txt" ]; then
  echo "[$(stamp)] frozen per-epoch probe control"
  "$PY" scripts/reve_probe_control.py > "$ROOT/probe.txt" 2>&1 || { echo "FATAL: probe failed"; exit 1; }
  tail -3 "$ROOT/probe.txt"
fi
echo OK > "$ROOT/QUEUE_DONE"
echo "[$(stamp)] REVE STACK QUEUE DONE"
