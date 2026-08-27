#!/usr/bin/env bash
# HMC external-validation ladder (docs/SLEEP_DATASET_CANDIDATES.md; EXP-0024).
# Same recipe as the Gate-0 sleep tf64 ladder: scratch variant, p_in=1 p_out=16,
# 60-epoch input-space PC pretraining + matched --epochs 0 random-init control,
# then full fine-tuning (8 epochs, lr 1e-4, best epoch by val kappa) on the fixed
# published split (NeuroLM protocol; see physiofm/hmc.py).
# Pretraining corpus & standardizer: subjects SN001-SN127 only (train+val, never test).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="${ROOT:-results/phase4/hmc}"
EPOCHS="${EPOCHS:-60}"
SEEDS="${SEEDS:-42 1 2}"
mkdir -p "$ROOT/logs"
stamp() { date +%H:%M:%S; }

NEED_BUILD=0
for f in hmc_tf64.npz hmc_labels.npz hmc_tf64_pretrain.npz; do
  [ -f "data/physiofm/tf_features/$f" ] || NEED_BUILD=1
done
if [ "$NEED_BUILD" = 1 ]; then
  echo "[$(stamp)] building HMC tf64 archives"
  "$PY" scripts/build_hmc_dataset.py --workers "${WORKERS:-6}" 2>&1 | tee "$ROOT/logs/build.log"
fi

for SEED in $SEEDS; do
  OUT="$ROOT/pretrain/seed$SEED"
  for ARM in pc rand; do
    case "$ARM" in
      pc)   EXTRA="--epochs $EPOCHS" ;;
      rand) EXTRA="--epochs 0" ;;
    esac
    if [ -f "$OUT/$ARM/DONE" ]; then echo "skip pretrain seed$SEED/$ARM"; continue; fi
    echo "[$(stamp)] pretrain seed$SEED $ARM"
    "$PY" scripts/phase2_pretrain.py --variant scratch --datasets hmc_tf64_pretrain \
      --p_in 1 --p_out 16 --batch 16 --seed "$SEED" --objective input $EXTRA \
      --output_dir "$OUT" --tag "$ARM" \
      2>&1 | tee "$ROOT/logs/pretrain_seed${SEED}_${ARM}.log" \
           | { grep -E "epoch (1|20|40|60)/|corpus|params" || true; }
    [ -f "$OUT/$ARM/DONE" ] || { echo "FATAL: pretrain seed$SEED/$ARM left no DONE sentinel"; exit 1; }
  done
  if [ -f "$OUT/pc/DONE" ] && [ -f "$OUT/rand/DONE" ] \
     && ! grep -q "_seed$SEED," "$ROOT/finetune.csv" 2>/dev/null; then
    echo "[$(stamp)] fine-tune seed$SEED"
    "$PY" scripts/phase2_hmc_finetune.py --pc_dir "$OUT/pc" --rand_dir "$OUT/rand" \
      --mode full --epochs 8 --ft_seed "$SEED" --tag "_seed$SEED" --out_csv "$ROOT/finetune.csv" \
      2>&1 | tee "$ROOT/logs/finetune_seed$SEED.log" \
           | { grep -E "RESULT|split:|best epoch|wrote" || true; }
    grep -q "_seed$SEED," "$ROOT/finetune.csv" || { echo "FATAL: fine-tune seed$SEED wrote no rows"; exit 1; }
  fi
done
echo OK > "$ROOT/QUEUE_DONE"
echo "[$(stamp)] HMC QUEUE DONE"
