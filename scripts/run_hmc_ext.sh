#!/usr/bin/env bash
# HMC extension: seeds 3-7 (same ladder) + 20-epoch FT robustness on seeds 42 1 2.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS="3 4 5 6 7" bash scripts/run_hmc.sh
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="results/phase4/hmc"
for SEED in 42 1 2 3 4 5 6 7; do
  if ! grep -q "_e20_seed$SEED," "$ROOT/finetune.csv" 2>/dev/null; then
    "$PY" scripts/phase2_hmc_finetune.py --pc_dir "$ROOT/pretrain/seed$SEED/pc" \
      --rand_dir "$ROOT/pretrain/seed$SEED/rand" --mode full --epochs 20 \
      --ft_seed "$SEED" --tag "_e20_seed$SEED" --out_csv "$ROOT/finetune.csv" \
      2>&1 | tee "$ROOT/logs/finetune_e20_seed$SEED.log" | { grep -E "RESULT|best epoch" || true; }
  fi
done
echo OK > "$ROOT/EXT_DONE"
echo "HMC EXT DONE"
