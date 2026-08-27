#!/usr/bin/env bash
# Physio2018 ladder (docs/SLEEP_DATASET_CANDIDATES.md): SleePyCo 5-fold protocol.
# Per fold: PC pretrain (60 ep) + rand control on the fold's non-test recordings,
# then full fine-tune with val-kappa best-epoch; pooled + per-fold metrics.
# Two channel variants: 6x64 (our full recipe) and C3-M2-only 1x64 (ladder-identical).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="${ROOT:-results/phase4/p2018}"
EPOCHS="${EPOCHS:-60}"
SEED="${SEED:-42}"
VARIANTS="${VARIANTS:-6ch c3}"
mkdir -p "$ROOT/logs"
stamp() { date +%H:%M:%S; }

if [ ! -f data/physiofm/tf_features/p2018_tf64.npz ]; then
  echo "[$(stamp)] building P2018 tf64 archives (994 records)"
  "$PY" scripts/build_p2018_dataset.py --workers "${WORKERS:-7}" 2>&1 | tee "$ROOT/logs/build.log"
fi
if [ ! -f data/physiofm/tf_features/p2018_pretrain_fold5.npz ]; then
  echo "[$(stamp)] preparing fold corpora + c3 slices"
  "$PY" scripts/prepare_p2018_folds.py 2>&1 | tee "$ROOT/logs/folds.log"
fi

for V in $VARIANTS; do
  case "$V" in
    6ch) PREKEY="p2018_pretrain_fold";    PROOT="$ROOT/pretrain";    AK="p2018_tf64" ;;
    c3)  PREKEY="p2018_c3_pretrain_fold"; PROOT="$ROOT/pretrain_c3"; AK="p2018_tf64_c3" ;;
  esac
  for K in 1 2 3 4 5; do
    for ARM in pc rand; do
      case "$ARM" in
        pc)   EXTRA="--epochs $EPOCHS" ;;
        rand) EXTRA="--epochs 0" ;;
      esac
      OUT="$PROOT/fold$K"
      if [ -f "$OUT/$ARM/DONE" ]; then echo "skip $V fold$K/$ARM"; continue; fi
      echo "[$(stamp)] pretrain $V fold$K $ARM"
      # ARCH lookup by key: fold corpora are registered ad hoc via --datasets path form
      "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "data/physiofm/tf_features/${PREKEY}${K}.npz" \
        --p_in 1 --p_out 16 --batch 16 --seed "$SEED" --objective input $EXTRA \
        --output_dir "$OUT" --tag "$ARM" \
        2>&1 | tee "$ROOT/logs/pretrain_${V}_fold${K}_${ARM}.log" \
             | { grep -E "epoch (1|30|60)/|corpus|params" || true; }
      [ -f "$OUT/$ARM/DONE" ] || { echo "FATAL: pretrain $V fold$K/$ARM no DONE"; exit 1; }
    done
  done
  if ! grep -q "full_${V}," "$ROOT/finetune.csv" 2>/dev/null; then
    echo "[$(stamp)] fine-tune $V (5 folds x pc/rand)"
    "$PY" scripts/phase2_p2018_finetune.py --arch_key "$AK" --pretrain_root "$PROOT" \
      --mode full --epochs 8 --ft_seed "$SEED" --tag "_${V}" --out_csv "$ROOT/finetune.csv" \
      2>&1 | tee "$ROOT/logs/finetune_${V}.log" | { grep -E "RESULT|POOLED|wrote" || true; }
    grep -q "_${V}," "$ROOT/finetune.csv" || { echo "FATAL: fine-tune $V wrote no rows"; exit 1; }
  fi
done
echo OK > "$ROOT/QUEUE_DONE"
echo "[$(stamp)] P2018 QUEUE DONE"
