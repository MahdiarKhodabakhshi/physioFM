#!/usr/bin/env bash
# EXP-0028 extension: REVE stacks on Sleep-EDF-78 and Physio2018 (REVE-base frozen).
# Sequential (GPU-bound extraction). SEDF: whole-corpus pretrain convention + 5-fold FT
# (e8 seeds 42/1/2 + e16 seed42). P2018: SleePyCo folds, per-fold pretrain, e8.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
RF=data/physiofm/reve_features
ROOT=results/phase4/reve_stack
mkdir -p "$ROOT/logs"
stamp() { date +%H:%M:%S; }

# ---- 1. SEDF extraction + ladder ----
if [ ! -f "$RF/sleep_edf_reve_labels.npz" ]; then
  echo "[$(stamp)] SEDF REVE extraction"
  "$PY" scripts/build_sedf_reve.py 2>&1 | tee "$ROOT/logs/build_sedf.log" \
    | { grep -E "positions|built|labels identical|wrote|Error|Trace" || true; }
  [ -f "$RF/sleep_edf_reve_labels.npz" ] || { echo "FATAL: sedf extraction"; exit 1; }
fi
for SEED in 42 1 2; do
  OUT="$ROOT/sedf_pretrain/seed$SEED"
  for ARM in pc rand; do
    EP=60; [ "$ARM" = rand ] && EP=0
    [ -f "$OUT/$ARM/DONE" ] && continue
    echo "[$(stamp)] SEDF pretrain seed$SEED $ARM"
    "$PY" scripts/phase2_pretrain.py --variant scratch --datasets sleep_edf_reve \
      --p_in 1 --p_out 16 --batch 4 --seed "$SEED" --objective input --epochs "$EP" \
      --output_dir "$OUT" --tag "$ARM" 2>&1 | tee "$ROOT/logs/sedf_pre_${SEED}_${ARM}.log" \
      | { grep -E "epoch 60/|corpus|params" || true; }
    [ -f "$OUT/$ARM/DONE" ] || { echo "FATAL: sedf pretrain $SEED/$ARM"; exit 1; }
  done
  if ! grep -q "_seed$SEED" "$ROOT/sedf_finetune.csv" 2>/dev/null; then
    echo "[$(stamp)] SEDF FT seed$SEED (e8)"
    "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_reve \
      --labels "$RF/sleep_edf_reve_labels.npz" \
      --pc_dir "$OUT/pc" --rand_dir "$OUT/rand" \
      --mode full --epochs 8 --k 5 --ft_seed "$SEED" \
      --out_csv "$ROOT/sedf_finetune.csv" --tag "_seed$SEED" \
      2>&1 | tee "$ROOT/logs/sedf_ft_$SEED.log" | { grep -E "RESULT|wrote" || true; }
  fi
done
if ! grep -q "_e16_seed42" "$ROOT/sedf_finetune.csv" 2>/dev/null; then
  echo "[$(stamp)] SEDF FT e16 seed42"
  "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_reve \
    --labels "$RF/sleep_edf_reve_labels.npz" \
    --pc_dir "$ROOT/sedf_pretrain/seed42/pc" --rand_dir "$ROOT/sedf_pretrain/seed42/rand" \
    --mode full --epochs 16 --k 5 --ft_seed 42 \
    --out_csv "$ROOT/sedf_finetune.csv" --tag "_e16_seed42" \
    2>&1 | tee "$ROOT/logs/sedf_ft_e16.log" | { grep -E "RESULT|wrote" || true; }
fi

# ---- 2. P2018 extraction + fold corpora + ladder ----
if [ ! -f "$RF/p2018_reve_labels.npz" ]; then
  echo "[$(stamp)] P2018 REVE extraction (~3 h)"
  "$PY" scripts/build_p2018_reve.py 2>&1 | tee "$ROOT/logs/build_p2018.log" \
    | { grep -E "extracted|wrote|/994|Error|Trace|Assertion" || true; }
  [ -f "$RF/p2018_reve_labels.npz" ] || { echo "FATAL: p2018 extraction"; exit 1; }
fi
if [ ! -f "$RF/p2018_reve_pretrain_fold5.npz" ]; then
  echo "[$(stamp)] P2018 fold corpora"
  "$PY" - << 'PYX'
import sys; sys.path.insert(0, ".")
from pathlib import Path
from physiofm.de import load_de_archive, save_de_archive
from physiofm.physio2018 import load_folds
out = Path("data/physiofm/reve_features")
trials = load_de_archive(out / "p2018_reve.npz")
assert len(trials) == 994
for k, f in enumerate(load_folds(), start=1):
    test = set(f["test"].tolist())
    keep = [t for i, t in enumerate(trials) if i not in test]
    save_de_archive(keep, out / f"p2018_reve_pretrain_fold{k}.npz")
    print(f"fold{k}: {len(keep)} recs")
PYX
fi
for K in 1 2 3 4 5; do
  for ARM in pc rand; do
    EP=60; [ "$ARM" = rand ] && EP=0
    OUT="$ROOT/p2018_pretrain/fold$K"
    [ -f "$OUT/$ARM/DONE" ] && continue
    echo "[$(stamp)] P2018 pretrain fold$K $ARM"
    "$PY" scripts/phase2_pretrain.py --variant scratch \
      --datasets "$RF/p2018_reve_pretrain_fold$K.npz" \
      --p_in 1 --p_out 16 --batch 4 --seed 42 --objective input --epochs "$EP" \
      --output_dir "$OUT" --tag "$ARM" 2>&1 | tee "$ROOT/logs/p2018_pre_${K}_${ARM}.log" \
      | { grep -E "epoch 60/|corpus" || true; }
    [ -f "$OUT/$ARM/DONE" ] || { echo "FATAL: p2018 pretrain fold$K/$ARM"; exit 1; }
  done
done
if ! grep -q "_reve" "$ROOT/p2018_finetune.csv" 2>/dev/null; then
  echo "[$(stamp)] P2018 FT (5 folds x pc/rand)"
  "$PY" scripts/phase2_p2018_finetune.py --arch_key p2018_reve \
    --labels "$RF/p2018_reve_labels.npz" --pretrain_root "$ROOT/p2018_pretrain" \
    --mode full --epochs 8 --ft_seed 42 --tag "_reve" --out_csv "$ROOT/p2018_finetune.csv" \
    2>&1 | tee "$ROOT/logs/p2018_ft.log" | { grep -E "RESULT|POOLED|wrote" || true; }
fi
echo OK > "$ROOT/EXT_DONE"
echo "[$(stamp)] REVE EXT QUEUE DONE"
