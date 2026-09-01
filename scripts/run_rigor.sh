#!/usr/bin/env bash
# Rigor + completeness queue (2026-09-01):
#  A. HMC REVE-stack -> 8 seeds (both loss variants)   [headline support]
#  B. multi-seed the single-seed e16 SEDF rows          [EXP-0027 flag]
#  C. REVE-feature transfer arm (P2018-REVE perch donor -> SEDF-REVE perch)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
RF=data/physiofm/reve_features
RS=results/phase4/reve_stack
CTX=results/phase4/context
mkdir -p "$RS/logs" results/phase4/logs
stamp() { date +%H:%M:%S; }

# ---- A. HMC stack seeds 3-7 ----
SEEDS="3 4 5 6 7" bash scripts/run_reve_stack.sh
for SEED in 3 4 5 6 7; do
  if ! grep -q "_noW_seed$SEED," "$RS/finetune.csv" 2>/dev/null; then
    echo "[$(stamp)] A: noW seed$SEED"
    "$PY" scripts/phase2_hmc_finetune.py --arch_key hmc_reve \
      --labels "$RF/hmc_reve_labels.npz" --rand_dir "$RS/pretrain/seed$SEED/rand" \
      --mode full --epochs 20 --ft_seed "$SEED" --class_weight none \
      --tag "_noW_seed$SEED" --out_csv "$RS/finetune.csv" \
      2>&1 | tee "$RS/logs/noW_seed$SEED.log" | { grep -E "RESULT.*test" || true; }
  fi
done

# ---- B. e16 multi-seed pairing rows ----
for FS in 1 2; do
  for HV in "linear _lin_e16" "context _ctx_e16"; do
    set -- $HV; HEAD=$1; TAG=$2
    if ! grep -q "${TAG}_seed$FS" "$CTX/sedf_structured.csv" 2>/dev/null; then
      echo "[$(stamp)] B: tf64 $HEAD e16 ftseed $FS"
      "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_tf64 \
        --pc_dir results/phase4/gate0/sleep_edf_tf64/seed42/pc \
        --rand_dir results/phase4/gate0/sleep_edf_tf64/seed42/rand \
        --mode full --epochs 16 --k 5 --head "$HEAD" --ft_seed "$FS" \
        --out_csv "$CTX/sedf_structured.csv" --tag "${TAG}_seed$FS" \
        2>&1 | tee "$CTX/logs/sedf_${TAG}_seed$FS.log" | { grep -E "RESULT|wrote" || true; }
    fi
  done
  if ! grep -q "_e16_seed$FS" "$RS/sedf_finetune.csv" 2>/dev/null; then
    echo "[$(stamp)] B: reve e16 ftseed $FS"
    "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_reve \
      --labels "$RF/sleep_edf_reve_labels.npz" \
      --pc_dir "$RS/sedf_pretrain/seed42/pc" --rand_dir "$RS/sedf_pretrain/seed42/rand" \
      --mode full --epochs 16 --k 5 --ft_seed "$FS" \
      --out_csv "$RS/sedf_finetune.csv" --tag "_e16_seed$FS" \
      2>&1 | tee "$RS/logs/sedf_e16_seed$FS.log" | { grep -E "RESULT|wrote" || true; }
  fi
done

# ---- C. REVE-feature transfer ----
[ -f "$RF/sleep_edf_reve_perch_labels.npz" ] || "$PY" scripts/build_perch_tf64.py \
  --base sleep_edf_reve --labels "$RF/sleep_edf_reve_labels.npz" --out_dir "$RF" \
  2>&1 | tee "$RS/logs/perch_sedf_reve.log"
[ -f "$RF/p2018_reve_perch_labels.npz" ] || "$PY" scripts/build_perch_tf64.py \
  --base p2018_reve --labels "$RF/p2018_reve_labels.npz" --out_dir "$RF" \
  2>&1 | tee "$RS/logs/perch_p2018_reve.log"
pretrain() { local DS="$1" OUT="$2" TAG="$3" EP="$4"
  [ -f "$OUT/$TAG/DONE" ] && return 0
  echo "[$(stamp)] C: pretrain $(basename "$OUT")/$TAG"
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" --p_in 1 --p_out 16 \
    --batch 4 --seed 42 --objective input --epochs "$EP" --output_dir "$OUT" --tag "$TAG" \
    2>&1 | tee "$RS/logs/pre_$(basename "$OUT")_$TAG.log" | { grep -E "epoch 60/|corpus" || true; }
  [ -f "$OUT/$TAG/DONE" ] || { echo FATAL; exit 1; }
}
pretrain "$RF/p2018_reve_perch.npz" "$RS/transfer/donor_p2018_reve_perch" pc 60
pretrain sleep_edf_reve_perch "$RS/transfer/sedf_reve_perch/seed42" pc 60
pretrain sleep_edf_reve_perch "$RS/transfer/sedf_reve_perch/seed42" rand 0
[ -f "$RS/transfer/hybrid_full/TRANSFER" ] || "$PY" scripts/make_transfer_ckpt.py --mode full \
  --donor "$RS/transfer/donor_p2018_reve_perch/pc" --target "$RS/transfer/sedf_reve_perch/seed42/rand" \
  --out "$RS/transfer/hybrid_full"
for FS in 42 1 2; do
  if ! grep -q "_tr_seed$FS" "$RS/transfer_ft.csv" 2>/dev/null; then
    echo "[$(stamp)] C: transfer FT ftseed $FS"
    "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_reve_perch \
      --labels "$RF/sleep_edf_reve_perch_labels.npz" --merge_every 2 \
      --pc_dir "$RS/transfer/sedf_reve_perch/seed42/pc" \
      --rand_dir "$RS/transfer/sedf_reve_perch/seed42/rand" \
      --arm physiofm_transfer_full "$RS/transfer/hybrid_full" \
      --mode full --epochs 8 --k 5 --ft_seed "$FS" \
      --out_csv "$RS/transfer_ft.csv" --tag "_tr_seed$FS" \
      2>&1 | tee "$RS/logs/transfer_ft_seed$FS.log" | { grep -E "RESULT|wrote" || true; }
  fi
done
echo OK > "$RS/RIGOR_DONE"
echo "[$(stamp)] RIGOR QUEUE DONE"
