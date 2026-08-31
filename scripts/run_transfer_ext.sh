#!/usr/bin/env bash
# EXP-0026 extension: robustness of the positive SEDF perch full-transfer result.
# (a) FT seeds 1,2 for rand/pc/transfer (donor seed 42 fixed);
# (b) independent donor pretrain seed 1 -> new hybrid -> FT seed 1.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="results/phase4/transfer"
TF=data/physiofm/tf_features
stamp() { date +%H:%M:%S; }

for FS in 1 2; do
  if ! grep -q "_ftseed$FS" "$ROOT/sedf_perch_ft.csv" 2>/dev/null; then
    echo "[$(stamp)] perch FT seed $FS (rand/pc/transfer)"
    "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_tf64_perch \
      --labels "$TF/sleep_edf_tf64_perch_labels.npz" --merge_every 2 \
      --pc_dir "$ROOT/sleep_edf_perch/seed42/pc" --rand_dir "$ROOT/sleep_edf_perch/seed42/rand" \
      --arm physiofm_transfer_full "$ROOT/hybrid_sedf_perch_full" \
      --mode full --epochs 8 --k 5 --ft_seed "$FS" --out_csv "$ROOT/sedf_perch_ft.csv" --tag "_ftseed$FS" \
      2>&1 | tee "$ROOT/logs/ft_sedf_perch_ftseed$FS.log" | { grep -E "RESULT|wrote" || true; }
  fi
done

for S in 1; do
  if [ ! -f "$ROOT/donor_p2018_perch_seed$S/pc/DONE" ]; then
    echo "[$(stamp)] donor perch pretrain seed $S"
    "$PY" scripts/phase2_pretrain.py --variant scratch --datasets p2018_tf64_perch --p_in 1 --p_out 16 \
      --batch 16 --seed $S --objective input --epochs 60 --output_dir "$ROOT/donor_p2018_perch_seed$S" --tag pc \
      2>&1 | tee "$ROOT/logs/pretrain_donor_perch_seed$S.log" | { grep -E "epoch (1|30|60)/|corpus" || true; }
    [ -f "$ROOT/donor_p2018_perch_seed$S/pc/DONE" ] || { echo FATAL; exit 1; }
  fi
  for ARM in pc rand; do
    EP=60; [ "$ARM" = rand ] && EP=0
    [ -f "$ROOT/sleep_edf_perch/seed$S/$ARM/DONE" ] || "$PY" scripts/phase2_pretrain.py --variant scratch \
      --datasets sleep_edf_tf64_perch --p_in 1 --p_out 16 --batch 16 --seed $S --objective input \
      --epochs $EP --output_dir "$ROOT/sleep_edf_perch/seed$S" --tag $ARM \
      2>&1 | tee "$ROOT/logs/pretrain_sedf_perch_seed${S}_$ARM.log" | { grep -E "epoch 60/|corpus" || true; }
  done
  [ -f "$ROOT/hybrid_sedf_perch_full_seed$S/TRANSFER" ] || "$PY" scripts/make_transfer_ckpt.py --mode full \
    --donor "$ROOT/donor_p2018_perch_seed$S/pc" --target "$ROOT/sleep_edf_perch/seed$S/rand" \
    --out "$ROOT/hybrid_sedf_perch_full_seed$S"
  if ! grep -q "_dseed$S" "$ROOT/sedf_perch_ft.csv" 2>/dev/null; then
    echo "[$(stamp)] perch FT donor-seed $S"
    "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_tf64_perch \
      --labels "$TF/sleep_edf_tf64_perch_labels.npz" --merge_every 2 \
      --pc_dir "$ROOT/sleep_edf_perch/seed$S/pc" --rand_dir "$ROOT/sleep_edf_perch/seed$S/rand" \
      --arm physiofm_transfer_full "$ROOT/hybrid_sedf_perch_full_seed$S" \
      --mode full --epochs 8 --k 5 --ft_seed "$S" --out_csv "$ROOT/sedf_perch_ft.csv" --tag "_dseed$S" \
      2>&1 | tee "$ROOT/logs/ft_sedf_perch_dseed$S.log" | { grep -E "RESULT|wrote" || true; }
  fi
done
echo OK > "$ROOT/EXT_DONE"
echo "[$(stamp)] TRANSFER EXT DONE"
