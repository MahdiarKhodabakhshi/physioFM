#!/usr/bin/env bash
# EXP-0026: cross-corpus transfer — the literature's one reliably-positive pretraining
# regime (SleepTransformer SHHS->SEDF +3.5 acc). Donors pretrained on ALL 994 P2018
# records; targets Sleep-EDF-78 (full-weight perch + trunk) and HMC (trunk).
# Every transfer arm is judged against its matched random-init AND same-corpus-PC arm
# under the identical fine-tuning harness.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="results/phase4/transfer"
SEED=42
mkdir -p "$ROOT/logs"
stamp() { date +%H:%M:%S; }
TF=data/physiofm/tf_features

# A. per-electrode archives
[ -f "$TF/sleep_edf_tf64_perch_labels.npz" ] || "$PY" scripts/build_perch_tf64.py --base sleep_edf_tf64 --labels data/physiofm/de_features/sleep_edf_labels.npz 2>&1 | tee "$ROOT/logs/perch_sedf.log"
# (no hmc perch build: unused by this queue, and slicing the full 151-subject archive
#  would be a pretraining-leakage footgun for future HMC perch arms)
[ -f "$TF/p2018_tf64_perch_labels.npz" ]  || "$PY" scripts/build_perch_tf64.py --base p2018_tf64 --labels p2018_labels.npz 2>&1 | tee "$ROOT/logs/perch_p2018.log"

pretrain() { # dataset outdir tag epochs
  local DS="$1" OUT="$2" TAG="$3" EP="$4"
  [ -f "$OUT/$TAG/DONE" ] && { echo "skip $OUT/$TAG"; return 0; }
  echo "[$(stamp)] pretrain $DS -> $OUT/$TAG (epochs $EP)"
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets "$DS" --p_in 1 --p_out 16 \
    --batch 16 --seed $SEED --objective input --epochs "$EP" --output_dir "$OUT" --tag "$TAG" \
    2>&1 | tee "$ROOT/logs/pretrain_$(basename "$OUT")_$TAG.log" | { grep -E "epoch (1|30|60)/|corpus|params" || true; }
  [ -f "$OUT/$TAG/DONE" ] || { echo "FATAL: $OUT/$TAG no DONE"; exit 1; }
}

# B. donors (whole 994-record corpus — target test subjects live in OTHER corpora)
pretrain p2018_tf64        "$ROOT/donor_p2018"        pc 60
pretrain p2018_tf64_perch  "$ROOT/donor_p2018_perch"  pc 60

# C. target perch baselines (SEDF; corpus-wide pretrain = the gate-0 sleep convention)
pretrain sleep_edf_tf64_perch "$ROOT/sleep_edf_perch/seed$SEED" pc 60
pretrain sleep_edf_tf64_perch "$ROOT/sleep_edf_perch/seed$SEED" rand 0

# D. hybrid transfer checkpoints
[ -f "$ROOT/hybrid_sedf_perch_full/TRANSFER" ] || "$PY" scripts/make_transfer_ckpt.py --mode full \
  --donor "$ROOT/donor_p2018_perch/pc" --target "$ROOT/sleep_edf_perch/seed$SEED/rand" \
  --out "$ROOT/hybrid_sedf_perch_full"
[ -f "$ROOT/hybrid_sedf_trunk/TRANSFER" ] || "$PY" scripts/make_transfer_ckpt.py --mode trunk \
  --donor "$ROOT/donor_p2018/pc" --target "results/phase4/gate0/sleep_edf_tf64/seed42/rand" \
  --out "$ROOT/hybrid_sedf_trunk"
[ -f "$ROOT/hybrid_hmc_trunk/TRANSFER" ] || "$PY" scripts/make_transfer_ckpt.py --mode trunk \
  --donor "$ROOT/donor_p2018/pc" --target "results/phase4/hmc/pretrain/seed42/rand" \
  --out "$ROOT/hybrid_hmc_trunk"

# E. fine-tuned evaluations (identical harnesses; transfer vs matched rand vs same-corpus pc)
if ! grep -q "physiofm_transfer_trunk" "$ROOT/sedf_structured_ft.csv" 2>/dev/null; then
  echo "[$(stamp)] FT sleep_edf_tf64 structured (rand/pc/trunk-transfer)"
  "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_tf64 \
    --pc_dir results/phase4/gate0/sleep_edf_tf64/seed42/pc \
    --rand_dir results/phase4/gate0/sleep_edf_tf64/seed42/rand \
    --arm physiofm_transfer_trunk "$ROOT/hybrid_sedf_trunk" \
    --mode full --epochs 8 --k 5 --out_csv "$ROOT/sedf_structured_ft.csv" --tag "_seed$SEED" \
    2>&1 | tee "$ROOT/logs/ft_sedf_structured.log" | { grep -E "RESULT|wrote" || true; }
fi
if ! grep -q "physiofm_transfer_full" "$ROOT/sedf_perch_ft.csv" 2>/dev/null; then
  echo "[$(stamp)] FT sleep_edf_tf64_perch (rand/pc/full-transfer, merge 2)"
  "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_tf64_perch \
    --labels "$TF/sleep_edf_tf64_perch_labels.npz" --merge_every 2 \
    --pc_dir "$ROOT/sleep_edf_perch/seed$SEED/pc" --rand_dir "$ROOT/sleep_edf_perch/seed$SEED/rand" \
    --arm physiofm_transfer_full "$ROOT/hybrid_sedf_perch_full" \
    --mode full --epochs 8 --k 5 --out_csv "$ROOT/sedf_perch_ft.csv" --tag "_seed$SEED" \
    2>&1 | tee "$ROOT/logs/ft_sedf_perch.log" | { grep -E "RESULT|wrote" || true; }
fi
if ! grep -q "physiofm_transfer_trunk" "$ROOT/hmc_ft.csv" 2>/dev/null; then
  echo "[$(stamp)] FT hmc structured e20 (rand/pc/trunk-transfer)"
  "$PY" scripts/phase2_hmc_finetune.py \
    --pc_dir results/phase4/hmc/pretrain/seed42/pc --rand_dir results/phase4/hmc/pretrain/seed42/rand \
    --arm physiofm_transfer_trunk "$ROOT/hybrid_hmc_trunk" \
    --mode full --epochs 20 --ft_seed $SEED --tag "_transfer_seed$SEED" --out_csv "$ROOT/hmc_ft.csv" \
    2>&1 | tee "$ROOT/logs/ft_hmc.log" | { grep -E "RESULT|best epoch|split:" || true; }
fi
echo OK > "$ROOT/QUEUE_DONE"
echo "[$(stamp)] TRANSFER QUEUE DONE"
