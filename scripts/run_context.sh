#!/usr/bin/env bash
# EXP-0027: sequence-context head — the offline operating point.
# Same causal encoders and harnesses as today's linear-head rows; only --head changes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
ROOT="results/phase4/context"
TROOT="results/phase4/transfer"
TF=data/physiofm/tf_features
mkdir -p "$ROOT/logs"
stamp() { date +%H:%M:%S; }

# 1. SEDF-78 structured (pc / rand / nothing-else), ft seeds 42 1 2
for FS in 42 1 2; do
  if ! grep -q "_ctx_seed$FS" "$ROOT/sedf_structured.csv" 2>/dev/null; then
    echo "[$(stamp)] SEDF structured context ftseed $FS"
    "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_tf64 \
      --pc_dir results/phase4/gate0/sleep_edf_tf64/seed42/pc \
      --rand_dir results/phase4/gate0/sleep_edf_tf64/seed42/rand \
      --mode full --epochs 8 --k 5 --head context --ft_seed "$FS" \
      --out_csv "$ROOT/sedf_structured.csv" --tag "_ctx_seed$FS" \
      2>&1 | tee "$ROOT/logs/sedf_structured_seed$FS.log" | { grep -E "RESULT|wrote" || true; }
  fi
done

# 1b. SEDF structured LINEAR rows on the SAME seed42 models + ft seeds (valid pairing)
for FS in 42 1 2; do
  if ! grep -q "_lin_seed$FS" "$ROOT/sedf_structured.csv" 2>/dev/null; then
    echo "[$(stamp)] SEDF structured linear ftseed $FS (pairing rows)"
    "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_tf64 \
      --pc_dir results/phase4/gate0/sleep_edf_tf64/seed42/pc \
      --rand_dir results/phase4/gate0/sleep_edf_tf64/seed42/rand \
      --mode full --epochs 8 --k 5 --head linear --ft_seed "$FS" \
      --out_csv "$ROOT/sedf_structured.csv" --tag "_lin_seed$FS" \
      2>&1 | tee "$ROOT/logs/sedf_structured_lin_seed$FS.log" | { grep -E "RESULT|wrote" || true; }
  fi
done

# 1c. Ladder-matched context window (per-layer |i-j|<=5; 2 layers -> effective +-10 receptive field ~ SleepTransformer L=21) — isolates
#     "context machinery" from "10-40x longer context"
for FS in 42 1 2; do
  if ! grep -q "_ctxw5_seed$FS" "$ROOT/sedf_structured.csv" 2>/dev/null; then
    echo "[$(stamp)] SEDF structured context-w5 ftseed $FS"
    "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_tf64 \
      --pc_dir results/phase4/gate0/sleep_edf_tf64/seed42/pc \
      --rand_dir results/phase4/gate0/sleep_edf_tf64/seed42/rand \
      --mode full --epochs 8 --k 5 --head context --ctx_window 5 --ft_seed "$FS" \
      --out_csv "$ROOT/sedf_structured.csv" --tag "_ctxw5_seed$FS" \
      2>&1 | tee "$ROOT/logs/sedf_structured_ctxw5_seed$FS.log" | { grep -E "RESULT|wrote" || true; }
  fi
done

# 1d. Budget robustness: 16 FT epochs, seed 42 (fresh 1M head may need more than 8)
if ! grep -q "_ctx_e16_seed42" "$ROOT/sedf_structured.csv" 2>/dev/null; then
  echo "[$(stamp)] SEDF structured context e16 seed42"
  "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_tf64 \
    --pc_dir results/phase4/gate0/sleep_edf_tf64/seed42/pc \
    --rand_dir results/phase4/gate0/sleep_edf_tf64/seed42/rand \
    --mode full --epochs 16 --k 5 --head context --ft_seed 42 \
    --out_csv "$ROOT/sedf_structured.csv" --tag "_ctx_e16_seed42" \
    2>&1 | tee "$ROOT/logs/sedf_structured_ctx_e16.log" | { grep -E "RESULT|wrote" || true; }
fi

# 2. SEDF-78 perch incl. the P2018-transfer arm (the best-combo hunt), ft seeds 42 1 2
for FS in 42 1 2; do
  if ! grep -q "_ctx_seed$FS" "$ROOT/sedf_perch.csv" 2>/dev/null; then
    echo "[$(stamp)] SEDF perch context ftseed $FS"
    "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_tf64_perch \
      --labels "$TF/sleep_edf_tf64_perch_labels.npz" --merge_every 2 \
      --pc_dir "$TROOT/sleep_edf_perch/seed42/pc" --rand_dir "$TROOT/sleep_edf_perch/seed42/rand" \
      --arm physiofm_transfer_full "$TROOT/hybrid_sedf_perch_full" \
      --mode full --epochs 8 --k 5 --head context --ft_seed "$FS" \
      --out_csv "$ROOT/sedf_perch.csv" --tag "_ctx_seed$FS" \
      2>&1 | tee "$ROOT/logs/sedf_perch_seed$FS.log" | { grep -E "RESULT|wrote" || true; }
  fi
done

# 3. HMC (e20, val-kappa selection), ft seeds 42 1 2
for FS in 42 1 2; do
  if ! grep -q "_ctx_seed$FS," "$ROOT/hmc.csv" 2>/dev/null; then
    echo "[$(stamp)] HMC context ftseed $FS"
    "$PY" scripts/phase2_hmc_finetune.py \
      --pc_dir results/phase4/hmc/pretrain/seed$FS/pc --rand_dir results/phase4/hmc/pretrain/seed$FS/rand \
      --mode full --epochs 20 --head context --ft_seed "$FS" \
      --tag "_ctx_seed$FS" --out_csv "$ROOT/hmc.csv" \
      2>&1 | tee "$ROOT/logs/hmc_seed$FS.log" | { grep -E "RESULT|best epoch" || true; }
  fi
done

# 4. P2018 6ch (SleePyCo split, per-fold pretrains), seed 42
if ! grep -q "_ctx" "$ROOT/p2018.csv" 2>/dev/null; then
  echo "[$(stamp)] P2018 6ch context"
  "$PY" scripts/phase2_p2018_finetune.py --arch_key p2018_tf64 \
    --pretrain_root results/phase4/p2018/pretrain \
    --mode full --epochs 8 --head context --ft_seed 42 --tag "_ctx" --out_csv "$ROOT/p2018.csv" \
    2>&1 | tee "$ROOT/logs/p2018.log" | { grep -E "RESULT|POOLED|wrote" || true; }
fi
echo OK > "$ROOT/QUEUE_DONE"
echo "[$(stamp)] CONTEXT QUEUE DONE"
