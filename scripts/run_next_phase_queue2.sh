#!/usr/bin/env bash
# Second sequential GPU queue (runs after run_next_phase_queue.sh prints QUEUE DONE):
#   F. Gate 1 multi-seed on sleep DE: seeds 1,2,3 x {pc, latent, rand} pretraining + fine-tuning
#      (paired per-fold tests need >= 3 pretraining seeds; sleep FT was single-seed before).
#   G. Gate 1 latent-objective robustness sweep on sleep DE (single seed): 4 variants, fine-tuned.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
G1=results/phase4/gate1; LOGS=results/phase4/logs; mkdir -p "$LOGS"
stamp() { date "+%H:%M:%S"; }
until grep -q "QUEUE DONE" "$LOGS/queue.log" 2>/dev/null; do sleep 120; done
echo "[$(stamp)] queue2 start"

# ---------------- F. multi-seed sleep DE ----------------
for SEED in 1 2 3; do
  ROOT="$G1" TASKS="sleep_edf" SEED=$SEED bash scripts/run_gate1_pretrain.sh > "$LOGS/gate1_pretrain_sleep_seed$SEED.log" 2>&1
  if ! grep -q "_seed$SEED" "$G1/sleep_edf/finetune.csv" 2>/dev/null; then
    echo "[$(stamp)] F: sleep DE fine-tune seed $SEED"
    ARCH=sleep_edf SEED=$SEED DO_FROZEN=0 bash scripts/run_gate1_eval.sh > "$LOGS/gate1_eval_sleep_edf_seed$SEED.log" 2>&1
  fi
done

# ---------------- G. latent-objective variants (sleep DE, seed 42) ----------------
V="$G1/sleep_edf/seed42"
declare -A VARIANTS=(
  [latent_cos_nonorm]="--objective latent --target_norm none --latent_loss cos"
  [latent_varreg]="--objective latent --var_reg 1.0"
  [latent_ema099]="--objective latent --ema 0.99"
  [latent_pout4]="--objective latent --p_out 4"
  [latent_delta]="--objective latent --target_mode delta"
)
ARMS=""
for NAME in latent_delta latent_cos_nonorm latent_varreg latent_ema099 latent_pout4; do
  if [ ! -f "$V/$NAME/DONE" ]; then
    echo "[$(stamp)] G: pretrain $NAME"
    POUT=16; [ "$NAME" = latent_pout4 ] && POUT=4
    "$PY" scripts/phase2_pretrain.py --variant scratch --datasets sleep_edf --p_in 1 --p_out $POUT --batch 16 --seed 42 \
       --epochs 60 ${VARIANTS[$NAME]/--p_out 4/} --output_dir "$V" --tag "$NAME" > "$LOGS/gate1_pretrain_sleep_$NAME.log" 2>&1
  fi
  ARMS="$ARMS --arm $NAME $V/$NAME"
done
if ! grep -q "latent_cos_nonorm" "$G1/sleep_edf/finetune.csv" 2>/dev/null; then
  echo "[$(stamp)] G: fine-tune latent variants"
  "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf $ARMS --mode full --epochs 8 --k 5 \
     --out_csv "$G1/sleep_edf/finetune.csv" --tag "_seed42" > "$LOGS/gate1_eval_sleep_variants.log" 2>&1
fi
echo "[$(stamp)] QUEUE2 DONE"
