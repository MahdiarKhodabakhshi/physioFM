#!/usr/bin/env bash
# Sequential GPU queue for the next-phase plan (docs/NEXT_PHASE_PLAN.md), Gates 0-3.
# Idempotent: every stage skips if its output exists, so it can be re-run after interruption.
# Stage list (in order):
#   A. Gate 1 evals on CHB-MIT DE (frozen + fine-tuned)          -> EXP-0021
#   B. Gate 0 ladder on sleep_edf_tf64 (pc/latent/rand + evals)   -> EXP-0020
#   C. Gate 2 raw sleep: pretrain input-PC / latent / rand (20 ep, max_len 3000) + perch latent/rand
#      then frozen + fine-tuned evals (tokens_per_epoch=150, chunks of 20 epochs)  -> EXP-0022
#   D. Gate 3: bidirectional twin (rand) + streaming eval on sleep DE   -> EXP-0023
#   E. Gate 0 ladder on chbmit_tf64 (only if GATE0_SEIZURE=1)
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
G1=results/phase4/gate1; G0=results/phase4/gate0; G2=results/phase4/gate2; G3=results/phase4/gate3
LOGS=results/phase4/logs; mkdir -p "$LOGS" "$G0" "$G2" "$G3"
RAW_EPOCHS="${RAW_EPOCHS:-20}"
GATE0_SEIZURE="${GATE0_SEIZURE:-0}"

wait_for() { while [ ! -f "$1" ]; do sleep 60; done; }
stamp() { date "+%H:%M:%S"; }

echo "[$(stamp)] queue start"

# ---------------- A. Gate 1 CHB-MIT evals (needs the three DE seizure models) ----------------
wait_for "$G1/chbmit/seed42/rand/DONE"; wait_for "$G1/chbmit/seed42/latent/DONE"; wait_for "$G1/chbmit/seed42/pc/DONE"
if [ ! -f "$G1/chbmit/finetune.csv" ]; then
  echo "[$(stamp)] A: gate1 chbmit evals"
  ARCH=chbmit SEED=42 bash scripts/run_gate1_eval.sh > "$LOGS/gate1_eval_chbmit_seed42.log" 2>&1
fi

# ---------------- B. Gate 0 sleep tf64 ladder ----------------
if [ ! -f "$G0/sleep_edf_tf64/finetune.csv" ]; then
  echo "[$(stamp)] B: gate0 sleep_edf_tf64 pretrain + evals"
  ROOT="$G0" TASKS="sleep_edf_tf64" bash scripts/run_gate1_pretrain.sh > "$LOGS/gate0_pretrain_sleep_tf64.log" 2>&1
  ROOT="$G0" ARCH=sleep_edf_tf64 SEED=42 bash scripts/run_gate1_eval.sh > "$LOGS/gate0_eval_sleep_tf64.log" 2>&1
  # dimension-matched control on tf64 (raw tf64 -> random 256-d projection vs PC vs latent)
  "$PY" scripts/diagnose_encoder.py --task sleep --arch_key sleep_edf_tf64 \
      --pc_dir "$G0/sleep_edf_tf64/seed42/pc" --rand_dir "$G0/sleep_edf_tf64/seed42/rand" \
      --latent_dir "$G0/sleep_edf_tf64/seed42/latent" --out_csv "$G0/diagnose_encoder_tf64.csv" \
      > "$LOGS/gate0_diag_sleep_tf64.log" 2>&1 || echo "diagnose_encoder tf64 failed (see log)"
fi

# ---------------- C. Gate 2 raw sleep ----------------
wait_for data/physiofm/raw_tokens/sleep_edf_raw200ms.npz
R="$G2/sleep_edf_raw/seed42"
for ARM in latent pc rand; do
  [ -f "$R/$ARM/DONE" ] && continue
  case "$ARM" in
    pc)     EXTRA="--objective input  --epochs $RAW_EPOCHS" ;;
    latent) EXTRA="--objective latent --epochs $RAW_EPOCHS" ;;
    rand)   EXTRA="--objective input  --epochs 0" ;;
  esac
  echo "[$(stamp)] C: raw sleep pretrain $ARM"
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets sleep_edf_raw --p_in 1 --p_out 16 \
     --batch 8 --max_len 3000 --seed 42 $EXTRA --output_dir "$R" --tag "$ARM" > "$LOGS/gate2_pretrain_raw_$ARM.log" 2>&1
done
if [ ! -f "$G2/sleep_edf_raw/finetune.csv" ]; then
  echo "[$(stamp)] C: raw sleep evals"
  "$PY" scripts/phase2_f13_sleep.py --arch_key sleep_edf_raw --labels data/physiofm/raw_tokens/sleep_edf_raw200ms_labels.npz \
     --pc_dir "$R/pc" --latent_dir "$R/latent" --rand_dir "$R/rand" --classifiers logreg --k 5 \
     --max_len 3000 --batch_size 8 --out_dir "$G2/sleep_edf_raw" --tag "_frozen_seed42" > "$LOGS/gate2_eval_raw_frozen.log" 2>&1
  "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_raw --labels data/physiofm/raw_tokens/sleep_edf_raw200ms_labels.npz \
     --pc_dir "$R/pc" --latent_dir "$R/latent" --rand_dir "$R/rand" --mode full --epochs 3 --batch 8 --max_len 20 --k 5 \
     --out_csv "$G2/sleep_edf_raw/finetune.csv" --tag "_seed42" > "$LOGS/gate2_eval_raw_ft.log" 2>&1
fi
# per-electrode ablation (BrainGPT-style decomposition), same objective/protocol
wait_for data/physiofm/raw_tokens/sleep_edf_raw200ms_perch.npz
P="$G2/sleep_edf_raw_perch/seed42"
for ARM in latent rand; do
  [ -f "$P/$ARM/DONE" ] && continue
  case "$ARM" in
    latent) EXTRA="--objective latent --epochs $RAW_EPOCHS" ;;
    rand)   EXTRA="--objective input  --epochs 0" ;;
  esac
  echo "[$(stamp)] C: raw sleep PER-CHANNEL pretrain $ARM"
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets sleep_edf_raw_perch --p_in 1 --p_out 16 \
     --batch 16 --max_len 3000 --seed 42 $EXTRA --output_dir "$P" --tag "$ARM" > "$LOGS/gate2_pretrain_perch_$ARM.log" 2>&1
done
if [ ! -f "$G2/sleep_edf_raw_perch/finetune.csv" ]; then
  echo "[$(stamp)] C: raw sleep PER-CHANNEL evals"
  "$PY" scripts/phase2_f13_sleep.py --arch_key sleep_edf_raw_perch --labels data/physiofm/raw_tokens/sleep_edf_raw200ms_perch_labels.npz \
     --latent_dir "$P/latent" --rand_dir "$P/rand" --classifiers logreg --k 5 --merge_every 2 \
     --max_len 3000 --batch_size 16 --out_dir "$G2/sleep_edf_raw_perch" --tag "_frozen_seed42" > "$LOGS/gate2_eval_perch_frozen.log" 2>&1
  "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_raw_perch --labels data/physiofm/raw_tokens/sleep_edf_raw200ms_perch_labels.npz \
     --latent_dir "$P/latent" --rand_dir "$P/rand" --mode full --epochs 3 --batch 4 --max_len 20 --k 5 --merge_every 2 \
     --out_csv "$G2/sleep_edf_raw_perch/finetune.csv" --tag "_seed42" > "$LOGS/gate2_eval_perch_ft.log" 2>&1
fi

# ---------------- D. Gate 3 streaming (sleep DE) ----------------
if [ ! -f "$G3/streaming.csv" ]; then
  echo "[$(stamp)] D: gate3 bidirectional twin + streaming eval"
  [ -f "$G1/sleep_edf/seed42/rand_bidir/DONE" ] || "$PY" scripts/phase2_pretrain.py --variant scratch --datasets sleep_edf \
      --p_in 1 --p_out 16 --batch 16 --seed 42 --epochs 0 --causal 0 --output_dir "$G1/sleep_edf/seed42" --tag rand_bidir > "$LOGS/gate3_pretrain_bidir.log" 2>&1
  "$PY" scripts/gate3_streaming_eval.py --arm causal_rand "$G1/sleep_edf/seed42/rand" --arm bidir_rand "$G1/sleep_edf/seed42/rand_bidir" \
      --arm causal_latent "$G1/sleep_edf/seed42/latent" --arm causal_pc "$G1/sleep_edf/seed42/pc" \
      --out_csv "$G3/streaming.csv" > "$LOGS/gate3_streaming.log" 2>&1
fi

# ---------------- E. Gate 0 seizure tf64 ladder (optional; ~3 h) ----------------
if [ "$GATE0_SEIZURE" = 1 ] && [ ! -f "$G0/chbmit_tf64/finetune.csv" ]; then
  echo "[$(stamp)] E: gate0 chbmit_tf64 pretrain + evals"
  ROOT="$G0" TASKS="chbmit_tf64" bash scripts/run_gate1_pretrain.sh > "$LOGS/gate0_pretrain_chbmit_tf64.log" 2>&1
  ROOT="$G0" ARCH=chbmit_tf64 SEED=42 bash scripts/run_gate1_eval.sh > "$LOGS/gate0_eval_chbmit_tf64.log" 2>&1
fi
echo "[$(stamp)] QUEUE DONE"
