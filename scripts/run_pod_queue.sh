#!/usr/bin/env bash
# Pod-side sequential GPU queue for the next-phase plan (H100 80 GB, 64 cores).
# Stages moved here from run_next_phase_queue.sh (C/D/E) so the local 20 GB card can run the
# multi-seed / variant sweeps in parallel:
#   P1. Gate 2 raw sleep structured tokens: pc / latent / rand pretraining (RAW_EPOCHS) + frozen + FT evals
#   P2. Gate 2 per-electrode ablation: latent / rand + evals (--merge_every 2)
#   P3. Gate 3 streaming eval on sleep DE (needs results/phase4/gate1/sleep_edf/seed42/* copied from local)
#   P4. Gate 0-E chbmit tf64 ladder: pc / latent / rand + frozen + FT evals
#   P5. latent pretext diagnostics on every latent checkpoint present
# Idempotent via DONE sentinels / output CSVs. Usage (on the pod): PY=python nohup bash scripts/run_pod_queue.sh > results/phase4/logs/pod_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PY="${PY:-python}"
G0=results/phase4/gate0; G1=results/phase4/gate1; G2=results/phase4/gate2; G3=results/phase4/gate3
LOGS=results/phase4/logs; mkdir -p "$LOGS" "$G0" "$G1" "$G2" "$G3"
RAW_EPOCHS="${RAW_EPOCHS:-10}"
RAW_BATCH="${RAW_BATCH:-32}"
stamp() { date "+%H:%M:%S"; }
wait_for() { while [ ! -f "$1" ]; do sleep 30; done; }
echo "[$(stamp)] pod queue start"

# ---------------- P0. Gate 1 CHB-MIT evals (OOM-killed twice on the 62 GB local box) ----------------
if [ ! -f "$G1/chbmit/finetune.csv" ] && [ -f "$G1/chbmit/seed42/rand/model.pt" ]; then
  echo "[$(stamp)] P0: gate1 chbmit frozen + fine-tuned evals"
  ARCH=chbmit SEED=42 bash scripts/run_gate1_eval.sh > "$LOGS/gate1_eval_chbmit_seed42.log" 2>&1
fi

# ---------------- P1. Gate 2 raw sleep (structured tokens) ----------------
wait_for data/physiofm/raw_tokens/sleep_edf_raw200ms.npz
R="$G2/sleep_edf_raw/seed42"
for ARM in pc latent rand; do
  [ -f "$R/$ARM/DONE" ] && continue
  case "$ARM" in
    pc)     EXTRA="--objective input  --epochs $RAW_EPOCHS" ;;
    latent) EXTRA="--objective latent --epochs $RAW_EPOCHS" ;;
    rand)   EXTRA="--objective input  --epochs 0" ;;
  esac
  echo "[$(stamp)] P1: raw sleep pretrain $ARM"
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets sleep_edf_raw --p_in 1 --p_out 16 \
     --batch "$RAW_BATCH" --max_len 3000 --seed 42 $EXTRA --output_dir "$R" --tag "$ARM" > "$LOGS/gate2_pretrain_raw_$ARM.log" 2>&1
done
if [ ! -f "$G2/sleep_edf_raw/finetune.csv" ]; then
  echo "[$(stamp)] P1: raw sleep evals"
  "$PY" scripts/phase2_f13_sleep.py --arch_key sleep_edf_raw --labels data/physiofm/raw_tokens/sleep_edf_raw200ms_labels.npz \
     --pc_dir "$R/pc" --latent_dir "$R/latent" --rand_dir "$R/rand" --classifiers logreg --k 5 \
     --max_len 3000 --batch_size 32 --out_dir "$G2/sleep_edf_raw" --tag "_frozen_seed42" > "$LOGS/gate2_eval_raw_frozen.log" 2>&1
  "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_raw --labels data/physiofm/raw_tokens/sleep_edf_raw200ms_labels.npz \
     --pc_dir "$R/pc" --latent_dir "$R/latent" --rand_dir "$R/rand" --mode full --epochs 3 --batch 16 --max_len 20 --k 5 \
     --out_csv "$G2/sleep_edf_raw/finetune.csv" --tag "_seed42" > "$LOGS/gate2_eval_raw_ft.log" 2>&1
fi

# ---------------- P2. per-electrode ablation ----------------
wait_for data/physiofm/raw_tokens/sleep_edf_raw200ms_perch.npz
P="$G2/sleep_edf_raw_perch/seed42"
for ARM in latent rand; do
  [ -f "$P/$ARM/DONE" ] && continue
  case "$ARM" in
    latent) EXTRA="--objective latent --epochs $RAW_EPOCHS" ;;
    rand)   EXTRA="--objective input  --epochs 0" ;;
  esac
  echo "[$(stamp)] P2: raw sleep PER-CHANNEL pretrain $ARM"
  "$PY" scripts/phase2_pretrain.py --variant scratch --datasets sleep_edf_raw_perch --p_in 1 --p_out 16 \
     --batch $((RAW_BATCH * 2)) --max_len 3000 --seed 42 $EXTRA --output_dir "$P" --tag "$ARM" > "$LOGS/gate2_pretrain_perch_$ARM.log" 2>&1
done
if [ ! -f "$G2/sleep_edf_raw_perch/finetune.csv" ]; then
  echo "[$(stamp)] P2: raw sleep PER-CHANNEL evals"
  "$PY" scripts/phase2_f13_sleep.py --arch_key sleep_edf_raw_perch --labels data/physiofm/raw_tokens/sleep_edf_raw200ms_perch_labels.npz \
     --latent_dir "$P/latent" --rand_dir "$P/rand" --classifiers logreg --k 5 --merge_every 2 \
     --max_len 3000 --batch_size 64 --out_dir "$G2/sleep_edf_raw_perch" --tag "_frozen_seed42" > "$LOGS/gate2_eval_perch_frozen.log" 2>&1
  "$PY" scripts/phase2_sleep_finetune.py --arch_key sleep_edf_raw_perch --labels data/physiofm/raw_tokens/sleep_edf_raw200ms_perch_labels.npz \
     --latent_dir "$P/latent" --rand_dir "$P/rand" --mode full --epochs 3 --batch 8 --max_len 20 --k 5 --merge_every 2 \
     --out_csv "$G2/sleep_edf_raw_perch/finetune.csv" --tag "_seed42" > "$LOGS/gate2_eval_perch_ft.log" 2>&1
fi

# ---------------- P3. Gate 3 streaming (sleep DE) ----------------
if [ ! -f "$G3/streaming.csv" ] && [ -f "$G1/sleep_edf/seed42/pc/model.pt" ]; then
  echo "[$(stamp)] P3: gate3 bidirectional twin + streaming eval"
  [ -f "$G1/sleep_edf/seed42/rand_bidir/DONE" ] || "$PY" scripts/phase2_pretrain.py --variant scratch --datasets sleep_edf \
      --p_in 1 --p_out 16 --batch 16 --seed 42 --epochs 0 --causal 0 --output_dir "$G1/sleep_edf/seed42" --tag rand_bidir > "$LOGS/gate3_pretrain_bidir.log" 2>&1
  "$PY" scripts/gate3_streaming_eval.py --arm causal_rand "$G1/sleep_edf/seed42/rand" --arm bidir_rand "$G1/sleep_edf/seed42/rand_bidir" \
      --arm causal_latent "$G1/sleep_edf/seed42/latent" --arm causal_pc "$G1/sleep_edf/seed42/pc" \
      --out_csv "$G3/streaming.csv" > "$LOGS/gate3_streaming.log" 2>&1
fi

# ---------------- P4. Gate 0-E chbmit tf64 ladder ----------------
wait_for data/physiofm/tf_features/chbmit_tf64.npz
if [ ! -f "$G0/chbmit_tf64/finetune.csv" ]; then
  echo "[$(stamp)] P4: gate0 chbmit_tf64 pretrain + evals"
  ROOT="$G0" TASKS="chbmit_tf64" bash scripts/run_gate1_pretrain.sh > "$LOGS/gate0_pretrain_chbmit_tf64.log" 2>&1
  ROOT="$G0" ARCH=chbmit_tf64 SEED=42 bash scripts/run_gate1_eval.sh > "$LOGS/gate0_eval_chbmit_tf64.log" 2>&1
fi

# ---------------- P5. latent pretext diagnostics ----------------
for D in sleep_edf chbmit seed_iv_raw seed_iv bci_iv_2a; do
  M="$G1/$D/seed42/latent"
  [ -f "$M/model.pt" ] || continue
  grep -q "$M" "$G1/diagnose_pretext_latent.csv" 2>/dev/null && continue
  "$PY" scripts/diagnose_pretext_latent.py --dataset "$D" --model_dir "$M" --batch 8 >> "$LOGS/pretext_latent.log" 2>&1
done
for D in sleep_edf_tf64:$G0/sleep_edf_tf64/seed42/latent sleep_edf_raw:$G2/sleep_edf_raw/seed42/latent chbmit_tf64:$G0/chbmit_tf64/seed42/latent; do
  K="${D%%:*}"; M="${D#*:}"
  [ -f "$M/model.pt" ] || continue
  grep -q "$M" "$G1/diagnose_pretext_latent.csv" 2>/dev/null && continue
  "$PY" scripts/diagnose_pretext_latent.py --dataset "$K" --model_dir "$M" --batch 8 --max_len 3000 >> "$LOGS/pretext_latent.log" 2>&1
done
echo "[$(stamp)] POD QUEUE DONE"
