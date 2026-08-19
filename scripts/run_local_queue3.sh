#!/usr/bin/env bash
# Local queue 3 (after queue2): (H) Gate 0 sleep_edf_tf64 multi-seed (seeds 1-3, fine-tuned),
# (I) latent pretext diagnostics on local latent checkpoints that saved their EMA target,
# (J) latent/pc/rand frozen gains on the small datasets (emotion SEED-IV raw/smoothed, MI) for the P2 correlation.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
G0=results/phase4/gate0; G1=results/phase4/gate1; LOGS=results/phase4/logs
stamp() { date "+%H:%M:%S"; }
echo "[$(stamp)] queue3 start"
for SEED in 1 2 3; do
  ROOT="$G0" TASKS="sleep_edf_tf64" SEED=$SEED bash scripts/run_gate1_pretrain.sh > "$LOGS/gate0_pretrain_sleep_tf64_seed$SEED.log" 2>&1
  if ! grep -q "_seed$SEED" "$G0/sleep_edf_tf64/finetune.csv" 2>/dev/null; then
    echo "[$(stamp)] H: sleep tf64 fine-tune seed $SEED"
    ROOT="$G0" ARCH=sleep_edf_tf64 SEED=$SEED DO_FROZEN=0 bash scripts/run_gate1_eval.sh > "$LOGS/gate0_eval_sleep_tf64_seed$SEED.log" 2>&1
  fi
done
echo "[$(stamp)] I: latent pretext diagnostics (local)"
for M in $G1/sleep_edf/seed1/latent $G1/sleep_edf/seed2/latent $G1/sleep_edf/seed3/latent; do
  "$PY" scripts/diagnose_pretext_latent.py --dataset sleep_edf --model_dir "$M" --batch 8 >> "$LOGS/pretext_latent_local.log" 2>&1
done
for D in seed_iv_raw seed_iv bci_iv_2a; do
  "$PY" scripts/diagnose_pretext_latent.py --dataset "$D" --model_dir "$G1/$D/seed42/latent" --batch 32 >> "$LOGS/pretext_latent_local.log" 2>&1
done
for V in latent_delta latent_cos_nonorm latent_varreg latent_ema099 latent_pout4; do
  "$PY" scripts/diagnose_pretext_latent.py --dataset sleep_edf --model_dir "$G1/sleep_edf/seed42/$V" --batch 8 >> "$LOGS/pretext_latent_local.log" 2>&1
done
echo "[$(stamp)] QUEUE3 DONE"
# ---- J. small-dataset frozen gains (latent vs pc vs rand vs raw) for the P2 correlation ----
echo "[$(date +%H:%M:%S)] J: small-dataset frozen gains"
for D in seed_iv_raw seed_iv; do
  "$PY" scripts/phase2_emotion_parity.py --datasets $D --raw --pc_dir $G1/$D/seed42/pc --latent_dir $G1/$D/seed42/latent \
     --rand_dir $G1/$D/seed42/rand --out_dir $G1/$D --tag _gate1_seed42 > "$LOGS/gate1_eval_${D}_seed42.log" 2>&1
done
"$PY" scripts/phase2_bci_eval.py --raw --pc_dir $G1/bci_iv_2a/seed42/pc --latent_dir $G1/bci_iv_2a/seed42/latent \
   --rand_dir $G1/bci_iv_2a/seed42/rand --out_dir $G1/bci_iv_2a --tag _gate1_seed42 > "$LOGS/gate1_eval_bci_iv_2a_seed42.log" 2>&1
echo "[$(date +%H:%M:%S)] QUEUE3b DONE"
