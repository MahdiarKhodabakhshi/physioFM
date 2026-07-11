#!/usr/bin/env bash
# F14 driver: subject-invariance (DANN adversarial + CORAL) on DE, strict inductive
# LOSO. Tests whether targeting cross-subject invariance helps — and helps a learned
# encoder more than fixed raw-DE — in the one regime F10 showed is NOT saturated.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
EPOCHS="${EPOCHS:-40}"
DATASETS="${DATASETS:-seed seed_iv seed_v}"

"$PY" scripts/phase2_f14_invariance.py --datasets $DATASETS --epochs "$EPOCHS" \
  --out_dir results/phase3/f14

echo "F14 DONE -> results/phase3/f14/f14_invariance.csv"
