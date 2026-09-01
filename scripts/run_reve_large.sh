#!/usr/bin/env bash
# EXP-0028 final arm: REVE-LARGE (408M, dim 1216) stack on HMC.
# Same protocol as the Base stack: extract -> pc/rand x 8 seeds -> e20 FT (both loss
# variants) -> frozen probe control.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahdiar/.conda/envs/xcqa/bin/python}"
RF=data/physiofm/reve_features
ROOT=results/phase4/reve_large
mkdir -p "$ROOT/logs"
stamp() { date +%H:%M:%S; }

if [ ! -f "$RF/hmc_revelarge_labels.npz" ]; then
  echo "[$(stamp)] LARGE extraction (151 recordings)"
  "$PY" scripts/build_hmc_reve.py --model brain-bzh/reve-large --batch 48 \
    2>&1 | tee "$ROOT/logs/build.log" \
    | { grep -E "loaded|/151|extracted|split recordings|wrote|Error|Trace" || true; }
  [ -f "$RF/hmc_revelarge_labels.npz" ] || { echo "FATAL: extraction"; exit 1; }
fi

for SEED in 42 1 2 3 4 5 6 7; do
  OUT="$ROOT/pretrain/seed$SEED"
  for ARM in pc rand; do
    EP=60; [ "$ARM" = rand ] && EP=0
    [ -f "$OUT/$ARM/DONE" ] && continue
    echo "[$(stamp)] pretrain seed$SEED $ARM"
    "$PY" scripts/phase2_pretrain.py --variant scratch --datasets hmc_revelarge_pretrain \
      --p_in 1 --p_out 16 --batch "${BATCH:-2}" --seed "$SEED" --objective input --epochs "$EP" \
      --output_dir "$OUT" --tag "$ARM" 2>&1 | tee "$ROOT/logs/pre_${SEED}_${ARM}.log" \
      | { grep -E "epoch 60/|corpus|params" || true; }
    [ -f "$OUT/$ARM/DONE" ] || { echo "FATAL: pretrain $SEED/$ARM"; exit 1; }
  done
  for CW in balanced none; do
    TAG="_seed$SEED"; [ "$CW" = none ] && TAG="_noW_seed$SEED"
    if ! grep -q "$TAG," "$ROOT/finetune.csv" 2>/dev/null; then
      echo "[$(stamp)] FT seed$SEED cw=$CW"
      ARMS=(--pc_dir "$OUT/pc" --rand_dir "$OUT/rand")
      [ "$CW" = none ] && ARMS=(--rand_dir "$OUT/rand")
      "$PY" scripts/phase2_hmc_finetune.py --arch_key hmc_revelarge \
        --labels "$RF/hmc_revelarge_labels.npz" "${ARMS[@]}" \
        --mode full --epochs 20 --ft_seed "$SEED" --class_weight "$CW" \
        --tag "$TAG" --out_csv "$ROOT/finetune.csv" \
        2>&1 | tee "$ROOT/logs/ft_${SEED}_${CW}.log" | { grep -E "RESULT.*test|best epoch" || true; }
    fi
  done
done

if [ ! -f "$ROOT/probe.txt" ]; then
  echo "[$(stamp)] frozen probe control (large)"
  "$PY" - << 'PYX' > "$ROOT/probe.txt" 2>&1 || { echo "probe failed (non-fatal)"; }
import sys; sys.path.insert(0, ".")
import numpy as np
from physiofm.de import load_de_archive
from physiofm.hmc import split_masks
from physiofm.sleep_edf import load_sleep_labels
trials = load_de_archive("data/physiofm/reve_features/hmc_revelarge.npz")
labels, subj, night, key = load_sleep_labels("data/physiofm/reve_features/hmc_revelarge_labels.npz")
tr_m, va_m, te_m = split_masks(subj)
X = {n: np.concatenate([trials[i].values.reshape(len(labels[i]), -1)
                        for i in range(len(trials)) if m[i]]).astype(np.float32)
     for n, m in (("tr", tr_m), ("te", te_m))}
y = {n: np.concatenate([labels[i] for i in range(len(trials)) if m[i]])
     for n, m in (("tr", tr_m), ("te", te_m))}
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
sc = StandardScaler().fit(X["tr"])
clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
clf.fit(sc.transform(X["tr"]), y["tr"])
p = clf.predict(sc.transform(X["te"]))
print(f"LARGE frozen per-epoch logreg: acc={accuracy_score(y['te'],p)*100:.2f} "
      f"bac={balanced_accuracy_score(y['te'],p)*100:.2f} kappa={cohen_kappa_score(y['te'],p):.4f} "
      f"wf1={f1_score(y['te'],p,average='weighted')*100:.2f}")
PYX
  tail -2 "$ROOT/probe.txt"
fi
echo OK > "$ROOT/QUEUE_DONE"
echo "[$(stamp)] LARGE QUEUE DONE"
