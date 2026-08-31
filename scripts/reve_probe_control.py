#!/usr/bin/env python3
"""EXP-0028 control: frozen REVE per-epoch features + balanced logistic regression on
the fixed HMC split — REVE WITHOUT any sequence context. The stacked model's gain over
this row (and over REVE's own published fine-tune) is attributable to sequence modeling."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physiofm.de import load_de_archive
from physiofm.hmc import split_masks
from physiofm.sleep_edf import load_sleep_labels

trials = load_de_archive("data/physiofm/reve_features/hmc_reve.npz")
labels, subj, night, key = load_sleep_labels("data/physiofm/reve_features/hmc_reve_labels.npz")
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
clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
clf.fit(sc.transform(X["tr"]), y["tr"])
p = clf.predict(sc.transform(X["te"]))
print(f"REVE frozen per-epoch logreg (fixed split, test n={len(p)}): "
      f"acc={accuracy_score(y['te'], p)*100:.2f} bac={balanced_accuracy_score(y['te'], p)*100:.2f} "
      f"kappa={cohen_kappa_score(y['te'], p):.4f} "
      f"mf1={f1_score(y['te'], p, average='macro')*100:.2f} "
      f"wf1={f1_score(y['te'], p, average='weighted')*100:.2f}")
