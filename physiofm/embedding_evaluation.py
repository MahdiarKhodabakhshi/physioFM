"""Evaluation helpers for Phase 1 trial embeddings."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def seed_v_fold_mask(subject: np.ndarray, trial: np.ndarray, subject_id: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    subj = subject == subject_id
    train_trials = set()
    start = fold * 5
    for session_start in (0, 15, 30):
        train_trials.update(range(session_start + start, session_start + start + 5))
    train = subj & np.isin(trial, list(train_trials))
    test = subj & ~np.isin(trial, list(train_trials))
    return train, test


def seed_iv_fold_mask(subject: np.ndarray, trial: np.ndarray, subject_id: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    subj = subject == subject_id
    session_start = fold * 24
    train_trials = np.arange(session_start, session_start + 16)
    test_trials = np.arange(session_start + 16, session_start + 24)
    return subj & np.isin(trial, train_trials), subj & np.isin(trial, test_trials)


def classify_trial_embeddings(
    embeddings_path: str | Path,
    output_csv: str | Path,
    dataset: str,
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    with np.load(embeddings_path, allow_pickle=True) as data:
        x = data["embeddings"]
        subject = data["subject"]
        trial = data["trial"]
        labels = data["label"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    dataset = dataset.lower().replace("-", "_")
    subjects = sorted(int(item) for item in np.unique(subject))
    folds = range(3)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject_id in subjects:
        for fold in folds:
            if dataset == "seed_v":
                train_mask, test_mask = seed_v_fold_mask(subject, trial, subject_id, fold)
            elif dataset == "seed_iv":
                train_mask, test_mask = seed_iv_fold_mask(subject, trial, subject_id, fold)
            else:
                raise ValueError("Trial-level Phase 1 classification is implemented for SEED-IV/V")

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            )
            clf.fit(x[train_mask], y[train_mask])
            pred = clf.predict(x[test_mask])
            acc = accuracy_score(y[test_mask], pred)
            f1 = f1_score(y[test_mask], pred, average="macro", zero_division=0)
            rows.append(
                {
                    "subject": subject_id,
                    "fold": fold + 1,
                    "train_trials": int(train_mask.sum()),
                    "test_trials": int(test_mask.sum()),
                    "accuracy": float(acc),
                    "macro_f1": float(f1),
                }
            )

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "fold", "train_trials", "test_trials", "accuracy", "macro_f1"],
        )
        writer.writeheader()
        writer.writerows(rows)

    accs = np.array([row["accuracy"] for row in rows], dtype=np.float64)
    f1s = np.array([row["macro_f1"] for row in rows], dtype=np.float64)
    return {
        "runs": float(len(rows)),
        "accuracy_mean": float(accs.mean()) if len(accs) else float("nan"),
        "macro_f1_mean": float(f1s.mean()) if len(f1s) else float("nan"),
    }
