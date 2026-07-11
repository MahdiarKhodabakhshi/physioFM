"""Canonical Phase 2 evaluation harness.

Every Phase 2 model (raw-DE ceilings, PhysioFM-S from-scratch, PhysioFM-S
TimesFM-init) is scored through these functions so the comparison ladder uses
identical splits, seeds, metrics, and classifier protocol.

Granularity is **segment-level** (one feature vector per DE time window),
directly comparable to PC-SSL which classifies individual windows. The
subject-dependent splits reuse the Phase 1 fold masks in
``physiofm.embedding_evaluation`` so Phase 1 and Phase 2 numbers are aligned.

A "features provider" is any object exposing per-segment arrays:
    X       : (N, D) float features
    subject : (N,)   int subject id
    trial   : (N,)   int trial id
    label   : (N,)   int class label
plus a ``trials`` list of the source ``DETrial`` objects (used only to enumerate
the per-trial split membership, exactly as Phase 1 did).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .de import DETrial
from .embedding_evaluation import seed_iv_fold_mask, seed_v_fold_mask

SEED = 42

FOLD_MASK: dict[str, Callable] = {
    "seed_v": seed_v_fold_mask,
    "seed_iv": seed_iv_fold_mask,
}
CHANCE: dict[str, float] = {"seed_v": 20.0, "seed_iv": 25.0, "seed": 33.3}
N_FOLDS = 3


def base_dataset(dataset: str) -> str:
    """Normalize a dataset key to its split/label family.

    Follow-up archives such as ``seed_iv_raw`` (un-smoothed DE) share SEED-IV's
    trials, labels, and folds, so they resolve to ``seed_iv`` for the fold mask
    and chance level while keeping a distinct archive on disk.
    """
    return dataset.lower().replace("-", "_").replace("_raw", "")


@dataclass
class SegmentFeatures:
    X: np.ndarray
    subject: np.ndarray
    trial: np.ndarray
    label: np.ndarray
    trials: list[DETrial]


# ---------------------------------------------------------------------------
# Classifier factories. Each returns a *fresh* unfitted sklearn pipeline.
# ---------------------------------------------------------------------------
def make_logreg():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )


def make_linear_svm():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    return make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", random_state=SEED),
    )


def make_rbf_svm():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
    )


def make_mlp():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=300,
            early_stopping=True,
            random_state=SEED,
        ),
    )


def make_mlp_bal():
    """F4 matched head: 2-hidden-layer MLP with class balancing + per-fold val.

    ``sklearn``'s ``MLPClassifier`` has no ``class_weight``, so we wrap it in a
    deterministic balanced-oversampling estimator and use ``early_stopping`` (a
    per-fold validation split) for regularization. The identical head is applied
    to raw-DE, PhysioFM-S (pretrained) and PhysioFM-S (random-init) features so
    the comparison isolates the representation, not the classifier.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), _BalancedMLP())


class _BalancedMLP:
    """sklearn-style estimator: balanced oversampling + 2-hidden-layer MLP."""

    def __init__(self, hidden=(256, 128), alpha=1e-3, max_iter=300, seed=SEED):
        self.hidden = hidden
        self.alpha = alpha
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X, y):
        from sklearn.neural_network import MLPClassifier

        rng = np.random.default_rng(self.seed)
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        target = counts.max()
        idx = []
        for c in classes:
            ci = np.where(y == c)[0]
            extra = rng.choice(ci, size=target - ci.size, replace=True) if target > ci.size else np.array([], dtype=int)
            idx.append(np.concatenate([ci, extra]))
        idx = rng.permutation(np.concatenate(idx))
        self.clf_ = MLPClassifier(
            hidden_layer_sizes=self.hidden, alpha=self.alpha, max_iter=self.max_iter,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
            random_state=self.seed,
        )
        self.clf_.fit(X[idx], y[idx])
        return self

    def predict(self, X):
        return self.clf_.predict(X)


CLASSIFIERS: dict[str, Callable] = {
    "logreg": make_logreg,
    "linear_svm": make_linear_svm,
    "rbf_svm": make_rbf_svm,
    "mlp": make_mlp,
    "mlp_bal": make_mlp_bal,
}


# ---------------------------------------------------------------------------
# Raw-DE segment features (the E0 ceiling control).
# ---------------------------------------------------------------------------
def build_raw_de_segments(trials: list[DETrial], instance_norm: bool = False) -> SegmentFeatures:
    """One 310-d raw-DE feature per time window, with id/label arrays.

    If ``instance_norm`` is set, each (channel,band) series is z-scored over time
    within its trial — the TimesFM-style RevIN that the Phase 1 diagnostics
    identified as the signal killer. This is the C5 control.
    """
    X, subj, trid, lab = [], [], [], []
    for t in trials:
        if t.label is None:
            continue
        v = np.asarray(t.values, dtype=np.float32)  # (T, C, B)
        flat = v.reshape(v.shape[0], -1)  # (T, C*B)
        if instance_norm:
            mu = flat.mean(axis=0, keepdims=True)
            sd = flat.std(axis=0, keepdims=True) + 1e-6
            flat = (flat - mu) / sd
        X.append(flat)
        subj.append(np.full(flat.shape[0], t.subject, dtype=np.int64))
        trid.append(np.full(flat.shape[0], t.trial, dtype=np.int64))
        lab.append(np.full(flat.shape[0], t.label, dtype=np.int64))
    return SegmentFeatures(
        X=np.concatenate(X).astype(np.float32),
        subject=np.concatenate(subj),
        trial=np.concatenate(trid),
        label=np.concatenate(lab),
        trials=[t for t in trials if t.label is not None],
    )


# ---------------------------------------------------------------------------
# Subject-dependent evaluation (PC-SSL splits) — the primary protocol.
# ---------------------------------------------------------------------------
def subject_dependent_eval(
    feats: SegmentFeatures,
    dataset: str,
    classifier: str = "logreg",
) -> dict[str, object]:
    """Per-subject, per-fold segment classification using the PC-SSL splits.

    Mirrors the Phase 1 segment classify loop exactly: the fold mask selects
    train/test *trials* per subject, then segments are assigned by trial id.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    dataset = base_dataset(dataset)
    fold_fn = FOLD_MASK[dataset]
    factory = CLASSIFIERS[classifier]

    y = LabelEncoder().fit_transform(feats.label)
    tsubj = np.array([t.subject for t in feats.trials])
    ttrial = np.array([t.trial for t in feats.trials])

    rows = []
    for s in sorted(set(tsubj.tolist())):
        for f in range(N_FOLDS):
            tr_t, te_t = fold_fn(tsubj, ttrial, s, f)
            train_trials = set(ttrial[tr_t].tolist())
            test_trials = set(ttrial[te_t].tolist())
            a = (feats.subject == s) & np.isin(feats.trial, list(train_trials))
            b = (feats.subject == s) & np.isin(feats.trial, list(test_trials))
            if a.sum() == 0 or b.sum() == 0:
                continue
            clf = factory()
            clf.fit(feats.X[a], y[a])
            pred = clf.predict(feats.X[b])
            rows.append(
                {
                    "subject": int(s),
                    "fold": f + 1,
                    "train_segments": int(a.sum()),
                    "test_segments": int(b.sum()),
                    "accuracy": float(accuracy_score(y[b], pred)),
                    "macro_f1": float(f1_score(y[b], pred, average="macro", zero_division=0)),
                }
            )

    accs = np.array([r["accuracy"] for r in rows], dtype=np.float64)
    f1s = np.array([r["macro_f1"] for r in rows], dtype=np.float64)
    return {
        "dataset": dataset,
        "classifier": classifier,
        "protocol": "subject_dependent",
        "runs": len(rows),
        "accuracy_mean": float(accs.mean()) if len(accs) else float("nan"),
        "accuracy_std": float(accs.std()) if len(accs) else float("nan"),
        "macro_f1_mean": float(f1s.mean()) if len(f1s) else float("nan"),
        "macro_f1_std": float(f1s.std()) if len(f1s) else float("nan"),
        "chance": CHANCE.get(dataset, float("nan")),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Subject-independent LOSO evaluation — the stretch generalization protocol.
# ---------------------------------------------------------------------------
def loso_eval(
    feats: SegmentFeatures,
    dataset: str,
    classifier: str = "logreg",
    max_train: int = 8000,
    seed: int = SEED,
) -> dict[str, object]:
    """Leave-one-subject-out segment classification.

    Training sets are large (all-but-one subject), so they are randomly
    subsampled to ``max_train`` segments per fold for tractable, comparable fits.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    dataset = base_dataset(dataset)
    factory = CLASSIFIERS[classifier]
    y = LabelEncoder().fit_transform(feats.label)
    rng = np.random.default_rng(seed)

    rows = []
    for s in sorted(set(feats.subject.tolist())):
        train = np.where(feats.subject != s)[0]
        test = np.where(feats.subject == s)[0]
        if train.size == 0 or test.size == 0:
            continue
        if train.size > max_train:
            train = rng.choice(train, size=max_train, replace=False)
        clf = factory()
        clf.fit(feats.X[train], y[train])
        pred = clf.predict(feats.X[test])
        rows.append(
            {
                "subject": int(s),
                "train_segments": int(train.size),
                "test_segments": int(test.size),
                "accuracy": float(accuracy_score(y[test], pred)),
                "macro_f1": float(f1_score(y[test], pred, average="macro", zero_division=0)),
            }
        )

    accs = np.array([r["accuracy"] for r in rows], dtype=np.float64)
    f1s = np.array([r["macro_f1"] for r in rows], dtype=np.float64)
    return {
        "dataset": dataset,
        "classifier": classifier,
        "protocol": "loso",
        "runs": len(rows),
        "accuracy_mean": float(accs.mean()) if len(accs) else float("nan"),
        "accuracy_std": float(accs.std()) if len(accs) else float("nan"),
        "macro_f1_mean": float(f1s.mean()) if len(f1s) else float("nan"),
        "macro_f1_std": float(f1s.std()) if len(f1s) else float("nan"),
        "chance": CHANCE.get(dataset, float("nan")),
        "rows": rows,
    }
