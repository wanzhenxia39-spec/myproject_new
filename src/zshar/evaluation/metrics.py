from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_summary(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    true = np.asarray(y_true).astype(str)
    pred = np.asarray(y_pred).astype(str)
    # sklearn's balanced_accuracy_score delegates to confusion_matrix and
    # warns when both arrays contain the same single label.  That case is a
    # perfectly valid constant-attribute evaluation and its recall is 1.0.
    observed_labels = np.unique(np.concatenate((true, pred)))
    if len(observed_labels) == 1:
        balanced_accuracy = 1.0
    else:
        balanced_accuracy = float(balanced_accuracy_score(true, pred))
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": balanced_accuracy,
    }


def harmonic_mean(left: float, right: float) -> float:
    return float(2 * left * right / (left + right)) if left + right > 0 else 0.0
