"""Common interfaces and metrics for activity-attribute classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import warnings
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from zshar.ordinal import ordered_ordinal_labels, resolve_ordinal_values

# Backward-compatible encoding used by legacy three-level-only backends.
ORDINAL_VALUE_MAP = {"Low": 0.0, "Medium": 0.5, "High": 1.0}


def ordered_labels(values: Sequence[Any]) -> list[str]:
    """Return deterministic labels while preserving supported ordinal orders."""
    return ordered_ordinal_labels(value for value in values if not pd.isna(value))


def ordinal_mae(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    labels = ordered_labels(list(y_true) + list(y_pred))
    mapping = resolve_ordinal_values(labels)
    true_values = np.asarray([mapping[str(value)] for value in y_true], dtype=float)
    pred_values = np.asarray([mapping[str(value)] for value in y_pred], dtype=float)
    return float(np.mean(np.abs(true_values - pred_values)))


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, float]:
    true = np.asarray(y_true).astype(str)
    pred = np.asarray(y_pred).astype(str)
    observed_labels = np.unique(np.concatenate((true, pred)))
    if len(observed_labels) == 1:
        # A constant validation target predicted correctly has one class with
        # recall 1.0. sklearn returns that value but also emits a misleading
        # confusion-matrix shape warning for this valid case.
        balanced = 1.0
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            balanced = float(balanced_accuracy_score(true, pred))
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": balanced,
        "ordinal_mae": ordinal_mae(true, pred),
    }


class BaseAttributeBackend(ABC):
    """Uniform interface implemented by every single-attribute backend."""

    backend_name = "base"

    def __init__(self, label: Optional[str] = None, **config: Any) -> None:
        self.label = label
        self.config: Dict[str, Any] = dict(config)
        self.feature_names: list[str] = []

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseAttributeBackend":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        return None

    def decision_function(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def predict_score(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        probabilities = self.predict_proba(X)
        if probabilities is not None:
            return probabilities.to_numpy()
        return self.decision_function(X)

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs: Any) -> "BaseAttributeBackend":
        raise NotImplementedError

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "label": self.label,
            "feature_count": len(self.feature_names),
        }

    def get_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_shuffle_sets: int = 3,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Backend-neutral permutation importance used when no native method exists."""
        result = permutation_importance(
            self,
            X,
            y.astype(str),
            scoring=lambda estimator, features, target: f1_score(
                np.asarray(target).astype(str),
                np.asarray(estimator.predict(features)).astype(str),
                average="macro",
                zero_division=0,
            ),
            n_repeats=max(1, int(num_shuffle_sets)),
            random_state=random_state,
            n_jobs=1,
        )
        return pd.DataFrame(
            {
                "importance": result.importances_mean,
                "stddev": result.importances_std,
            },
            index=list(X.columns),
        ).sort_values("importance", ascending=False)
