"""AutoGluon classifier-chain manager for correlated activity attributes.

This implements the ``MultilabelPredictor`` strategy from the AutoGluon
tutorial within ZSHAR's model-manager interface: one ``TabularPredictor`` is
kept per attribute, while every predictor after the first is trained with the
preceding ground-truth attributes and inferred with preceding predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from .autogluon_backend import AutoGluonBackend
from .base import classification_metrics
from .constant_backend import ConstantAttributeBackend
from .factory_helpers import (
    choose_problem_type,
    overall_attribute_metrics,
    sanitize_model_name,
)
from .serialization import MANAGER_MANIFEST, load_json, save_json


# Initial predictability prior estimated from the mean unseen-attribute
# accuracy of 60 existing AutoGluon ZSL splits (PAMAP2, USC-HAD and MHEALTH).
DEFAULT_BASE_ATTRIBUTE_ORDER = (
    "impulsiveness",
    "body_coordination",
    "motion_intensity",
    "motion_type",
    "cadence",
    "smoothness",
    "spatial_variability",
    "vertical_motion",
    "posture_variability",
    "posture_inclination",
)


class AutoGluonMultilabelModelManager:
    """Maintain an autoregressive chain of per-attribute AutoGluon models."""

    manager_type = "autogluon_multilabel"
    backend_name = "autogluon_multilabel"

    def __init__(
        self,
        attribute_cols: Optional[Sequence[str]] = None,
        **backend_config: Any,
    ) -> None:
        self.attribute_cols = list(attribute_cols or [])
        self.backend_config = dict(backend_config)
        correlation = self.backend_config.get("consider_labels_correlation", True)
        if not isinstance(correlation, bool):
            raise TypeError("consider_labels_correlation must be a boolean")
        self.consider_labels_correlation = correlation
        self.label_order = self._resolve_label_order(
            self.attribute_cols, self.backend_config.get("label_order")
        )
        self.models: dict[str, Any] = {}
        self.problem_types: dict[str, str] = {}
        self.class_counts: dict[str, dict[str, int]] = {}
        self.validation_report: dict[str, Any] = {}

    @staticmethod
    def _resolve_label_order(
        attribute_cols: Sequence[str], requested: Any = None
    ) -> list[str]:
        attributes = list(attribute_cols)
        if len(attributes) != len(set(attributes)):
            raise ValueError("attribute_cols contains duplicate labels")
        if requested is not None:
            if not isinstance(requested, (list, tuple)):
                raise TypeError("label_order must be a JSON list of attribute names")
            order = [str(label) for label in requested]
            if len(order) != len(set(order)) or set(order) != set(attributes):
                missing = sorted(set(attributes).difference(order))
                extra = sorted(set(order).difference(attributes))
                raise ValueError(
                    "label_order must contain every attribute exactly once; "
                    f"missing={missing}, extra={extra}"
                )
            return order
        known = [label for label in DEFAULT_BASE_ATTRIBUTE_ORDER if label in attributes]
        return known + [label for label in attributes if label not in known]

    @property
    def predictor_config(self) -> dict[str, Any]:
        """Configuration forwarded to each individual AutoGluon predictor."""
        return {
            key: value
            for key, value in self.backend_config.items()
            if key not in {"consider_labels_correlation", "label_order"}
        }

    def fit_all(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        Y_val: pd.DataFrame,
        root_path: Optional[str] = None,
        label_count_threshold: int = 1,
    ) -> dict[str, Any]:
        if not root_path:
            raise ValueError("AutoGluon multilabel manager requires root_path.")
        root = Path(root_path)
        root.mkdir(parents=True, exist_ok=True)
        self.attribute_cols = list(Y_train.columns)
        self.label_order = self._resolve_label_order(
            self.attribute_cols, self.backend_config.get("label_order")
        )

        train_features = X_train.copy()
        validation_features = X_val.copy()
        for position, attr in enumerate(self.label_order):
            problem_type, counts = choose_problem_type(
                Y_train[attr], label_count_threshold
            )
            self.problem_types[attr] = problem_type
            self.class_counts[attr] = {
                str(label): int(count) for label, count in counts.items()
            }
            attr_path = root / sanitize_model_name(attr)
            print(
                f"  Multilabel attribute [{position + 1}/{len(self.label_order)}: {attr}] "
                f"problem_type={problem_type}, class_counts={counts}"
            )
            if problem_type == "constant":
                model = ConstantAttributeBackend(label=attr)
            else:
                model = AutoGluonBackend(
                    label=attr,
                    model_path=str(attr_path),
                    problem_type=problem_type,
                    **self.predictor_config,
                )
            model.fit(
                train_features,
                Y_train[attr].astype(str),
                validation_features,
                Y_val[attr].astype(str),
            )
            self.models[attr] = model
            model.save(str(attr_path))

            if self.consider_labels_correlation:
                # Teacher forcing during fit follows AutoGluon's tutorial:
                # later labels condition on earlier ground-truth labels.
                train_features[attr] = Y_train[attr].astype(str).to_numpy()
                validation_features[attr] = Y_val[attr].astype(str).to_numpy()

        predictions = self.predict_all(X_val)
        per_attribute = {
            attr: classification_metrics(
                Y_val[attr].astype(str), predictions[attr].astype(str)
            )
            for attr in self.attribute_cols
        }
        exact, cosine = overall_attribute_metrics(
            Y_val[self.attribute_cols], predictions
        )
        self.validation_report = {
            "per_attribute": per_attribute,
            "exact_match": exact,
            "average_cosine_similarity": cosine,
            "label_order": self.label_order,
            "consider_labels_correlation": self.consider_labels_correlation,
            "validation_inference": "autoregressive_predicted_labels",
        }
        self.save(str(root))
        return self.validation_report

    def _working_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        working = self._working_features(X)
        predictions: dict[str, pd.Series] = {}
        for attr in self.label_order:
            prediction = self.models[attr].predict(working).astype(str)
            prediction.index = X.index
            predictions[attr] = prediction
            if self.consider_labels_correlation:
                working[attr] = prediction
        return pd.DataFrame(predictions, index=X.index)[self.attribute_cols]

    def predict_proba_all(self, X: pd.DataFrame) -> dict[str, pd.DataFrame]:
        working = self._working_features(X)
        probabilities: dict[str, pd.DataFrame] = {}
        for attr in self.label_order:
            model = self.models[attr]
            table = model.predict_proba(working)
            if table is None:
                raise RuntimeError(
                    f"AutoGluon multilabel predictor does not expose predict_proba "
                    f"for attribute {attr!r}."
                )
            normalized = table.copy()
            normalized.index = X.index
            normalized.columns = [str(column) for column in normalized.columns]
            probabilities[attr] = normalized.astype(float)
            if self.consider_labels_correlation:
                # Hard-label chaining intentionally mirrors the tutorial's
                # MultilabelPredictor._predict implementation.
                prediction = model.predict(working).astype(str)
                prediction.index = X.index
                working[attr] = prediction
        return {attr: probabilities[attr] for attr in self.attribute_cols}

    def _features_for_attribute(self, attr: str, X: pd.DataFrame) -> pd.DataFrame:
        if attr not in self.models:
            raise KeyError(attr)
        working = self._working_features(X)
        if not self.consider_labels_correlation:
            return working
        for previous in self.label_order:
            if previous == attr:
                return working
            prediction = self.models[previous].predict(working).astype(str)
            prediction.index = X.index
            working[previous] = prediction
        raise KeyError(attr)

    def get_feature_importance(
        self,
        attr: str,
        X: pd.DataFrame,
        y: pd.Series,
        num_shuffle_sets: int = 3,
        random_state: int = 42,
    ) -> pd.DataFrame:
        features = self._features_for_attribute(attr, X)
        return self.models[attr].get_feature_importance(
            features, y, num_shuffle_sets, random_state
        )

    def save(self, root_path: str) -> None:
        root = Path(root_path)
        root.mkdir(parents=True, exist_ok=True)
        model_entries: dict[str, dict[str, Any]] = {}
        for attr in self.attribute_cols:
            relative_path = sanitize_model_name(attr)
            model = self.models[attr]
            model.save(str(root / relative_path))
            model_entries[attr] = {
                "path": relative_path,
                "backend": model.backend_name,
                "problem_type": self.problem_types.get(attr),
                "class_counts": self.class_counts.get(attr, {}),
                "metadata": model.get_metadata(),
            }
        save_json(
            root / MANAGER_MANIFEST,
            {
                "format_version": 1,
                "manager_type": self.manager_type,
                "classifier_backend": self.backend_name,
                "attribute_cols": self.attribute_cols,
                "label_order": self.label_order,
                "consider_labels_correlation": self.consider_labels_correlation,
                "backend_config": self.backend_config,
                "models": model_entries,
                "problem_types": self.problem_types,
                "class_counts": self.class_counts,
                "validation_report": self.validation_report,
            },
        )

    @classmethod
    def load(cls, root_path: str) -> "AutoGluonMultilabelModelManager":
        root = Path(root_path)
        payload = load_json(root / MANAGER_MANIFEST)
        manager = cls(
            attribute_cols=payload["attribute_cols"],
            **payload.get("backend_config", {}),
        )
        saved_order = payload.get("label_order", manager.label_order)
        manager.label_order = manager._resolve_label_order(
            manager.attribute_cols, saved_order
        )
        manager.consider_labels_correlation = bool(
            payload.get(
                "consider_labels_correlation", manager.consider_labels_correlation
            )
        )
        manager.problem_types = dict(payload.get("problem_types", {}))
        manager.class_counts = dict(payload.get("class_counts", {}))
        manager.validation_report = dict(payload.get("validation_report", {}))
        for attr, entry in payload["models"].items():
            path = str(root / entry["path"])
            if entry["backend"] == "constant":
                model = ConstantAttributeBackend.load(path)
            else:
                model = AutoGluonBackend.load(path, label=attr)
            manager.models[attr] = model
        return manager

    def get_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "manager_type": self.manager_type,
            "label_order": self.label_order,
            "consider_labels_correlation": self.consider_labels_correlation,
            "models": {
                attr: self.models[attr].get_metadata() for attr in self.attribute_cols
            },
            "actual_trained_model_count": int(
                sum(
                    model.get_metadata().get("trained_model_count", 1)
                    for model in self.models.values()
                    if model.backend_name != "constant"
                )
            ),
            "constant_attribute_count": sum(
                model.backend_name == "constant" for model in self.models.values()
            ),
        }


class MultilabelPredictor(AutoGluonMultilabelModelManager):
    """Named project adapter for AutoGluon's tutorial classifier-chain pattern."""
