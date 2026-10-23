"""AutoGluon classifier chain driven by soft bagging OOF features.

For attribute ``A_i`` the training input is the original sensor feature frame
plus the native AutoGluon bagging OOF class probabilities and normalized
predictive entropy of every earlier attribute.  At inference time the same
columns are populated with ``predict_proba`` outputs.  Hard attribute labels
are predictions only; they are never chained into a later predictor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .autogluon_backend import AutoGluonBackend
from .autogluon_multilabel_backend import AutoGluonMultilabelModelManager
from .base import classification_metrics
from .constant_backend import ConstantAttributeBackend
from .factory_helpers import (
    choose_problem_type,
    overall_attribute_metrics,
    sanitize_model_name,
)
from .serialization import MANAGER_MANIFEST, load_json, save_json


SOFT_FEATURE_PREFIX = "__zshar_soft_oof__"
SUPPORTED_UNCERTAINTY_METRICS = ("normalized_entropy",)
DEFAULT_SOFT_OOF_TIME_LIMIT = 300
DEFAULT_SOFT_OOF_NUM_BAG_FOLDS = 2
# AutoGluon may combine float32 model outputs into values one ULP outside
# [0, 1] (for example, -1.1920929e-07 and 1.000000119).  Treat only this
# small boundary drift as numerical noise; materially invalid probabilities
# must still fail loudly.
PROBABILITY_BOUNDARY_TOLERANCE = 1e-6


class AutoGluonSoftOOFModelManager(AutoGluonMultilabelModelManager):
    """Autoregressive soft-probability chain with leakage-safe OOF training."""

    manager_type = "autogluon_soft_oof"
    backend_name = "autogluon_soft_oof"

    def __init__(
        self,
        attribute_cols: Optional[Sequence[str]] = None,
        **backend_config: Any,
    ) -> None:
        # Soft chaining needs bagged out-of-fold probabilities.  Keep the
        # leakage-safe minimum as the default while allowing experiments to
        # request more folds or a different per-attribute time budget.
        backend_config.setdefault("time_limit", DEFAULT_SOFT_OOF_TIME_LIMIT)
        backend_config.setdefault(
            "num_bag_folds", DEFAULT_SOFT_OOF_NUM_BAG_FOLDS
        )
        super().__init__(attribute_cols=attribute_cols, **backend_config)
        if not self.consider_labels_correlation:
            raise ValueError(
                "autogluon_soft_oof is a correlated soft chain; "
                "consider_labels_correlation cannot be false."
            )
        self.uncertainty_metric = str(
            self.backend_config.get("uncertainty_metric", "normalized_entropy")
        ).strip().lower()
        if self.uncertainty_metric not in SUPPORTED_UNCERTAINTY_METRICS:
            raise ValueError(
                f"uncertainty_metric must be one of {SUPPORTED_UNCERTAINTY_METRICS}"
            )
        self.soft_feature_schema: dict[str, dict[str, Any]] = {}

    @property
    def predictor_config(self) -> dict[str, Any]:
        """Only forward actual TabularPredictor settings to each attribute."""
        manager_keys = {
            "consider_labels_correlation",
            "label_order",
            "uncertainty_metric",
        }
        return {
            key: value
            for key, value in self.backend_config.items()
            if key not in manager_keys
        }

    def _validate_bagging(self) -> None:
        folds = int(self.predictor_config.get("num_bag_folds", 0))
        if folds < 2:
            raise ValueError(
                "autogluon_soft_oof requires num_bag_folds >= 2 because its "
                "training chain must use native bagging OOF probabilities."
            )

    @staticmethod
    def _normalize_probabilities(
        table: pd.DataFrame,
        *,
        index: pd.Index,
        classes: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        normalized = table.copy()
        if isinstance(normalized, pd.Series):
            normalized = normalized.to_frame()
        normalized.columns = [str(column) for column in normalized.columns]
        normalized = normalized.reindex(index)
        if classes is not None:
            expected = [str(label) for label in classes]
            missing = [label for label in expected if label not in normalized.columns]
            extra = [label for label in normalized.columns if label not in expected]
            if missing or extra:
                raise RuntimeError(
                    "Probability class schema changed between fit and inference: "
                    f"missing={missing}, extra={extra}"
                )
            normalized = normalized[expected]
        values = normalized.to_numpy(dtype=float)
        if values.shape[0] != len(index) or not np.isfinite(values).all():
            raise RuntimeError("Probability table contains missing or non-finite values.")
        minimum = float(values.min())
        maximum = float(values.max())
        if (
            minimum < -PROBABILITY_BOUNDARY_TOLERANCE
            or maximum > 1.0 + PROBABILITY_BOUNDARY_TOLERANCE
        ):
            raise RuntimeError(
                "Probability table contains values outside [0, 1] beyond "
                f"the numerical tolerance: min={minimum}, max={maximum}, "
                f"tolerance={PROBABILITY_BOUNDARY_TOLERANCE}."
            )
        values = np.clip(values, 0.0, 1.0)
        row_sums = values.sum(axis=1, keepdims=True)
        if (row_sums <= 0.0).any():
            raise RuntimeError("Probability table contains a zero-mass row.")
        values = values / row_sums
        return pd.DataFrame(values, index=index, columns=normalized.columns)

    def _build_schema(self, attr: str, classes: Sequence[str]) -> dict[str, Any]:
        position = self.label_order.index(attr)
        class_names = [str(label) for label in classes]
        probability_columns = [
            (
                f"{SOFT_FEATURE_PREFIX}{position:02d}__prob_{class_index:02d}_"
                f"{sanitize_model_name(label)}"
            )
            for class_index, label in enumerate(class_names)
        ]
        return {
            "classes": class_names,
            "probability_columns": probability_columns,
            "uncertainty_column": (
                f"{SOFT_FEATURE_PREFIX}{position:02d}__uncertainty_"
                f"{self.uncertainty_metric}"
            ),
        }

    def _soft_features(
        self,
        attr: str,
        probability_table: pd.DataFrame,
        *,
        index: pd.Index,
        fit_schema: bool = False,
    ) -> pd.DataFrame:
        if fit_schema:
            raw_classes = [str(column) for column in probability_table.columns]
            self.soft_feature_schema[attr] = self._build_schema(attr, raw_classes)
        if attr not in self.soft_feature_schema:
            raise RuntimeError(f"Missing soft feature schema for attribute {attr!r}.")
        schema = self.soft_feature_schema[attr]
        probabilities = self._normalize_probabilities(
            probability_table,
            index=index,
            classes=schema["classes"],
        )
        values = probabilities.to_numpy(dtype=float)
        if values.shape[1] <= 1:
            uncertainty = np.zeros(len(values), dtype=float)
        else:
            safe = np.clip(values, np.finfo(float).tiny, 1.0)
            entropy = -np.sum(values * np.log(safe), axis=1)
            uncertainty = entropy / np.log(values.shape[1])
        soft = pd.DataFrame(
            values,
            index=index,
            columns=list(schema["probability_columns"]),
        )
        soft[str(schema["uncertainty_column"])] = uncertainty
        return soft

    @staticmethod
    def _append_soft_features(
        working: pd.DataFrame, soft: pd.DataFrame
    ) -> pd.DataFrame:
        collisions = sorted(set(working.columns).intersection(soft.columns))
        if collisions:
            raise ValueError(f"Soft-chain feature name collision: {collisions}")
        return pd.concat((working, soft), axis=1)

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
            raise ValueError("AutoGluon soft-OOF manager requires root_path.")
        self._validate_bagging()
        root = Path(root_path)
        root.mkdir(parents=True, exist_ok=True)
        self.attribute_cols = list(Y_train.columns)
        self.label_order = self._resolve_label_order(
            self.attribute_cols, self.backend_config.get("label_order")
        )
        self.soft_feature_schema = {}

        train_features = X_train.copy()
        validation_features = X_val.copy()
        final_position = len(self.label_order) - 1
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
                f"  Soft-OOF attribute [{position + 1}/{len(self.label_order)}: {attr}] "
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

            if position < final_position:
                if problem_type == "constant":
                    train_probabilities = model.predict_proba(train_features)
                else:
                    train_probabilities = model.predict_proba_oof(
                        train_features,
                        Y_train[attr].astype(str),
                    )
                validation_probabilities = model.predict_proba(validation_features)
                if train_probabilities is None or validation_probabilities is None:
                    raise RuntimeError(
                        f"Attribute {attr!r} does not expose probabilities required "
                        "by the soft OOF chain."
                    )
                train_soft = self._soft_features(
                    attr,
                    train_probabilities,
                    index=X_train.index,
                    fit_schema=True,
                )
                validation_soft = self._soft_features(
                    attr,
                    validation_probabilities,
                    index=X_val.index,
                )
                train_features = self._append_soft_features(
                    train_features, train_soft
                )
                validation_features = self._append_soft_features(
                    validation_features, validation_soft
                )

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
            "consider_labels_correlation": True,
            "training_chain": "autogluon_bagging_oof_probabilities",
            "validation_inference": "autoregressive_soft_probabilities",
            "uncertainty_metric": self.uncertainty_metric,
            "hard_labels_chained": False,
        }
        self.save(str(root))
        return self.validation_report

    def _chain_probabilities(
        self,
        X: pd.DataFrame,
        *,
        include_predictions: bool,
        stop_before: Optional[str] = None,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], pd.DataFrame]:
        working = X.copy()
        probabilities: dict[str, pd.DataFrame] = {}
        predictions: dict[str, pd.Series] = {}
        final_position = len(self.label_order) - 1
        for position, attr in enumerate(self.label_order):
            if attr == stop_before:
                return probabilities, predictions, working
            model = self.models[attr]
            table = model.predict_proba(working)
            if table is None:
                raise RuntimeError(
                    f"Attribute {attr!r} does not expose predict_proba."
                )
            schema = self.soft_feature_schema.get(attr)
            classes = schema["classes"] if schema else None
            normalized = self._normalize_probabilities(
                table, index=X.index, classes=classes
            )
            probabilities[attr] = normalized
            if include_predictions:
                prediction = model.predict(working).astype(str)
                prediction.index = X.index
                predictions[attr] = prediction
            if position < final_position:
                soft = self._soft_features(attr, normalized, index=X.index)
                working = self._append_soft_features(working, soft)
        if stop_before is not None:
            raise KeyError(stop_before)
        return probabilities, predictions, working

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        _, predictions, _ = self._chain_probabilities(
            X, include_predictions=True
        )
        return pd.DataFrame(predictions, index=X.index)[self.attribute_cols]

    def predict_proba_all(self, X: pd.DataFrame) -> dict[str, pd.DataFrame]:
        probabilities, _, _ = self._chain_probabilities(
            X, include_predictions=False
        )
        return {attr: probabilities[attr] for attr in self.attribute_cols}

    def _features_for_attribute(self, attr: str, X: pd.DataFrame) -> pd.DataFrame:
        if attr not in self.models:
            raise KeyError(attr)
        _, _, working = self._chain_probabilities(
            X,
            include_predictions=False,
            stop_before=attr,
        )
        return working

    def save(self, root_path: str) -> None:
        super().save(root_path)
        manifest_path = Path(root_path) / MANAGER_MANIFEST
        payload = load_json(manifest_path)
        payload.update(
            {
                "format_version": 2,
                "soft_feature_schema": self.soft_feature_schema,
                "uncertainty_metric": self.uncertainty_metric,
                "training_chain": "autogluon_bagging_oof_probabilities",
                "inference_chain": "predict_proba_probabilities",
                "hard_labels_chained": False,
            }
        )
        save_json(manifest_path, payload)

    @classmethod
    def load(cls, root_path: str) -> "AutoGluonSoftOOFModelManager":
        root = Path(root_path)
        payload = load_json(root / MANAGER_MANIFEST)
        manager = cls(
            attribute_cols=payload["attribute_cols"],
            **payload.get("backend_config", {}),
        )
        manager.label_order = manager._resolve_label_order(
            manager.attribute_cols,
            payload.get("label_order", manager.label_order),
        )
        manager.problem_types = dict(payload.get("problem_types", {}))
        manager.class_counts = dict(payload.get("class_counts", {}))
        manager.validation_report = dict(payload.get("validation_report", {}))
        manager.soft_feature_schema = dict(payload.get("soft_feature_schema", {}))
        if len(manager.label_order) > 1 and not manager.soft_feature_schema:
            raise ValueError(
                f"Soft-OOF manifest lacks soft_feature_schema: {root / MANAGER_MANIFEST}"
            )
        for attr, entry in payload["models"].items():
            path = str(root / entry["path"])
            if entry["backend"] == "constant":
                model = ConstantAttributeBackend.load(path)
            else:
                model = AutoGluonBackend.load(path, label=attr)
            manager.models[attr] = model
        return manager

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "training_chain": "autogluon_bagging_oof_probabilities",
                "inference_chain": "predict_proba_probabilities",
                "uncertainty_metric": self.uncertainty_metric,
                "hard_labels_chained": False,
                "soft_feature_schema": self.soft_feature_schema,
            }
        )
        return metadata
