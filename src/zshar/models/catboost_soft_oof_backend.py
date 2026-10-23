"""CatBoost classifier chain driven by out-of-fold class probabilities."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .base import classification_metrics, ordered_labels
from .boosting_backends import CatBoostBackend
from .catboost_classifier_chain_backend import CatBoostClassifierChainModelManager
from .constant_backend import ConstantAttributeBackend
from .factory_helpers import (
    choose_problem_type,
    overall_attribute_metrics,
    sanitize_model_name,
)
from .serialization import MANAGER_MANIFEST, load_json, save_json


SOFT_FEATURE_PREFIX = "__zshar_catboost_soft_oof__"
SUPPORTED_UNCERTAINTY_METRICS = ("normalized_entropy",)


class CatBoostSoftOOFModelManager(CatBoostClassifierChainModelManager):
    """Autoregressive CatBoost chain with leakage-safe OOF soft features."""

    manager_type = "catboost_soft_oof"
    backend_name = "catboost_soft_oof"

    def __init__(
        self,
        attribute_cols: Optional[Sequence[str]] = None,
        **backend_config: Any,
    ) -> None:
        super().__init__(attribute_cols=attribute_cols, **backend_config)
        if not self.consider_labels_correlation:
            raise ValueError(
                "catboost_soft_oof is a correlated soft chain; "
                "consider_labels_correlation cannot be false."
            )
        self.oof_folds = int(self.backend_config.get("oof_folds", 5))
        if self.oof_folds < 2:
            raise ValueError("catboost_soft_oof requires oof_folds >= 2")
        shuffle = self.backend_config.get("oof_shuffle", True)
        if not isinstance(shuffle, bool):
            raise TypeError("oof_shuffle must be a boolean")
        self.oof_shuffle = shuffle
        self.oof_random_state = int(
            self.backend_config.get(
                "oof_random_state", self.backend_config.get("random_state", 42)
            )
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
    def model_config(self) -> dict[str, Any]:
        """Return only settings accepted by each CatBoost classifier."""
        manager_keys = {
            "chain_feature_encoding",
            "consider_labels_correlation",
            "label_order",
            "oof_folds",
            "oof_shuffle",
            "oof_random_state",
            "uncertainty_metric",
        }
        return {
            key: value
            for key, value in self.backend_config.items()
            if key not in manager_keys
        }

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
                    "Probability class schema changed between folds or inference: "
                    f"missing={missing}, extra={extra}"
                )
            normalized = normalized[expected]
        values = normalized.to_numpy(dtype=float)
        if values.shape[0] != len(index) or not np.isfinite(values).all():
            raise RuntimeError("Probability table contains missing or non-finite values.")
        if (values < -1e-8).any():
            raise RuntimeError("Probability table contains negative probabilities.")
        values = np.clip(values, 0.0, 1.0)
        row_sums = values.sum(axis=1, keepdims=True)
        if (row_sums <= 0.0).any():
            raise RuntimeError("Probability table contains a zero-mass row.")
        return pd.DataFrame(
            values / row_sums,
            index=index,
            columns=normalized.columns,
        )

    def _build_schema(self, attr: str, classes: Sequence[str]) -> dict[str, Any]:
        position = self.label_order.index(attr)
        class_names = [str(label) for label in classes]
        return {
            "classes": class_names,
            "probability_columns": [
                (
                    f"{SOFT_FEATURE_PREFIX}{position:02d}__prob_{class_index:02d}_"
                    f"{sanitize_model_name(label)}"
                )
                for class_index, label in enumerate(class_names)
            ],
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
            classes = [str(column) for column in probability_table.columns]
            self.soft_feature_schema[attr] = self._build_schema(attr, classes)
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

    def _oof_probabilities(
        self,
        attr: str,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        string_labels = labels.astype(str)
        counts = string_labels.value_counts()
        if int(counts.min()) < self.oof_folds:
            raise ValueError(
                f"Attribute {attr!r} cannot use {self.oof_folds}-fold stratified "
                f"OOF because its smallest class has only {int(counts.min())} rows."
            )
        classes = sorted(string_labels.unique().tolist())
        oof_values = np.empty((len(features), len(classes)), dtype=float)
        splitter = StratifiedKFold(
            n_splits=self.oof_folds,
            shuffle=self.oof_shuffle,
            random_state=self.oof_random_state if self.oof_shuffle else None,
        )
        for fold_index, (train_positions, holdout_positions) in enumerate(
            splitter.split(features, string_labels), start=1
        ):
            fold_train = features.iloc[train_positions].reset_index(drop=True)
            fold_labels = string_labels.iloc[train_positions].reset_index(drop=True)
            fold_holdout = features.iloc[holdout_positions].reset_index(drop=True)
            holdout_labels = string_labels.iloc[holdout_positions].reset_index(drop=True)
            fold_config = dict(self.model_config)
            fold_config["random_state"] = self.oof_random_state + fold_index
            fold_model = CatBoostBackend(label=attr, **fold_config)
            fold_model.fit(
                fold_train,
                fold_labels,
                fold_holdout,
                holdout_labels,
            )
            table = fold_model.predict_proba(fold_holdout)
            if table is None:
                raise RuntimeError(
                    f"CatBoost did not expose OOF probabilities for attribute {attr!r}."
                )
            normalized = self._normalize_probabilities(
                table,
                index=fold_holdout.index,
                classes=classes,
            )
            oof_values[holdout_positions] = normalized.to_numpy(dtype=float)
            del fold_model
            gc.collect()
        return pd.DataFrame(oof_values, index=features.index, columns=classes)

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
            raise ValueError("CatBoost soft-OOF manager requires root_path.")
        root = Path(root_path)
        root.mkdir(parents=True, exist_ok=True)
        self.attribute_cols = list(Y_train.columns)
        self.label_order = self._resolve_label_order(
            self.attribute_cols, self.backend_config.get("label_order")
        )
        self.chain_categories = {
            attr: ordered_labels(Y_train[attr].dropna().astype(str).tolist())
            for attr in self.label_order
        }
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
                f"  CatBoost soft-OOF attribute "
                f"[{position + 1}/{len(self.label_order)}: {attr}] "
                f"problem_type={problem_type}, class_counts={counts}"
            )

            train_probabilities = None
            if position < final_position and problem_type != "constant":
                train_probabilities = self._oof_probabilities(
                    attr,
                    train_features,
                    Y_train[attr],
                )

            if problem_type == "constant":
                model = ConstantAttributeBackend(label=attr)
            else:
                model = CatBoostBackend(label=attr, **self.model_config)
            model.fit(
                train_features,
                Y_train[attr].astype(str),
                validation_features,
                Y_val[attr].astype(str),
            )
            self.models[attr] = model
            model.save(str(attr_path))

            if position < final_position:
                if train_probabilities is None:
                    train_probabilities = model.predict_proba(train_features)
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
            "training_chain": "catboost_stratified_oof_probabilities",
            "validation_inference": "autoregressive_soft_probabilities",
            "oof_folds": self.oof_folds,
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
            table = self.models[attr].predict_proba(working)
            if table is None:
                raise RuntimeError(f"Attribute {attr!r} does not expose predict_proba.")
            schema = self.soft_feature_schema.get(attr)
            classes = schema["classes"] if schema else None
            normalized = self._normalize_probabilities(
                table,
                index=X.index,
                classes=classes,
            )
            probabilities[attr] = normalized
            if include_predictions:
                predictions[attr] = normalized.idxmax(axis=1).rename(attr)
            if position < final_position:
                soft = self._soft_features(attr, normalized, index=X.index)
                working = self._append_soft_features(working, soft)
        if stop_before is not None:
            raise KeyError(stop_before)
        return probabilities, predictions, working

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        _, predictions, _ = self._chain_probabilities(X, include_predictions=True)
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
                "oof_folds": self.oof_folds,
                "oof_shuffle": self.oof_shuffle,
                "oof_random_state": self.oof_random_state,
                "uncertainty_metric": self.uncertainty_metric,
                "training_chain": "catboost_stratified_oof_probabilities",
                "inference_chain": "predict_proba_probabilities",
                "hard_labels_chained": False,
            }
        )
        save_json(manifest_path, payload)

    @classmethod
    def load(cls, root_path: str) -> "CatBoostSoftOOFModelManager":
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
        manager.chain_categories = dict(payload.get("chain_categories", {}))
        manager.validation_report = dict(payload.get("validation_report", {}))
        manager.soft_feature_schema = dict(payload.get("soft_feature_schema", {}))
        expected_schema = set(manager.label_order[:-1])
        if set(manager.soft_feature_schema) != expected_schema:
            raise ValueError(
                "CatBoost soft-OOF manifest has an incomplete feature schema: "
                f"expected={sorted(expected_schema)}, "
                f"found={sorted(manager.soft_feature_schema)}"
            )
        for attr, entry in payload["models"].items():
            path = str(root / entry["path"])
            if entry["backend"] == "constant":
                model = ConstantAttributeBackend.load(path)
            else:
                model = CatBoostBackend.load(path, label=attr)
            manager.models[attr] = model
        return manager

    def get_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "manager_type": self.manager_type,
            "label_order": self.label_order,
            "consider_labels_correlation": True,
            "training_chain": "catboost_stratified_oof_probabilities",
            "inference_chain": "predict_proba_probabilities",
            "oof_folds": self.oof_folds,
            "oof_shuffle": self.oof_shuffle,
            "oof_random_state": self.oof_random_state,
            "uncertainty_metric": self.uncertainty_metric,
            "hard_labels_chained": False,
            "soft_feature_schema": self.soft_feature_schema,
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
