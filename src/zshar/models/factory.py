"""Dataset-independent backend factory and attribute-model managers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from .autogluon_backend import AutoGluonBackend
from .autogluon_multilabel_backend import MultilabelPredictor
from .autogluon_soft_oof_backend import AutoGluonSoftOOFModelManager
from .base import classification_metrics
from .boosting_backends import CatBoostBackend, LightGBMBackend, XGBoostBackend
from .catboost_classifier_chain_backend import CatBoostClassifierChainModelManager
from .catboost_soft_oof_backend import CatBoostSoftOOFModelManager
from .constant_backend import CONSTANT_MODEL_FILE, ConstantAttributeBackend
from .generic_classifier_chain_backend import GenericClassifierChainModelManager
from .factory_helpers import (
    choose_problem_type,
    overall_attribute_metrics,
    sanitize_model_name,
)
from .independent_mlp_backend import IndependentMLPBackend
from .logistic_regression_chain_backend import LogisticRegressionChainModelManager
from .multitask_mlp_backend import MultiTaskAttributeModelManager
from .native_neural_backends import NeuralNetFastAIBackend, NeuralNetTorchBackend
from .ordinal_backends import CORALMLPBackend, OrdinalLogisticBackend
from .serialization import BACKEND_METADATA, MANAGER_MANIFEST, load_json, save_json
from .sklearn_backends import LinearSVMBackend, LogisticRegressionBackend
from .tree_ensemble_backends import ExtraTreesBackend, RandomForestBackend


CLASSIFIER_BACKENDS = (
    "autogluon",
    "autogluon_multilabel",
    "autogluon_soft_oof",
    "lightgbm",
    "xgboost",
    "catboost",
    "catboost_classifier_chain",
    "catboost_soft_oof",
    "generic_classifier_chain",
    "logistic_regression",
    "logistic_regression_chain",
    "linear_svm",
    "extra_trees",
    "random_forest",
    "neuralnet_fastai",
    "neuralnet_torch",
    "ordinal_logistic",
    "coral_mlp",
    "mlp_independent",
    "mlp_multitask",
)


BACKEND_CLASSES = {
    "autogluon": AutoGluonBackend,
    "lightgbm": LightGBMBackend,
    "xgboost": XGBoostBackend,
    "catboost": CatBoostBackend,
    "logistic_regression": LogisticRegressionBackend,
    "linear_svm": LinearSVMBackend,
    "extra_trees": ExtraTreesBackend,
    "random_forest": RandomForestBackend,
    "neuralnet_fastai": NeuralNetFastAIBackend,
    "neuralnet_torch": NeuralNetTorchBackend,
    "ordinal_logistic": OrdinalLogisticBackend,
    "coral_mlp": CORALMLPBackend,
    "mlp_independent": IndependentMLPBackend,
    "constant": ConstantAttributeBackend,
}


def _overall_metrics(truth: pd.DataFrame, prediction: pd.DataFrame) -> tuple[float, float]:
    # Backward-compatible private alias used by the independent manager.
    return overall_attribute_metrics(truth, prediction)


def create_attribute_backend(
    backend_name: str,
    label: str,
    config: Mapping[str, Any],
    model_path: Optional[str] = None,
    problem_type: Optional[str] = None,
):
    if backend_name not in BACKEND_CLASSES or backend_name == "constant":
        raise ValueError(f"Unsupported independent backend: {backend_name}")
    kwargs = dict(config)
    if backend_name == "autogluon":
        kwargs.update({"model_path": str(model_path or ""), "problem_type": problem_type})
    return BACKEND_CLASSES[backend_name](label=label, **kwargs)


class IndependentAttributeModelManager:
    manager_type = "independent"

    def __init__(
        self,
        backend_name: str,
        attribute_cols: Optional[Sequence[str]] = None,
        backend_config: Optional[Mapping[str, Any]] = None,
    ):
        if backend_name == "mlp_multitask":
            raise ValueError("Use MultiTaskAttributeModelManager for mlp_multitask.")
        self.backend_name = backend_name
        self.attribute_cols = list(attribute_cols) if attribute_cols is not None else []
        self.backend_config = dict(backend_config or {})
        raw_label_order = self.backend_config.pop("label_order", None)
        if raw_label_order is not None and not isinstance(raw_label_order, (list, tuple)):
            raise TypeError("label_order must be a JSON list of attribute names")
        self.label_order = (
            [str(attribute) for attribute in raw_label_order]
            if raw_label_order is not None
            else list(self.attribute_cols)
        )
        if self.attribute_cols and (
            len(self.label_order) != len(set(self.label_order))
            or set(self.label_order) != set(self.attribute_cols)
        ):
            raise ValueError("label_order must contain every attribute exactly once")
        if self.label_order:
            self.attribute_cols = list(self.label_order)
        raw_feature_columns = self.backend_config.pop("attribute_feature_columns", {})
        if raw_feature_columns is None:
            raw_feature_columns = {}
        if not isinstance(raw_feature_columns, Mapping):
            raise ValueError("attribute_feature_columns must be an attribute-to-columns mapping")
        self.attribute_feature_columns = {
            str(attribute): [str(column) for column in columns]
            for attribute, columns in raw_feature_columns.items()
        }
        self.models: dict[str, Any] = {}
        self.problem_types: dict[str, str] = {}
        self.class_counts: dict[str, dict[str, int]] = {}
        self.validation_report: dict[str, Any] = {}

    def _attribute_features(self, X: pd.DataFrame, attr: str) -> pd.DataFrame:
        """Return exactly the sensor columns bound to one attribute.

        An empty mapping preserves the historical all-feature behavior.  With a
        mapping enabled, every attribute must have a nonempty, duplicate-free
        list and inference must provide all columns used during training.
        """
        if not self.attribute_feature_columns:
            return X
        if attr not in self.attribute_feature_columns:
            raise ValueError(f"No feature subset is configured for attribute {attr!r}")
        columns = self.attribute_feature_columns[attr]
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(
                f"Feature subset for attribute {attr!r} must be nonempty and duplicate-free"
            )
        missing = [column for column in columns if column not in X.columns]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} bound features for attribute {attr!r}: "
                f"{missing[:10]}"
            )
        return X.loc[:, columns]

    def fit_all(self, X_train, Y_train, X_val, Y_val, root_path=None, label_count_threshold=1):
        if not root_path:
            raise ValueError("Independent model manager requires root_path for serialization.")
        root = Path(root_path)
        root.mkdir(parents=True, exist_ok=True)
        training_attributes = list(Y_train.columns)
        if self.label_order:
            if (
                len(self.label_order) != len(set(self.label_order))
                or set(self.label_order) != set(training_attributes)
            ):
                raise ValueError("label_order must contain every trained attribute exactly once")
            self.attribute_cols = list(self.label_order)
        else:
            self.attribute_cols = training_attributes
            self.label_order = list(training_attributes)
        extra = sorted(set(self.attribute_feature_columns).difference(self.attribute_cols))
        missing = sorted(set(self.attribute_cols).difference(self.attribute_feature_columns))
        if self.attribute_feature_columns and (extra or missing):
            raise ValueError(
                "attribute_feature_columns must cover the trained schema exactly: "
                f"missing={missing}, extra={extra}"
            )
        per_attribute = {}
        for attr in self.attribute_cols:
            problem_type, counts = choose_problem_type(Y_train[attr], label_count_threshold)
            self.problem_types[attr] = problem_type
            self.class_counts[attr] = {str(label): int(count) for label, count in counts.items()}
            attr_path = root / sanitize_model_name(attr)
            print(f"  Attribute [{attr}] problem_type={problem_type}, class_counts={counts}")
            if problem_type == "constant":
                model = ConstantAttributeBackend(label=attr)
            else:
                model = create_attribute_backend(
                    self.backend_name,
                    attr,
                    self.backend_config,
                    model_path=str(attr_path),
                    problem_type=problem_type,
                )
            model.fit(
                self._attribute_features(X_train, attr),
                Y_train[attr],
                self._attribute_features(X_val, attr),
                Y_val[attr],
            )
            self.models[attr] = model
            prediction = model.predict(self._attribute_features(X_val, attr))
            per_attribute[attr] = classification_metrics(Y_val[attr].astype(str), prediction.astype(str))
            model.save(str(attr_path))
        predictions = self.predict_all(X_val)
        exact, cosine = _overall_metrics(Y_val[self.attribute_cols], predictions)
        self.validation_report = {
            "per_attribute": per_attribute,
            "exact_match": exact,
            "average_cosine_similarity": cosine,
        }
        self.save(str(root))
        return self.validation_report

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                attr: self.models[attr].predict(self._attribute_features(X, attr))
                for attr in self.attribute_cols
            },
            index=X.index,
            columns=self.attribute_cols,
        )

    def predict_proba_all(self, X: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Return one class-probability table per attribute."""
        probabilities: dict[str, pd.DataFrame] = {}
        for attr in self.attribute_cols:
            table = self.models[attr].predict_proba(self._attribute_features(X, attr))
            if table is None:
                raise RuntimeError(
                    f"Backend {self.backend_name!r} does not expose predict_proba "
                    f"for attribute {attr!r}; ordinal_probability matching is unavailable."
                )
            normalized = table.copy()
            normalized.index = X.index
            normalized.columns = [str(column) for column in normalized.columns]
            probabilities[attr] = normalized.astype(float)
        return probabilities

    def save(self, root_path: str) -> None:
        root = Path(root_path)
        root.mkdir(parents=True, exist_ok=True)
        model_entries = {}
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
                "format_version": 2,
                "manager_type": self.manager_type,
                "classifier_backend": self.backend_name,
                "attribute_cols": self.attribute_cols,
                "label_order": self.label_order,
                "backend_config": self.backend_config,
                "attribute_feature_columns": self.attribute_feature_columns,
                "models": model_entries,
                "problem_types": self.problem_types,
                "class_counts": self.class_counts,
                "validation_report": self.validation_report,
            },
        )

    @classmethod
    def load(cls, root_path: str):
        root = Path(root_path)
        payload = load_json(root / MANAGER_MANIFEST)
        manager = cls(
            backend_name=payload["classifier_backend"],
            attribute_cols=payload["attribute_cols"],
            backend_config={
                **payload.get("backend_config", {}),
                "attribute_feature_columns": payload.get("attribute_feature_columns", {}),
                "label_order": payload.get("label_order", payload["attribute_cols"]),
            },
        )
        manager.problem_types = dict(payload.get("problem_types", {}))
        manager.class_counts = dict(payload.get("class_counts", {}))
        manager.validation_report = dict(payload.get("validation_report", {}))
        for attr, entry in payload["models"].items():
            backend = entry["backend"]
            manager.models[attr] = BACKEND_CLASSES[backend].load(str(root / entry["path"]), label=attr)
        return manager

    @classmethod
    def load_legacy_autogluon(cls, root_path: str, attribute_cols: Sequence[str]):
        root = Path(root_path)
        manager = cls("autogluon", attribute_cols=attribute_cols)
        for attr in attribute_cols:
            attr_path = root / sanitize_model_name(attr)
            if not attr_path.exists():
                raise FileNotFoundError(f"Model for attribute [{attr}] not found: {attr_path}")
            if (attr_path / CONSTANT_MODEL_FILE).exists():
                model = ConstantAttributeBackend.load(str(attr_path))
                problem_type = "constant"
            else:
                model = AutoGluonBackend.load(str(attr_path), label=attr)
                problem_type = getattr(model, "problem_type", None) or "unknown"
            manager.models[attr] = model
            manager.problem_types[attr] = problem_type
        return manager

    def get_feature_importance(self, attr, X, y, num_shuffle_sets=3, random_state=42):
        return self.models[attr].get_feature_importance(
            self._attribute_features(X, attr), y, num_shuffle_sets, random_state
        )

    def get_metadata(self):
        return {
            "backend": self.backend_name,
            "manager_type": self.manager_type,
            "attribute_feature_columns": self.attribute_feature_columns,
            "models": {attr: self.models[attr].get_metadata() for attr in self.attribute_cols},
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


# 决定十个属性模型如何组织和交互
def create_model_manager(
    backend_name: str,
    attribute_cols: Sequence[str],
    backend_config: Optional[Mapping[str, Any]] = None,
):
    if backend_name not in CLASSIFIER_BACKENDS:
        raise ValueError(f"Unknown classifier backend {backend_name!r}. Choices: {CLASSIFIER_BACKENDS}")
    config = dict(backend_config or {})
    if backend_name == "autogluon_multilabel":
        return MultilabelPredictor(attribute_cols=attribute_cols, **config)
    if backend_name == "autogluon_soft_oof":
        return AutoGluonSoftOOFModelManager(
            attribute_cols=attribute_cols, **config
        )
    if backend_name == "catboost_classifier_chain":
        return CatBoostClassifierChainModelManager(
            attribute_cols=attribute_cols, **config
        )
    if backend_name == "catboost_soft_oof":
        return CatBoostSoftOOFModelManager(
            attribute_cols=attribute_cols, **config
        )
    if backend_name == "generic_classifier_chain":
        return GenericClassifierChainModelManager(
            attribute_cols=attribute_cols, **config
        )
    if backend_name == "logistic_regression_chain":
        return LogisticRegressionChainModelManager(
            attribute_cols=attribute_cols, **config
        )
    if backend_name == "mlp_multitask":
        return MultiTaskAttributeModelManager(attribute_cols=attribute_cols, **config)
    return IndependentAttributeModelManager(backend_name, attribute_cols, config)


def load_model_manager(
    root_path: str,
    expected_backend: Optional[str] = None,
    attribute_cols: Optional[Sequence[str]] = None,
):
    root = Path(root_path)
    manifest_path = root / MANAGER_MANIFEST
    if manifest_path.exists():
        payload = load_json(manifest_path)
        actual_backend = payload.get("classifier_backend")
        if expected_backend and actual_backend != expected_backend:
            raise ValueError(
                f"Reusable model backend mismatch: requested {expected_backend}, found {actual_backend} in {manifest_path}."
            )
        if payload.get("manager_type") == "multitask":
            return MultiTaskAttributeModelManager.load(root_path)
        if payload.get("manager_type") == "autogluon_multilabel":
            return MultilabelPredictor.load(root_path)
        if payload.get("manager_type") == "autogluon_soft_oof":
            return AutoGluonSoftOOFModelManager.load(root_path)
        if payload.get("manager_type") == "catboost_classifier_chain":
            return CatBoostClassifierChainModelManager.load(root_path)
        if payload.get("manager_type") == "catboost_soft_oof":
            return CatBoostSoftOOFModelManager.load(root_path)
        if payload.get("manager_type") == "generic_classifier_chain":
            return GenericClassifierChainModelManager.load(root_path)
        if payload.get("manager_type") == "logistic_regression_chain":
            return LogisticRegressionChainModelManager.load(root_path)
        return IndependentAttributeModelManager.load(root_path)
    # Backward compatibility for model directories created by the original AutoGluon-only pipeline.
    if expected_backend in (None, "autogluon") and attribute_cols:
        return IndependentAttributeModelManager.load_legacy_autogluon(root_path, attribute_cols)
    if (root / BACKEND_METADATA).exists():
        payload = load_json(root / BACKEND_METADATA)
        if payload.get("backend") == "mlp_multitask":
            return MultiTaskAttributeModelManager.load(root_path)
    raise FileNotFoundError(f"No model-manager manifest found in {root_path}")
