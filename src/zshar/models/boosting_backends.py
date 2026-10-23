"""LightGBM, XGBoost and CatBoost attribute backends (all lazily imported)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from .base import BaseAttributeBackend
from .serialization import BACKEND_METADATA, load_json, save_json


MODEL_FILE = "model.joblib"


class EncodedBoostingBackend(BaseAttributeBackend):
    """Shared explicit string-label encoding and median-imputation behavior."""

    estimator: Any

    def __init__(self, label=None, random_state=42, n_jobs=-1, **config):
        super().__init__(label=label, random_state=random_state, n_jobs=n_jobs, **config)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs) if n_jobs is not None else -1
        self.imputer = SimpleImputer(strategy="median")
        self.label_encoder = LabelEncoder()

    def _create_estimator(self, num_classes: int) -> Any:
        raise NotImplementedError

    def _transform_features(
        self,
        X,
        *,
        fit: bool = False,
        use_estimator_feature_names: bool = False,
    ) -> pd.DataFrame:
        """Impute features while preserving the names expected by sklearn models."""
        if isinstance(X, pd.DataFrame):
            frame = X.loc[:, self.feature_names]
            index = X.index
        else:
            frame = pd.DataFrame(X, columns=self.feature_names)
            index = frame.index
        values = self.imputer.fit_transform(frame) if fit else self.imputer.transform(frame)
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                "Median imputation changed the number of features. "
                "Check for training columns containing only missing values."
            )
        output_names = self.feature_names
        if use_estimator_feature_names:
            # Old saved models may have been fitted from an ndarray and therefore
            # carry synthetic names such as Column_0. Preserve load compatibility.
            try:
                fitted_names = list(self.estimator.feature_names_in_)
            except (AttributeError, TypeError):
                fitted_names = []
            if len(fitted_names) == values.shape[1]:
                output_names = fitted_names
        return pd.DataFrame(values, index=index, columns=output_names)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names = list(X_train.columns)
        train_values = self._transform_features(X_train, fit=True)
        encoded = self.label_encoder.fit_transform(y_train.astype(str))
        self.estimator = self._create_estimator(len(self.label_encoder.classes_))
        fit_kwargs = self._training_fit_kwargs(encoded)
        if X_val is not None and y_val is not None:
            known = pd.Series(y_val).astype(str).isin(self.label_encoder.classes_).to_numpy()
            if known.any():
                val_values = self._transform_features(X_val.loc[known])
                val_encoded = self.label_encoder.transform(pd.Series(y_val).astype(str).loc[known])
                fit_kwargs.update(self._validation_fit_kwargs(val_values, val_encoded))
        self.estimator.fit(train_values, encoded, **fit_kwargs)
        return self

    def _training_fit_kwargs(self, y_train: np.ndarray) -> dict[str, Any]:
        return {}

    def _validation_fit_kwargs(self, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
        return {}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        features = self._transform_features(X, use_estimator_feature_names=True)
        encoded = np.asarray(self.estimator.predict(features)).reshape(-1).astype(int)
        labels = self.label_encoder.inverse_transform(encoded)
        return pd.Series(labels, index=X.index, name=self.label)

    def predict_proba(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        if not hasattr(self.estimator, "predict_proba"):
            return None
        features = self._transform_features(X, use_estimator_feature_names=True)
        probabilities = np.asarray(self.estimator.predict_proba(features))
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1.0 - probabilities, probabilities])
        return pd.DataFrame(probabilities, index=X.index, columns=self.label_encoder.classes_)

    def save(self, path: str) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "imputer": self.imputer,
                "label_encoder": self.label_encoder,
            },
            directory / MODEL_FILE,
        )
        save_json(
            directory / BACKEND_METADATA,
            {
                "backend": self.backend_name,
                "label": self.label,
                "config": self.config,
                "feature_names": self.feature_names,
                "classes": self.label_encoder.classes_.tolist(),
            },
        )

    @classmethod
    def load(cls, path: str, **kwargs: Any):
        payload = load_json(Path(path) / BACKEND_METADATA)
        model = cls(label=payload.get("label"), **payload.get("config", {}))
        state = joblib.load(Path(path) / MODEL_FILE)
        model.estimator = state["estimator"]
        model.imputer = state["imputer"]
        model.label_encoder = state["label_encoder"]
        model.feature_names = list(payload.get("feature_names", []))
        return model

    def get_feature_importance(self, X, y, num_shuffle_sets=3, random_state=42):
        importance = getattr(self.estimator, "feature_importances_", None)
        if importance is None:
            return super().get_feature_importance(X, y, num_shuffle_sets, random_state)
        return pd.DataFrame(
            {"importance": np.asarray(importance, dtype=float), "stddev": np.nan},
            index=self.feature_names,
        ).sort_values("importance", ascending=False)


class LightGBMBackend(EncodedBoostingBackend):
    backend_name = "lightgbm"

    def _create_estimator(self, num_classes: int):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm backend requires `pip install lightgbm`.") from exc
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": -1,
        }
        params.update({key: value for key, value in self.config.items() if key not in {"random_state", "n_jobs"}})
        return LGBMClassifier(**params)


class XGBoostBackend(EncodedBoostingBackend):
    backend_name = "xgboost"

    def _training_fit_kwargs(self, y_train: np.ndarray) -> dict[str, Any]:
        """Balance ordinal attribute classes without changing test-time behavior."""
        if not bool(self.config.get("balance_classes", True)):
            return {}
        return {"sample_weight": compute_sample_weight(class_weight="balanced", y=y_train)}

    def _create_estimator(self, num_classes: int):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost backend requires `pip install xgboost`.") from exc
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "tree_method": "hist",
            "device": "cpu",
            "objective": "binary:logistic" if num_classes == 2 else "multi:softprob",
            "eval_metric": "logloss" if num_classes == 2 else "mlogloss",
        }
        if num_classes > 2:
            params["num_class"] = num_classes
        params.update(
            {
                key: value
                for key, value in self.config.items()
                if key not in {"random_state", "n_jobs", "balance_classes"}
            }
        )
        # Device/tree_method remain configurable (for example device="cuda",
        # tree_method="hist"). Derive only the task-dependent parameters here.
        params["objective"] = "binary:logistic" if num_classes == 2 else "multi:softprob"
        params["eval_metric"] = "logloss" if num_classes == 2 else "mlogloss"
        if num_classes > 2:
            params["num_class"] = num_classes
        else:
            params.pop("num_class", None)
        return XGBClassifier(**params)


class CatBoostBackend(EncodedBoostingBackend):
    backend_name = "catboost"

    def _create_estimator(self, num_classes: int):
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("catboost backend requires `pip install catboost`.") from exc
        params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "verbose": False,
            "random_seed": self.random_state,
            "thread_count": self.n_jobs,
            "loss_function": "Logloss" if num_classes == 2 else "MultiClass",
            "auto_class_weights": "Balanced",
            # Avoid creating a catboost_info directory for every attribute/split.
            "allow_writing_files": False,
        }
        params.update(
            {
                key: value
                for key, value in self.config.items()
                if key not in {"random_state", "random_seed", "n_jobs", "thread_count"}
            }
        )
        params.pop("cat_features", None)
        params["verbose"] = False
        params["random_seed"] = self.random_state
        params["loss_function"] = "Logloss" if num_classes == 2 else "MultiClass"
        return CatBoostClassifier(**params)
