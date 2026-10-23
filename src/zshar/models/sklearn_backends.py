"""Scikit-learn attribute classifier backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .base import BaseAttributeBackend
from .serialization import BACKEND_METADATA, load_json, save_json


MODEL_FILE = "model.joblib"


class SklearnAttributeBackend(BaseAttributeBackend):
    estimator: Any

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names = list(X_train.columns)
        self.estimator.fit(X_train, y_train.astype(str))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.estimator.predict(X), index=X.index, name=self.label).astype(str)

    def predict_proba(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        if not hasattr(self.estimator, "predict_proba"):
            return None
        values = self.estimator.predict_proba(X)
        classes = getattr(self.estimator, "classes_", None)
        if classes is None and hasattr(self.estimator, "named_steps"):
            classes = getattr(self.estimator.named_steps.get("classifier"), "classes_", None)
        return pd.DataFrame(values, index=X.index, columns=list(classes) if classes is not None else None)

    def decision_function(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if hasattr(self.estimator, "decision_function"):
            return np.asarray(self.estimator.decision_function(X))
        return None

    def save(self, path: str) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, directory / MODEL_FILE)
        save_json(
            directory / BACKEND_METADATA,
            {
                "backend": self.backend_name,
                "label": self.label,
                "config": self.config,
                "feature_names": self.feature_names,
            },
        )

    @classmethod
    def load(cls, path: str, **kwargs: Any):
        payload = load_json(Path(path) / BACKEND_METADATA)
        model = cls(label=payload.get("label"), **payload.get("config", {}))
        model.estimator = joblib.load(Path(path) / MODEL_FILE)
        model.feature_names = list(payload.get("feature_names", []))
        return model


class LogisticRegressionBackend(SklearnAttributeBackend):
    backend_name = "logistic_regression"

    def __init__(self, label=None, random_state=42, max_iter=3000, class_weight="balanced", **config):
        super().__init__(
            label=label,
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight,
            **config,
        )
        classifier_kwargs = dict(config)
        self.estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=max_iter,
                        class_weight=class_weight,
                        random_state=random_state,
                        **classifier_kwargs,
                    ),
                ),
            ]
        )


class LinearSVMBackend(SklearnAttributeBackend):
    backend_name = "linear_svm"

    def __init__(
        self,
        label=None,
        random_state=42,
        C=1.0,
        class_weight="balanced",
        calibrate_proba=False,
        **config,
    ):
        super().__init__(
            label=label,
            random_state=random_state,
            C=C,
            class_weight=class_weight,
            calibrate_proba=calibrate_proba,
            **config,
        )
        self.calibrate_proba = bool(calibrate_proba)
        svc_kwargs = dict(config)
        base = LinearSVC(
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            **svc_kwargs,
        )
        classifier = CalibratedClassifierCV(estimator=base, cv=3) if self.calibrate_proba else base
        self.estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.calibrate_proba and int(pd.Series(y_train).value_counts().min()) < 3:
            raise ValueError("--svm_calibrate_proba requires at least 3 training rows in every class.")
        return super().fit(X_train, y_train, X_val, y_val)
