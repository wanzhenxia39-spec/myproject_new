"""Constant predictor for attributes with one seen-training label."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .base import BaseAttributeBackend
from .serialization import load_json, save_json


CONSTANT_MODEL_FILE = "constant_model.json"


class ConstantAttributeBackend(BaseAttributeBackend):
    backend_name = "constant"

    def __init__(self, label: Optional[str] = None, constant_value: str = "", **config: Any) -> None:
        super().__init__(label=label, **config)
        self.constant_value = str(constant_value)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        values = pd.Series(y_train).dropna().astype(str).unique()
        if len(values) != 1:
            raise ValueError(f"Constant backend requires exactly one class, got {values.tolist()}")
        self.constant_value = str(values[0])
        self.feature_names = list(X_train.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.constant_value, index=X.index, name=self.label, dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({self.constant_value: np.ones(len(X))}, index=X.index)

    def save(self, path: str) -> None:
        save_json(
            Path(path) / CONSTANT_MODEL_FILE,
            {
                "model_type": "constant",
                "backend": self.backend_name,
                "label": self.label,
                "constant_value": self.constant_value,
                "feature_names": self.feature_names,
            },
        )

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "ConstantAttributeBackend":
        payload = load_json(Path(path) / CONSTANT_MODEL_FILE)
        model = cls(label=payload.get("label"), constant_value=payload["constant_value"])
        model.feature_names = list(payload.get("feature_names", []))
        return model

    def get_feature_importance(self, X, y, num_shuffle_sets=3, random_state=42):
        return pd.DataFrame(columns=["importance", "stddev"])
