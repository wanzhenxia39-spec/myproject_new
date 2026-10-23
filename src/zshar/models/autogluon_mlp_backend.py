"""AutoGluon adapter for the native ZSHAR independent MLP.

The maintained ``mlp_independent`` backend predates the AutoGluon ensemble
integration.  This module exposes the same network and training behavior as an
AutoGluon ``AbstractModel`` so it can participate in bagging, stacking and the
weighted ensemble alongside AutoGluon's native tabular models.

AutoGluon is imported only when this optional model suite is requested.  The
rest of ZSHAR therefore remains usable without the AutoGluon dependency.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd

from autogluon.common.features.types import R_BOOL, R_FLOAT, R_INT
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY
from autogluon.core.models import AbstractModel
from autogluon.core.utils.exceptions import TimeLimitExceeded

from .independent_mlp_backend import IndependentMLPBackend


class AutoGluonIndependentMLPModel(AbstractModel):
    """AutoGluon-compatible form of ZSHAR's independent numerical MLP."""

    ag_key = "ZSHAR_MLP"
    ag_name = "MLPIndependent"
    ag_priority = 65

    def _set_default_params(self) -> None:
        defaults = {
            "hidden_dims": [256, 128],
            "dropout": 0.3,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 128,
            "epochs": 200,
            "patience": 20,
            "random_state": 42,
        }
        for name, value in defaults.items():
            self._set_default_param_value(name, value)

    def _get_default_auxiliary_params(self) -> dict[str, Any]:
        params = super()._get_default_auxiliary_params()
        params.update({"valid_raw_types": [R_BOOL, R_INT, R_FLOAT]})
        return params

    def _get_default_resources(self) -> tuple[int, int]:
        try:
            import torch

            gpu_count = 1 if torch.cuda.is_available() else 0
        except ImportError:
            gpu_count = 0
        return max(1, ResourceManager.get_cpu_count()), gpu_count

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass"]

    def _more_tags(self) -> dict[str, bool]:
        return {"can_refit_full": True}

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_gpus: int = 0,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if time_limit is not None and float(time_limit) <= 0:
            raise TimeLimitExceeded

        X = self.preprocess(X)
        X_val = self.preprocess(X_val) if X_val is not None else None
        params = self._get_model_params()
        device = "cuda:0" if int(num_gpus or 0) > 0 else "cpu"
        params["device"] = device
        params["time_limit"] = time_limit

        # AutoGluon supplies internally encoded integer labels.  Converting
        # them to strings lets the existing backend retain its deterministic
        # label ordering while keeping probability columns in class-index order.
        train_labels = pd.Series(y, index=X.index).astype(str)
        validation_labels = (
            pd.Series(y_val, index=X_val.index).astype(str)
            if X_val is not None and y_val is not None
            else None
        )
        self.model = IndependentMLPBackend(label=self.name, **params)
        self.model.fit(X, train_labels, X_val, validation_labels)
        self.params_trained["best_epoch"] = int(self.model.best_epoch)

    def _predict_proba(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        del kwargs
        X = self.preprocess(X)
        probabilities = self.model.predict_proba(X).to_numpy(dtype=np.float32)
        if self.problem_type == BINARY:
            return probabilities[:, 1]
        return probabilities


ZSHAR_FIVE_BASE_MODELS = (
    "CatBoost",
    "LightGBM",
    "ExtraTrees",
    "MLPIndependent",
    "LogisticRegression",
)


def zshar_five_hyperparameters(
    model_config: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[Any, list[dict[str, Any]]]:
    """Return exactly five level-1 models, with optional per-model overrides."""

    overrides = {str(key): dict(value) for key, value in (model_config or {}).items()}
    allowed = {
        "catboost",
        "lightgbm",
        "extra_trees",
        "logistic_regression",
        "mlp_independent",
    }
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown zshar_five model_suite_config entries: {unknown}; "
            f"choose from {sorted(allowed)}."
        )

    def configured(name: str, defaults: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(dict(defaults))
        result.update(overrides.get(name, {}))
        return result

    return {
        "CAT": [
            configured("catboost", {"ag_args": {"name_suffix": "ZSHAR"}})
        ],
        "GBM": [
            configured("lightgbm", {"ag_args": {"name_suffix": "ZSHAR"}})
        ],
        "XT": [
            configured(
                "extra_trees",
                {
                    "criterion": "gini",
                    "ag_args": {
                        "name_suffix": "ZSHAR",
                        "problem_types": ["binary", "multiclass"],
                    },
                },
            )
        ],
        "LR": [
            configured(
                "logistic_regression", {"ag_args": {"name_suffix": "ZSHAR"}}
            )
        ],
        AutoGluonIndependentMLPModel: [
            configured(
                "mlp_independent",
                {
                    "hidden_dims": [256, 128],
                    "dropout": 0.3,
                    "lr": 0.001,
                    "weight_decay": 0.0001,
                    "batch_size": 128,
                    "epochs": 200,
                    "patience": 20,
                    "random_state": 42,
                    "ag_args": {"name_suffix": "ZSHAR"},
                },
            )
        ],
    }
