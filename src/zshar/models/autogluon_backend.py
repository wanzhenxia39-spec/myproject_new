"""AutoGluon compatibility backend and experiment metadata extraction."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from .base import BaseAttributeBackend
from .serialization import BACKEND_METADATA, load_json, save_json


class AutoGluonBackend(BaseAttributeBackend):
    backend_name = "autogluon"

    def __init__(
        self,
        label=None,
        model_path="",
        problem_type=None,
        time_limit=3600,
        presets="medium_quality",
        num_gpus=1,
        num_cpus=None,
        eval_metric="f1_macro",
        num_bag_folds=0,
        num_stack_levels=0,
        fit_weighted_ensemble=True,
        model_suite="default",
        verbosity=0,
        gpu_model_types=None,
        **config,
    ):
        super().__init__(
            label=label,
            problem_type=problem_type,
            time_limit=time_limit,
            presets=presets,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            eval_metric=eval_metric,
            num_bag_folds=num_bag_folds,
            num_stack_levels=num_stack_levels,
            fit_weighted_ensemble=fit_weighted_ensemble,
            model_suite=str(model_suite),
            verbosity=int(verbosity),
            gpu_model_types=list(gpu_model_types or []),
            **config,
        )
        self.model_path = str(model_path)
        self.problem_type = problem_type
        self.predictor = None
        self.training_metadata: dict[str, Any] = {}

    @staticmethod
    def _model_suite_hyperparameters(
        model_suite: str,
        model_suite_config: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[Any, list[str]]:
        suite = str(model_suite).strip().lower()
        if suite == "default":
            return None, []
        if suite == "zshar_five":
            from .autogluon_mlp_backend import (
                ZSHAR_FIVE_BASE_MODELS,
                zshar_five_hyperparameters,
            )

            return (
                zshar_five_hyperparameters(model_suite_config),
                list(ZSHAR_FIVE_BASE_MODELS),
            )
        raise ValueError(
            f"Unknown AutoGluon model_suite {model_suite!r}; "
            "choose 'default' or 'zshar_five'."
        )

    @staticmethod
    def _predictor_class():
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError as exc:
            raise ImportError("autogluon backend requires `pip install autogluon.tabular`.") from exc
        return TabularPredictor

    @staticmethod
    def _configure_lightgbm_device(hyperparameters: Any, device: str) -> Any:
        """Set the LightGBM device without changing the other AutoGluon models.

        AutoGluon 1.4 requests LightGBM's OpenCL backend (``device=gpu``)
        whenever the global ``num_gpus`` value is positive.  A LightGBM build
        compiled with ``USE_CUDA=ON`` instead expects ``device=cuda``.  Adding
        the device to each GBM configuration prevents AutoGluon from
        overwriting it while leaving CAT, XGB, NN_TORCH, etc. on their normal
        resource configuration.
        """
        normalized = str(device).strip().lower()
        choices = {"cpu", "gpu", "cuda"}
        if normalized not in choices:
            raise ValueError(
                f"Invalid AutoGluon LightGBM device {device!r}; choose one of {sorted(choices)}."
            )

        if hyperparameters is None:
            from autogluon.tabular.configs.hyperparameter_configs import (
                get_hyperparameter_config,
            )

            configured = get_hyperparameter_config("default")
        elif isinstance(hyperparameters, str):
            from autogluon.tabular.configs.hyperparameter_configs import (
                get_hyperparameter_config,
            )

            configured = get_hyperparameter_config(hyperparameters)
        else:
            configured = deepcopy(hyperparameters)

        if not isinstance(configured, Mapping) or "GBM" not in configured:
            return configured

        configured = dict(configured)
        raw_gbm = configured["GBM"]
        gbm_configs = raw_gbm if isinstance(raw_gbm, list) else [raw_gbm]
        updated_gbm = []
        for raw_config in gbm_configs:
            if not isinstance(raw_config, Mapping):
                raise ValueError(
                    "AutoGluon GBM hyperparameters must be dictionaries when "
                    "`lightgbm_device` is specified."
                )
            model_config = deepcopy(dict(raw_config))
            model_config["device"] = normalized
            ag_args_fit = dict(model_config.get("ag_args_fit", {}))
            ag_args_fit["num_gpus"] = 0 if normalized == "cpu" else 1
            model_config["ag_args_fit"] = ag_args_fit
            updated_gbm.append(model_config)
        configured["GBM"] = updated_gbm if isinstance(raw_gbm, list) else updated_gbm[0]
        return configured

    @staticmethod
    def _configure_gpu_model_resources(
        hyperparameters: Any,
        model_types: list[str],
    ) -> Any:
        """Explicitly opt selected AutoGluon model families into one GPU.

        AutoGluon 1.4 defaults every tabular model family to zero GPUs, even
        when ``TabularPredictor.fit(num_gpus=1)`` makes a GPU available.  The
        per-model ``ag_args_fit`` resource request is therefore required.
        """
        supported = {"NN_TORCH", "FASTAI", "CAT", "XGB"}
        requested = [str(name).strip().upper() for name in model_types]
        invalid = sorted(set(requested).difference(supported))
        if invalid:
            raise ValueError(
                f"Unsupported gpu_model_types={invalid}; choose from {sorted(supported)}. "
                "LightGBM GPU selection is controlled by lightgbm_device."
            )
        if not requested:
            return hyperparameters
        if hyperparameters is None:
            from autogluon.tabular.configs.hyperparameter_configs import (
                get_hyperparameter_config,
            )

            configured = get_hyperparameter_config("default")
        elif isinstance(hyperparameters, str):
            from autogluon.tabular.configs.hyperparameter_configs import (
                get_hyperparameter_config,
            )

            configured = get_hyperparameter_config(hyperparameters)
        else:
            configured = deepcopy(hyperparameters)
        if not isinstance(configured, Mapping):
            raise TypeError("AutoGluon hyperparameters must be a dictionary.")
        configured = dict(configured)
        missing = sorted(set(requested).difference(configured))
        if missing:
            raise ValueError(
                f"GPU model types are absent from the selected model suite: {missing}"
            )
        for model_type in requested:
            raw_configs = configured[model_type]
            configs = raw_configs if isinstance(raw_configs, list) else [raw_configs]
            updated = []
            for raw_config in configs:
                if not isinstance(raw_config, Mapping):
                    raise ValueError(
                        f"AutoGluon {model_type} hyperparameters must be dictionaries."
                    )
                model_config = deepcopy(dict(raw_config))
                ag_args_fit = dict(model_config.get("ag_args_fit", {}))
                ag_args_fit["num_gpus"] = 1
                model_config["ag_args_fit"] = ag_args_fit
                updated.append(model_config)
            configured[model_type] = updated if isinstance(raw_configs, list) else updated[0]
        return configured

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names = list(X_train.columns)
        Predictor = self._predictor_class()
        path = self.model_path
        if not path:
            raise ValueError("AutoGluonBackend requires a model_path before fit().")
        verbosity = int(self.config.get("verbosity", 0))
        if verbosity < 0 or verbosity > 4:
            raise ValueError("AutoGluon verbosity must be between 0 and 4.")
        self.predictor = Predictor(
            label=self.label,
            path=path,
            verbosity=verbosity,
            problem_type=self.problem_type,
            eval_metric=self.config.get("eval_metric", "f1_macro"),
        )
        train_data = X_train.copy()
        train_data[self.label] = y_train.astype(str).to_numpy()
        num_bag_folds = int(self.config.get("num_bag_folds", 0))
        num_stack_levels = int(self.config.get("num_stack_levels", 0))
        if num_bag_folds not in {0} and num_bag_folds < 2:
            raise ValueError("AutoGluon num_bag_folds must be 0 or at least 2.")
        if num_stack_levels < 0:
            raise ValueError("AutoGluon num_stack_levels must be non-negative.")
        fit_kwargs = {
            "train_data": train_data,
            "time_limit": int(self.config.get("time_limit", 3600)),
            "presets": self.config.get("presets", "medium_quality"),
            # Deliberately omit `hyperparameters`: AutoGluon therefore uses
            # the candidate model library selected by its preset/defaults.
            "num_bag_folds": num_bag_folds,
            "num_stack_levels": num_stack_levels,
            "fit_weighted_ensemble": bool(
                self.config.get("fit_weighted_ensemble", True)
            ),
            "num_gpus": int(self.config.get("num_gpus", 1)),
        }
        num_cpus = self.config.get("num_cpus")
        if num_cpus is not None and int(num_cpus) > 0:
            fit_kwargs["num_cpus"] = int(num_cpus)
        extra_fit = self.config.get("fit_kwargs", {})
        if not isinstance(extra_fit, dict):
            raise TypeError("AutoGluon fit_kwargs must be a dictionary.")
        extra_fit = deepcopy(extra_fit)
        model_suite = str(self.config.get("model_suite", "default")).strip().lower()
        raw_suite_config = self.config.get("model_suite_config", {})
        if not isinstance(raw_suite_config, Mapping):
            raise TypeError("AutoGluon model_suite_config must be a dictionary.")
        suite_hyperparameters, requested_models = self._model_suite_hyperparameters(
            model_suite,
            raw_suite_config,
        )
        if suite_hyperparameters is not None:
            if "hyperparameters" in extra_fit:
                raise ValueError(
                    "model_suite='zshar_five' already defines the exact base-model set; "
                    "remove fit_kwargs.hyperparameters or use model_suite='default'."
                )
            extra_fit["hyperparameters"] = suite_hyperparameters

        raw_gpu_model_types = self.config.get("gpu_model_types", [])
        if not isinstance(raw_gpu_model_types, (list, tuple)):
            raise TypeError("AutoGluon gpu_model_types must be a JSON list.")
        gpu_model_types = [str(name).strip().upper() for name in raw_gpu_model_types]
        if gpu_model_types and int(self.config.get("num_gpus", 1)) <= 0:
            raise ValueError("gpu_model_types requires num_gpus > 0.")
        if gpu_model_types:
            extra_fit["hyperparameters"] = self._configure_gpu_model_resources(
                extra_fit.get("hyperparameters"), gpu_model_types
            )

        lightgbm_device = str(
            self.config.get("lightgbm_device", "auto")
        ).strip().lower()
        if lightgbm_device not in {"auto", "cpu", "gpu", "cuda"}:
            raise ValueError(
                "AutoGluon lightgbm_device must be one of: auto, cpu, gpu, cuda."
            )
        if lightgbm_device in {"gpu", "cuda"} and int(
            self.config.get("num_gpus", 1)
        ) <= 0:
            raise ValueError(
                f"lightgbm_device={lightgbm_device!r} requires num_gpus > 0."
            )
        if lightgbm_device != "auto":
            extra_fit["hyperparameters"] = self._configure_lightgbm_device(
                extra_fit.get("hyperparameters"),
                lightgbm_device,
            )
        fit_kwargs.update(extra_fit)
        self.training_metadata = {
            "model_suite": model_suite,
            "requested_base_models": requested_models,
            "gpu_model_types": gpu_model_types,
        }
        self.predictor.fit(**fit_kwargs)
        collected = self._collect_training_metadata()
        collected.update(self.training_metadata)
        self.training_metadata = collected
        return self

    def _collect_training_metadata(self) -> dict[str, Any]:
        model_best = getattr(self.predictor, "model_best", None)
        try:
            names = list(self.predictor.model_names())
        except Exception:
            names = []
        ensemble_names = [name for name in names if "ensemble" in str(name).lower()]
        best_single = None
        try:
            leaderboard = self.predictor.leaderboard(silent=True)
            model_column = "model" if "model" in leaderboard.columns else leaderboard.columns[0]
            for name in leaderboard[model_column].astype(str):
                if "ensemble" not in name.lower():
                    best_single = name
                    break
        except Exception:
            pass
        weights: dict[str, float] = {}
        ensemble_for_weights = model_best if model_best in ensemble_names else (ensemble_names[0] if ensemble_names else None)
        if ensemble_for_weights:
            try:
                info = self.predictor._trainer.get_model_info(ensemble_for_weights)
                raw_weights = info.get("weights") or info.get("model_weights") or {}
                weights = {str(name): float(weight) for name, weight in raw_weights.items()}
            except Exception:
                pass
            if not weights:
                try:
                    ensemble_model = self.predictor._trainer.load_model(ensemble_for_weights)
                    raw_weights = None
                    for method_name in ("get_model_weights", "_get_model_weights"):
                        method = getattr(ensemble_model, method_name, None)
                        if callable(method):
                            raw_weights = method()
                            break
                    if raw_weights is None:
                        raw_weights = getattr(ensemble_model, "model_weights", None)
                    if raw_weights is None:
                        raw_weights = getattr(ensemble_model, "weights_", None)
                    if isinstance(raw_weights, dict):
                        weights = {str(name): float(weight) for name, weight in raw_weights.items()}
                    elif raw_weights is not None:
                        base_names = list(getattr(ensemble_model, "base_model_names", []))
                        weights = {
                            str(name): float(weight)
                            for name, weight in zip(base_names, list(raw_weights))
                        }
                except Exception:
                    pass
        return {
            "predictor_model_best": model_best,
            "leaderboard_best_single_model": best_single,
            "weighted_ensemble_exists": bool(ensemble_names),
            "weighted_ensemble_models": ensemble_names,
            "ensemble_weights": weights,
            "trained_model_count": len(names),
            "trained_models": names,
            "num_bag_folds": int(self.config.get("num_bag_folds", 0)),
            "num_stack_levels": int(self.config.get("num_stack_levels", 0)),
            "fit_weighted_ensemble": bool(
                self.config.get("fit_weighted_ensemble", True)
            ),
            "model_suite": str(self.config.get("model_suite", "default")),
            "uses_default_hyperparameters": (
                str(self.config.get("model_suite", "default")) == "default"
                and "hyperparameters" not in self.config.get("fit_kwargs", {})
            ),
        }

    def _input_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preserve the fitted attribute subset and column order after reload.

        The independent manager selects each attribute's bound columns before
        fit. Old AutoGluon exports without our feature metadata retain their
        predictor's native schema checks.
        """
        if not self.feature_names:
            return X
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"Missing trained features for {self.label!r}: {missing[:10]}")
        return X.loc[:, self.feature_names]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        prediction = self.predictor.predict(self._input_features(X))
        return pd.Series(prediction, index=X.index, name=self.label).astype(str)

    def predict_proba(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        features = self._input_features(X)
        try:
            probabilities = self.predictor.predict_proba(features)
        except Exception:
            return None
        if isinstance(probabilities, pd.Series):
            probabilities = probabilities.to_frame()
        probabilities.index = X.index
        return probabilities

    def predict_proba_oof(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Return AutoGluon's native bagging OOF probabilities.

        This deliberately delegates to ``TabularPredictor.predict_proba_oof``
        instead of rebuilding folds in ZSHAR.  ``train_data`` is supplied so
        AutoGluon can restore any training rows that its label preprocessing
        may have dropped.  The method is additive: independent AutoGluon and
        the existing hard-label multilabel manager do not call it.
        """
        if self.predictor is None:
            raise RuntimeError("AutoGluon predictor is not fitted.")
        train_data = self._input_features(X_train).copy()
        if y_train is not None:
            train_data[self.label] = pd.Series(y_train, index=X_train.index).astype(
                str
            )
        try:
            probabilities = self.predictor.predict_proba_oof(
                train_data=train_data,
                as_multiclass=True,
            )
        except (AssertionError, ValueError) as exc:
            raise RuntimeError(
                "AutoGluon bagging OOF probabilities are unavailable. "
                "Use num_bag_folds >= 2 and do not replace the selected model "
                "with a non-bagged/refit-only model."
            ) from exc
        if isinstance(probabilities, pd.Series):
            probabilities = probabilities.to_frame()
        probabilities = probabilities.copy()
        probabilities.columns = [str(column) for column in probabilities.columns]
        probabilities = probabilities.reindex(X_train.index)
        if len(probabilities) != len(X_train) or probabilities.isna().any().any():
            raise RuntimeError(
                "AutoGluon OOF rows do not align with the original training frame."
            )
        return probabilities.astype(float)

    def save(self, path: str) -> None:
        # AutoGluon persists during fit; only our portable metadata is added here.
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        save_json(
            directory / BACKEND_METADATA,
            {
                "backend": self.backend_name,
                "label": self.label,
                "problem_type": self.problem_type,
                "config": self.config,
                "feature_names": self.feature_names,
                "training_metadata": self.training_metadata,
            },
        )

    @classmethod
    def load(cls, path: str, **kwargs: Any):
        metadata_path = Path(path) / BACKEND_METADATA
        payload = load_json(metadata_path) if metadata_path.exists() else {}
        config = dict(payload.get("config", {}))
        config.pop("problem_type", None)
        model = cls(
            label=payload.get("label", kwargs.get("label")),
            model_path=path,
            problem_type=payload.get("problem_type"),
            **config,
        )
        model.predictor = cls._predictor_class().load(path)
        model.feature_names = list(payload.get("feature_names", []))
        model.training_metadata = dict(payload.get("training_metadata", {}))
        if not model.training_metadata:
            model.training_metadata = model._collect_training_metadata()
        if model.label is None:
            model.label = getattr(model.predictor, "label", None)
        return model

    def get_metadata(self):
        result = super().get_metadata()
        result.update(self.training_metadata)
        return result

    def get_feature_importance(self, X, y, num_shuffle_sets=3, random_state=42):
        data = self._input_features(X).copy()
        data[self.label] = pd.Series(y, index=X.index).astype(str)
        kwargs = {
            "data": data,
            "feature_stage": "original",
            "num_shuffle_sets": max(1, int(num_shuffle_sets)),
        }
        try:
            kwargs["silent"] = True
            return self.predictor.feature_importance(**kwargs)
        except TypeError:
            kwargs.pop("silent", None)
            return self.predictor.feature_importance(**kwargs)
