"""Hard-label classifier chain with a selectable per-attribute base learner."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .catboost_classifier_chain_backend import CatBoostClassifierChainModelManager


class GenericClassifierChainModelManager(CatBoostClassifierChainModelManager):
    """Use any independent probability-capable backend at each chain node."""

    manager_type = "generic_classifier_chain"
    backend_name = "generic_classifier_chain"
    chain_feature_prefix = "__zshar_generic_chain__"
    unsupported_base_backends = {
        "autogluon_multilabel",
        "autogluon_soft_oof",
        "catboost_classifier_chain",
        "catboost_soft_oof",
        "generic_classifier_chain",
        "logistic_regression_chain",
        "mlp_multitask",
    }

    def __init__(
        self,
        attribute_cols: Optional[Sequence[str]] = None,
        base_backend: str | None = None,
        **backend_config: Any,
    ) -> None:
        from .factory import BACKEND_CLASSES

        if base_backend is None:
            raise ValueError("generic_classifier_chain requires base_backend")
        if base_backend not in BACKEND_CLASSES or base_backend in self.unsupported_base_backends:
            raise ValueError(f"Unsupported generic-chain base backend: {base_backend!r}")
        self.base_backend = str(base_backend)
        super().__init__(
            attribute_cols=attribute_cols,
            base_backend=self.base_backend,
            **backend_config,
        )

    @property
    def model_config(self) -> dict[str, Any]:
        config = super().model_config
        config.pop("base_backend", None)
        return config

    def _create_attribute_model(self, attr: str, model_path: str) -> Any:
        from .factory import create_attribute_backend

        return create_attribute_backend(
            self.base_backend,
            attr,
            self.model_config,
            model_path=model_path,
        )

    @classmethod
    def _load_attribute_model(cls, backend: str, path: str, attr: str) -> Any:
        from .factory import BACKEND_CLASSES

        if backend not in BACKEND_CLASSES or backend == "constant":
            raise ValueError(f"Unsupported saved generic-chain node backend: {backend!r}")
        return BACKEND_CLASSES[backend].load(path, label=attr)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata["base_backend"] = self.base_backend
        return metadata
