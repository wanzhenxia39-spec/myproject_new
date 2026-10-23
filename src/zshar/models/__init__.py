"""Unified attribute-classifier backends shared by every dataset."""

from .base import BaseAttributeBackend
from .autogluon_multilabel_backend import (
    DEFAULT_BASE_ATTRIBUTE_ORDER,
    AutoGluonMultilabelModelManager,
    MultilabelPredictor,
)
from .autogluon_soft_oof_backend import AutoGluonSoftOOFModelManager
from .catboost_classifier_chain_backend import CatBoostClassifierChainModelManager
from .generic_classifier_chain_backend import GenericClassifierChainModelManager
from .catboost_soft_oof_backend import CatBoostSoftOOFModelManager
from .logistic_regression_chain_backend import LogisticRegressionChainModelManager
from .factory import (
    CLASSIFIER_BACKENDS,
    IndependentAttributeModelManager,
    create_model_manager,
    load_model_manager,
)
from .multitask_mlp_backend import MultiTaskAttributeModelManager
from .native_neural_backends import NeuralNetFastAIBackend, NeuralNetTorchBackend
from .ordinal_backends import CORALMLPBackend, OrdinalLogisticBackend
from .tree_ensemble_backends import ExtraTreesBackend, RandomForestBackend

__all__ = [
    "BaseAttributeBackend",
    "AutoGluonMultilabelModelManager",
    "AutoGluonSoftOOFModelManager",
    "CatBoostClassifierChainModelManager",
    "GenericClassifierChainModelManager",
    "CatBoostSoftOOFModelManager",
    "LogisticRegressionChainModelManager",
    "MultilabelPredictor",
    "DEFAULT_BASE_ATTRIBUTE_ORDER",
    "CLASSIFIER_BACKENDS",
    "IndependentAttributeModelManager",
    "MultiTaskAttributeModelManager",
    "ExtraTreesBackend",
    "RandomForestBackend",
    "NeuralNetFastAIBackend",
    "NeuralNetTorchBackend",
    "OrdinalLogisticBackend",
    "CORALMLPBackend",
    "create_model_manager",
    "load_model_manager",
]
