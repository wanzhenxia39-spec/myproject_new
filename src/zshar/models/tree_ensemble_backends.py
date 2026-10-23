"""Native scikit-learn tree-ensemble attribute-classifier backends.

These backends deliberately instantiate scikit-learn estimators directly.
They do not route training or inference through AutoGluon.
"""

from __future__ import annotations

from typing import Any

from .boosting_backends import EncodedBoostingBackend


class ExtraTreesBackend(EncodedBoostingBackend):
    """One native ``ExtraTreesClassifier`` per non-constant attribute."""

    backend_name = "extra_trees"

    def _create_estimator(self, num_classes: int) -> Any:
        del num_classes  # sklearn infers binary/multiclass behavior from y.
        from sklearn.ensemble import ExtraTreesClassifier

        params = {
            "n_estimators": 500,
            "criterion": "gini",
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "bootstrap": False,
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        params.update(
            {
                key: value
                for key, value in self.config.items()
                if key not in {"random_state", "n_jobs"}
            }
        )
        return ExtraTreesClassifier(**params)


class RandomForestBackend(EncodedBoostingBackend):
    """One native ``RandomForestClassifier`` per non-constant attribute."""

    backend_name = "random_forest"

    def _create_estimator(self, num_classes: int) -> Any:
        del num_classes  # sklearn infers binary/multiclass behavior from y.
        from sklearn.ensemble import RandomForestClassifier

        params = {
            "n_estimators": 500,
            "criterion": "gini",
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "bootstrap": True,
            "class_weight": "balanced_subsample",
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        params.update(
            {
                key: value
                for key, value in self.config.items()
                if key not in {"random_state", "n_jobs"}
            }
        )
        return RandomForestClassifier(**params)
