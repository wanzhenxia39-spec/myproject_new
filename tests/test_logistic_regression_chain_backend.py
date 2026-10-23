from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zshar.models.factory import create_model_manager, load_model_manager


class _FakeLogisticRegressionBackend:
    backend_name = "logistic_regression"
    fitted_columns: dict[str, list[str]] = {}
    fitted_dtypes: dict[str, list[str]] = {}

    def __init__(self, label=None, **config):
        self.label = label
        self.config = config
        self.feature_names: list[str] = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names = list(X_train.columns)
        type(self).fitted_columns[self.label] = self.feature_names
        type(self).fitted_dtypes[self.label] = [str(dtype) for dtype in X_train.dtypes]
        return self

    def predict(self, X):
        if self.label == "a":
            values = np.where(X["signal"].to_numpy() >= 0, "High", "Low")
        else:
            previous = "a" if self.label == "b" else "b"
            chain_column = f"__zshar_logistic_chain__{previous}"
            if chain_column in X:
                values = np.where(X[chain_column].to_numpy() >= 0.75, "High", "Low")
            else:
                values = np.where(X["signal"].to_numpy() >= 0, "High", "Low")
        return pd.Series(values, index=X.index, name=self.label, dtype=object)

    def predict_proba(self, X):
        prediction = self.predict(X)
        return pd.DataFrame(
            {
                "Low": (prediction == "Low").astype(float),
                "High": (prediction == "High").astype(float),
            },
            index=X.index,
        )

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path, **kwargs):
        return cls(label=kwargs.get("label"))

    def get_metadata(self):
        return {
            "backend": self.backend_name,
            "label": self.label,
            "trained_model_count": 1,
        }

    def get_feature_importance(self, X, y, num_shuffle_sets=3, random_state=42):
        return pd.DataFrame({"importance": np.ones(X.shape[1])}, index=X.columns)


def test_explicit_order_must_be_an_exact_permutation():
    from zshar.models.logistic_regression_chain_backend import (
        LogisticRegressionChainModelManager,
    )

    with pytest.raises(ValueError, match="every attribute exactly once"):
        LogisticRegressionChainModelManager(
            ["a", "b"], label_order=["a", "missing"]
        )


def test_logistic_chain_teacher_forcing_and_autoregressive_inference(
    tmp_path, monkeypatch
):
    import zshar.models.logistic_regression_chain_backend as module

    monkeypatch.setattr(
        module, "LogisticRegressionBackend", _FakeLogisticRegressionBackend
    )
    _FakeLogisticRegressionBackend.fitted_columns = {}
    _FakeLogisticRegressionBackend.fitted_dtypes = {}
    attributes = ["c", "a", "b"]
    manager = create_model_manager(
        "logistic_regression_chain",
        attributes,
        {
            "label_order": ["a", "b", "c"],
            "max_iter": 17,
            "consider_labels_correlation": True,
        },
    )
    X_train = pd.DataFrame({"signal": [-2.0, -1.0, 1.0, 2.0]})
    Y_train = pd.DataFrame(
        {
            "c": ["Low", "Low", "High", "High"],
            "a": ["Low", "Low", "High", "High"],
            "b": ["Low", "Low", "High", "High"],
        }
    )
    X_val = pd.DataFrame({"signal": [-3.0, 3.0]}, index=[10, 20])
    Y_val = pd.DataFrame(
        {
            "c": ["High", "Low"],
            "a": ["High", "Low"],
            "b": ["High", "Low"],
        },
        index=X_val.index,
    )

    report = manager.fit_all(
        X_train, Y_train, X_val, Y_val, root_path=str(tmp_path / "models")
    )

    assert manager.label_order == ["a", "b", "c"]
    assert manager.model_config == {"max_iter": 17}
    assert _FakeLogisticRegressionBackend.fitted_columns == {
        "a": ["signal"],
        "b": ["signal", "__zshar_logistic_chain__a"],
        "c": [
            "signal",
            "__zshar_logistic_chain__a",
            "__zshar_logistic_chain__b",
        ],
    }
    assert all(
        dtype != "object"
        for dtypes in _FakeLogisticRegressionBackend.fitted_dtypes.values()
        for dtype in dtypes
    )
    prediction = manager.predict_all(X_val)
    expected = pd.Series(["Low", "High"], index=X_val.index)
    for attr in attributes:
        pd.testing.assert_series_equal(
            prediction[attr], expected, check_names=False, check_dtype=False
        )
    assert report["training_chain_inputs"] == "previous_ground_truth_labels_teacher_forcing"
    assert report["validation_inference"] == "autoregressive_predicted_labels"
    assert report["per_attribute"]["b"]["accuracy"] == 0.0

    probabilities = manager.predict_proba_all(X_val)
    assert list(probabilities) == attributes
    assert probabilities["c"].loc[10, "Low"] == 1.0

    loaded = load_model_manager(
        str(tmp_path / "models"),
        expected_backend="logistic_regression_chain",
        attribute_cols=attributes,
    )
    assert loaded.label_order == ["a", "b", "c"]
    assert loaded.chain_feature_encoding == "ordinal"
    pd.testing.assert_frame_equal(loaded.predict_all(X_val), prediction)


def test_ordinal_chain_features_use_low_medium_high_values():
    from zshar.models.logistic_regression_chain_backend import (
        LogisticRegressionChainModelManager,
    )

    manager = LogisticRegressionChainModelManager(["a"])
    features = pd.DataFrame({"signal": [1.0, 2.0, 3.0]})
    manager._append_chain_features(
        features, "a", pd.Series(["Low", "Medium", "High"])
    )
    assert features["__zshar_logistic_chain__a"].tolist() == [0.0, 0.5, 1.0]
