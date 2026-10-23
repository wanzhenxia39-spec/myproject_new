import numpy as np
import pandas as pd
import pytest

from zshar.models.autogluon_soft_oof_backend import (
    AutoGluonSoftOOFModelManager,
)


def test_soft_oof_uses_fast_leakage_safe_defaults():
    manager = AutoGluonSoftOOFModelManager(attribute_cols=["a", "b"])
    assert manager.backend_config["num_bag_folds"] == 2
    assert manager.backend_config["time_limit"] == 300


def test_soft_oof_defaults_remain_overridable():
    manager = AutoGluonSoftOOFModelManager(
        attribute_cols=["a", "b"], num_bag_folds=4, time_limit=120
    )
    assert manager.backend_config["num_bag_folds"] == 4
    assert manager.backend_config["time_limit"] == 120


def test_probability_normalization_tolerates_float32_boundary_drift():
    table = pd.DataFrame(
        {
            "High": [-np.finfo(np.float32).eps, 0.25],
            "Low": [1.0 + np.finfo(np.float32).eps, 0.75],
        }
    )

    normalized = AutoGluonSoftOOFModelManager._normalize_probabilities(
        table, index=table.index, classes=["High", "Low"]
    )

    np.testing.assert_allclose(normalized.to_numpy(), [[0.0, 1.0], [0.25, 0.75]])
    np.testing.assert_allclose(normalized.sum(axis=1).to_numpy(), 1.0)


@pytest.mark.parametrize(
    "row",
    [(-1e-3, 1.001), (0.5, 1.01)],
)
def test_probability_normalization_rejects_materially_invalid_values(row):
    table = pd.DataFrame([row], columns=["High", "Low"])

    with pytest.raises(RuntimeError, match=r"outside \[0, 1\] beyond"):
        AutoGluonSoftOOFModelManager._normalize_probabilities(
            table, index=table.index, classes=["High", "Low"]
        )
