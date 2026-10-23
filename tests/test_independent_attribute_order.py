from __future__ import annotations

import pandas as pd

from zshar.models.factory import IndependentAttributeModelManager


def test_independent_manager_trains_and_restores_explicit_label_order(tmp_path):
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    Y = pd.DataFrame(
        {
            "first": ["Low", "Low", "Low", "High", "High", "High"],
            "second": ["High", "High", "Low", "Low", "High", "Low"],
        }
    )
    manager = IndependentAttributeModelManager(
        "logistic_regression",
        attribute_cols=Y.columns,
        backend_config={"label_order": ["second", "first"], "max_iter": 100},
    )

    manager.fit_all(X.iloc[:4], Y.iloc[:4], X.iloc[4:], Y.iloc[4:], tmp_path)

    assert manager.attribute_cols == ["second", "first"]
    assert manager.label_order == ["second", "first"]
    loaded = IndependentAttributeModelManager.load(tmp_path)
    assert loaded.attribute_cols == ["second", "first"]
    assert loaded.label_order == ["second", "first"]
