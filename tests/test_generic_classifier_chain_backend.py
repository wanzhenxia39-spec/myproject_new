from __future__ import annotations

import pandas as pd

from zshar.models.factory import create_model_manager, load_model_manager


def test_generic_chain_uses_requested_order_and_restores_base_backend(tmp_path):
    X = pd.DataFrame(
        {"x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, -2.5, 2.5]}
    )
    Y = pd.DataFrame(
        {
            "first": ["Low", "Low", "Low", "High", "High", "High", "Low", "High"],
            "second": ["High", "High", "Low", "Low", "High", "High", "Low", "Low"],
        }
    )
    manager = create_model_manager(
        "generic_classifier_chain",
        attribute_cols=Y.columns,
        backend_config={
            "base_backend": "extra_trees",
            "label_order": ["second", "first"],
            "n_estimators": 10,
            "n_jobs": 1,
            "random_state": 42,
        },
    )

    manager.fit_all(X.iloc[:6], Y.iloc[:6], X.iloc[6:], Y.iloc[6:], tmp_path)

    assert manager.label_order == ["second", "first"]
    assert manager.base_backend == "extra_trees"
    assert list(manager.predict_all(X.iloc[6:]).columns) == ["first", "second"]
    loaded = load_model_manager(tmp_path, expected_backend="generic_classifier_chain")
    assert loaded.label_order == ["second", "first"]
    assert loaded.base_backend == "extra_trees"
