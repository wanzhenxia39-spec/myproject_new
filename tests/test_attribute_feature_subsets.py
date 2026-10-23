from __future__ import annotations

import numpy as np
import pandas as pd

from zshar.models.factory import IndependentAttributeModelManager
from zshar.models.multitask_mlp_backend import build_multitask_network


def test_independent_manager_trains_and_predicts_from_each_bound_subset(tmp_path):
    X = pd.DataFrame(
        {
            "g1_a": [-2, -1, 1, 2, -2, -1, 1, 2],
            "g1_b": [0, 0, 1, 1, 0, 0, 1, 1],
            "g2_a": [2, 1, -1, -2, 2, 1, -1, -2],
            "g2_b": [1, 1, 0, 0, 1, 1, 0, 0],
        }
    )
    Y = pd.DataFrame(
        {
            "posture": ["Low", "Low", "High", "High"] * 2,
            "rotation": ["High", "High", "Low", "Low"] * 2,
        }
    )
    manager = IndependentAttributeModelManager(
        "logistic_regression",
        attribute_cols=Y.columns,
        backend_config={
            "attribute_feature_columns": {
                "posture": ["g1_a", "g1_b"],
                "rotation": ["g2_a", "g2_b"],
            }
        },
    )
    manager.fit_all(X, Y, X, Y, root_path=tmp_path / "models")

    assert manager.models["posture"].feature_names == ["g1_a", "g1_b"]
    assert manager.models["rotation"].feature_names == ["g2_a", "g2_b"]
    assert list(manager.predict_all(X).columns) == ["posture", "rotation"]

    loaded = IndependentAttributeModelManager.load(tmp_path / "models")
    assert loaded.attribute_feature_columns == manager.attribute_feature_columns
    pd.testing.assert_frame_equal(loaded.predict_all(X), manager.predict_all(X))


def test_multitask_heads_mask_features_outside_their_bound_subset():
    import torch

    torch.manual_seed(7)
    network = build_multitask_network(
        input_dim=4,
        hidden_dims=(3,),
        head_dims={"head_0": 2, "head_1": 2},
        dropout=0.0,
        head_feature_indices={"head_0": [0, 1], "head_1": [2, 3]},
    ).eval()
    baseline = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    changed_second_group = torch.tensor([[1.0, 2.0, 30.0, 40.0]])
    changed_first_group = torch.tensor([[10.0, 20.0, 3.0, 4.0]])

    with torch.no_grad():
        base = network(baseline)
        second = network(changed_second_group)
        first = network(changed_first_group)

    assert torch.allclose(base["head_0"], second["head_0"])
    assert torch.allclose(base["head_1"], first["head_1"])
    assert not torch.allclose(base["head_0"], first["head_0"])
    assert not torch.allclose(base["head_1"], second["head_1"])
