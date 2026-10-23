from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zshar.models.factory import IndependentAttributeModelManager
from zshar.pipeline.training import load_or_train_attribute_models


def test_real_autogluon_bound_features_survive_reload_and_isolate_attributes(tmp_path):
    pytest.importorskip("autogluon.tabular")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["posture", "rotation", "irrelevant"])
    Y = pd.DataFrame({
        "inclination": np.where(X.posture > 0, "High", "Low"),
        "rotational_motion": np.where(X.rotation > 0, "High", "Low"),
        "constant": "Low",
    })
    mapping = {"inclination": ["posture"], "rotational_motion": ["rotation"],
               "constant": ["irrelevant"]}
    manager = IndependentAttributeModelManager("autogluon", backend_config={
        "attribute_feature_columns": mapping,
        "num_gpus": 0, "num_cpus": 1, "time_limit": 20,
        # Small real ensemble exercises fit/predict/serialization without GPU training.
        "fit_kwargs": {"hyperparameters": {
            "RF": {"n_estimators": 8}, "XT": {"n_estimators": 8},
        }},
    })
    manager.fit_all(X.iloc[:80], Y.iloc[:80], X.iloc[80:], Y.iloc[80:],
                    root_path=tmp_path / "models")
    loaded = IndependentAttributeModelManager.load(tmp_path / "models")
    assert loaded.attribute_feature_columns == mapping
    for attr, columns in mapping.items():
        assert loaded.models[attr].feature_names == columns
    assert loaded.models["inclination"].get_metadata()["weighted_ensemble_exists"]
    original = manager.predict_proba_all(X.iloc[80:])
    reloaded = loaded.predict_proba_all(X.iloc[80:])
    for attr in Y.columns:
        pd.testing.assert_frame_equal(original[attr], reloaded[attr])
    altered = X.iloc[80:].copy()
    altered["rotation"] += 1000
    altered["irrelevant"] -= 1000
    pd.testing.assert_frame_equal(reloaded["inclination"],
                                  loaded.predict_proba_all(altered)["inclination"])
    with pytest.raises(ValueError, match="bound features"):
        loaded.predict_all(X.drop(columns="posture"))
    with pytest.raises(ValueError, match="Missing trained features"):
        loaded.models["inclination"].predict_proba(X.drop(columns="posture"))
    # A model loaded directly also removes unrelated columns, not just the manager.
    pd.testing.assert_frame_equal(reloaded["inclination"],
                                  loaded.models["inclination"].predict_proba(altered))


@pytest.mark.parametrize("saved,requested", [
    ({}, {"a": ["x"]}),
    ({"a": ["x"]}, {}),
    ({"a": ["x"]}, {"a": ["y"]}),
])
def test_resume_rejects_changed_feature_mapping(tmp_path, monkeypatch, saved, requested):
    from types import SimpleNamespace
    monkeypatch.setattr("zshar.pipeline.training.load_model_manager",
                        lambda *a, **k: SimpleNamespace(attribute_feature_columns=saved))
    with pytest.raises(ValueError, match="feature bindings differ"):
        load_or_train_attribute_models(
            reuse_models=True, model_path=tmp_path, frame=pd.DataFrame(),
            feature_columns=["x", "y"], attribute_columns=["a"], backend="autogluon",
            backend_config={"attribute_feature_columns": requested}, seed=42,
        )
