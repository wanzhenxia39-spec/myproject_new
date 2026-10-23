from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from zshar.models.base import ordinal_mae
from zshar.ordinal import resolve_ordinal_values
from zshar.semantics.prototypes import match_prototype, probability_row


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_pamap2_a1_gzsl_autogluon_default_ensemble_level_ablation_gpu0.py"
)
SPEC = importlib.util.spec_from_file_location("pamap2_gzsl_level_ablation", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_supported_ordinal_schemas_are_normalized():
    assert resolve_ordinal_values(["Low", "High"]) == {"Low": 0.0, "High": 1.0}
    assert resolve_ordinal_values(["Low", "Medium", "High"]) == {
        "Low": 0.0,
        "Medium": 0.5,
        "High": 1.0,
    }
    assert resolve_ordinal_values(runner.LEVEL_LABELS[5]) == {
        "Very Low": 0.0,
        "Low": 0.25,
        "Medium": 0.5,
        "High": 0.75,
        "Very High": 1.0,
    }


def test_probability_matching_preserves_five_level_columns():
    table = pd.DataFrame(
        [{"Very Low": 0.0, "Low": 1.0, "Medium": 0.0, "High": 0.0, "Very High": 0.0}],
        index=[7],
    )
    distribution = probability_row({"motion": table}, "motion", 7)
    assert distribution == {
        "Very Low": 0.0,
        "Low": 1.0,
        "Medium": 0.0,
        "High": 0.0,
        "Very High": 0.0,
    }
    match = match_prototype(
        predicted_values={"motion": "Low"},
        prototype_values={"motion": "High"},
        attribute_cols=["motion"],
        metric="ordinal_probability",
        attribute_weights={"motion": 1.0},
        probability_distributions={"motion": distribution},
    )
    assert np.isclose(match.distance, 0.5)
    assert np.isclose(match.score, 0.5)


def test_ordinal_mae_supports_five_levels_without_changing_three_levels():
    assert np.isclose(ordinal_mae(["Very Low"], ["Very High"]), 1.0)
    assert np.isclose(ordinal_mae(["Low"], ["High"]), 1.0)
    assert np.isclose(ordinal_mae(["Low", "Medium"], ["Medium", "High"]), 0.5)


def test_level_runner_uses_matched_calibrated_gzsl_protocol(tmp_path):
    for level_count in (2, 5):
        args = runner.build_parser().parse_args(
            [
                "check",
                "--level-count",
                str(level_count),
                "--run-root",
                str(tmp_path),
                "--run-name",
                f"test_{level_count}",
            ]
        )
        config = runner.experiment_config(args, [1, 2])
        spec = runner.dataset_spec(level_count)

        assert args.gpu_id == 0
        assert config.dataset == "pamap2"
        assert config.scenario == "gzsl"
        assert config.backend == "autogluon"
        assert config.backend_config["model_suite"] == "default"
        assert config.backend_config["fit_weighted_ensemble"] is True
        assert config.backend_config["num_bag_folds"] == 0
        assert config.attribute_order_config is None
        assert config.prototype_metric == "ordinal_probability"
        assert config.calibration_mode == "grid"
        assert config.gzsl_bias_correction == "temperature_seen_penalty"
        assert config.samples_per_activity == 20
        assert spec.attributes_path == runner.VECTOR_PATHS[level_count].resolve()


def test_two_level_vectors_are_the_documented_five_level_collapse():
    runner._validate_level_vectors(runner.dataset_spec(2), 2)
    runner._validate_level_vectors(runner.dataset_spec(5), 5)
