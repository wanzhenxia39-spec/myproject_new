from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_mhealth_a1_gzsl_autogluon_default_ensemble_seen_penalty_only.py"
)
SPEC = importlib.util.spec_from_file_location("mhealth_gzsl_seen_penalty_only", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_seen_penalty_only_config_matches_base_protocol(tmp_path):
    args = runner.build_parser().parse_args(
        ["check", "--run-root", str(tmp_path), "--run-name", "test_run"]
    )
    config = runner.experiment_config(args, [13, 16])

    assert config.dataset == "mhealth"
    assert config.scenario == "gzsl"
    assert config.backend == "autogluon"
    assert config.backend_config["model_suite"] == "default"
    assert config.backend_config["fit_weighted_ensemble"] is True
    assert config.attribute_vectors_root is None
    assert config.attribute_order_config is None
    assert config.calibration_mode == "grid"
    assert config.calibration_class_folds == 3
    assert config.calibration_samples_per_activity == 3
    assert config.gzsl_bias_correction == "seen_penalty_only"
    assert config.temperature == 1.0
    assert tuple(config.calibration_temperature_grid) == (1.0,)
    assert config.gamma == 0.0
    assert config.gamma_grid is not None
    assert config.samples_per_activity == 20


def test_seen_penalty_only_reuses_exact_base_json_manifests():
    assert runner.base.SPLITS == runner.base.matched.SPLITS
    assert runner.base.SAMPLES == runner.base.matched.SAMPLES
    assert runner.base.A1_VECTORS.name == (
        "activity_attribute_vectors_initial_mhealth_v1_5.json"
    )
