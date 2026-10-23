from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_pamap2_a1_gzsl_autogluon_default_ensemble_no_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("pamap2_gzsl_no_calibration", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_no_calibration_config_matches_base_protocol(tmp_path):
    args = runner.build_parser().parse_args(
        ["check", "--run-root", str(tmp_path), "--run-name", "test_run"]
    )
    config = runner.experiment_config(args, [13, 16])

    assert args.gpu_id == 0
    assert config.dataset == "pamap2"
    assert config.scenario == "gzsl"
    assert config.backend == "autogluon"
    assert config.backend_config["model_suite"] == "default"
    assert config.backend_config["fit_weighted_ensemble"] is True
    assert config.backend_config["num_bag_folds"] == 0
    assert config.attribute_vectors_root is None
    assert config.attribute_order_config is None
    assert config.calibration_mode == "fixed"
    assert config.gzsl_bias_correction == "none"
    assert config.temperature == 1.0
    assert config.gamma == 0.0
    assert config.gamma_grid is None
    assert tuple(config.calibration_temperature_grid) == (1.0,)
    assert config.samples_per_activity == 20


def test_no_calibration_reuses_exact_base_json_manifests_and_initial_a1():
    assert runner.base.SPLITS.name == (
        "pamap2_unseen4_random20_gzsl_splits_seed42.json"
    )
    assert runner.base.SAMPLES.name == (
        "pamap2_unseen4_random20_gzsl_eval20_seen20_samples_seed42.json"
    )
    assert runner.base.ATTRIBUTE_SET == "initial_A1_v1_5_9d"
    assert runner.base.A1_VECTORS.name == (
        "activity_attribute_vectors_initial_pamap2_v1_5.json"
    )
