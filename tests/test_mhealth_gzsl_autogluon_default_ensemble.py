from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_mhealth_a2_gzsl_autogluon_default_ensemble.py"
)
SPEC = importlib.util.spec_from_file_location("mhealth_gzsl_default_ensemble", SCRIPT)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def test_default_ensemble_uses_matched_independent_gzsl_protocol(tmp_path):
    args = argparse.Namespace(
        gpu_id=0,
        num_cpus=8,
        time_limit=300,
        run_root=tmp_path,
        run_name=experiment.RUN_NAME,
        resume=True,
    )
    config = experiment.experiment_config(args, [1, 2])

    assert config.dataset == "mhealth"
    assert config.scenario == "gzsl"
    assert config.backend == "autogluon"
    assert config.attribute_order_config is None
    assert config.backend_config["model_suite"] == "default"
    assert config.backend_config["fit_weighted_ensemble"] is True
    assert config.backend_config["num_bag_folds"] == 0
    assert config.calibration_mode == "grid"
    assert config.calibration_class_folds == 3
    assert config.calibration_samples_per_activity == 3
    assert config.gzsl_bias_correction == "temperature_seen_penalty"
    assert config.samples_per_activity == 20
