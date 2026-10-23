from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_mhealth_a1_backend_benchmark.py"
SPEC = importlib.util.spec_from_file_location("mhealth_a1_backend_benchmark", SCRIPT)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def args(tmp_path: Path, *, resume: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        num_cpus=8,
        run_root=tmp_path / "runs",
        protocol_root=tmp_path / "protocol",
        comparison_root=tmp_path / "comparison",
        gpu_id=1,
        feature_subsets=False,
        primary_groups_only=False,
        resume=resume,
    )


def test_backend_set_matches_requested_non_autogluon_backends():
    assert len(experiment.BACKENDS) == 13
    assert "autogluon" not in experiment.BACKENDS
    assert set(experiment.BACKENDS) == {
        "lightgbm",
        "xgboost",
        "catboost",
        "logistic_regression",
        "linear_svm",
        "extra_trees",
        "random_forest",
        "neuralnet_fastai",
        "neuralnet_torch",
        "ordinal_logistic",
        "coral_mlp",
        "mlp_independent",
        "mlp_multitask",
    }


def test_backend_gpu_and_probability_configs():
    assert experiment.backend_config("lightgbm", 8)["device"] == "cuda"
    assert experiment.backend_config("xgboost", 8)["device"] == "cuda"
    assert experiment.backend_config("catboost", 8)["devices"] == "0"
    assert experiment.backend_config("linear_svm", 8)["calibrate_proba"] is True
    for backend in (
        "neuralnet_fastai",
        "neuralnet_torch",
        "ordinal_logistic",
        "coral_mlp",
        "mlp_independent",
        "mlp_multitask",
    ):
        assert experiment.backend_config(backend, 8)["device"] == "cuda:0"


def test_runner_command_locks_a1_seed42_protocol_and_gpu1(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "A1_SPLIT_ROOT", tmp_path / "vectors")
    command = experiment.runner_command(args(tmp_path), "linear_svm", "train-test")

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value("--backend") == "linear_svm"
    assert value("--attribute-vectors-root") == str(tmp_path / "vectors")
    assert value("--split-manifest") == str(experiment.SPLITS)
    assert value("--sample-manifest") == str(experiment.SAMPLES)
    assert value("--samples-per-activity") == "30"
    assert value("--prototype-metric") == "ordinal_probability"
    assert value("--attribute-weighting") == "validation_accuracy"
    assert value("--gpu-id") == "1"
    assert "--reuse-models" in command


def test_runner_command_enables_bound_feature_subsets_on_selected_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "A1_SPLIT_ROOT", tmp_path / "vectors")
    values = args(tmp_path)
    values.gpu_id = 0
    values.feature_subsets = True
    command = experiment.runner_command(values, "logistic_regression", "train-test")

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value("--gpu-id") == "0"
    assert value("--attribute-feature-bindings") == str(experiment.A1_BINDINGS)
    assert "bound_feature_subsets" in value("--run-name")


def test_autogluon_default_ensemble_uses_matched_bound_protocol(tmp_path):
    values = args(tmp_path)
    values.gpu_id = 0
    values.feature_subsets = True
    values.autogluon_time_limit = 600
    values.autogluon_bag_folds = 0
    assert experiment.parse_backends("autogluon") == ["autogluon"]
    command = experiment.runner_command(values, "autogluon", "train-test")
    config = json.loads(command[command.index("--backend-config") + 1])
    assert config["model_suite"] == "default"
    assert config["fit_weighted_ensemble"] is True
    assert config["num_bag_folds"] == config["num_stack_levels"] == 0
    assert config["lightgbm_device"] == "cuda"
    assert config["num_gpus"] == 1
    assert config["time_limit"] == 600
    assert command[command.index("--attribute-feature-bindings") + 1] == str(experiment.A1_BINDINGS)
    assert command[command.index("--gpu-id") + 1] == "0"
    assert experiment.expected_config(values, "autogluon")["backend_config"] == config
    evaluation = experiment.runner_command(values, "autogluon", "evaluate")
    assert "--evaluate-only" in evaluation and "--reuse-models" in evaluation
