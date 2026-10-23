from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_tridataset_a2_cross_subject_autogluon_default.py"
)
SPEC = importlib.util.spec_from_file_location("cross_subject_autogluon_default", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_protocol_covers_three_requested_datasets_without_order_dependencies():
    assert set(runner.DATASETS) == {"pamap2", "mhealth", "wisdm"}
    assert all(set(paths) == {"vectors", "subjects"} for paths in runner.DATASETS.values())


def test_backend_is_independent_autogluon_default_with_two_bag_folds():
    args = runner.build_parser().parse_args(
        ["check", "--dataset", "pamap2", "--gpu-id", "0"]
    )
    config = runner.backend_config(args)
    assert config["model_suite"] == "default"
    assert config["num_bag_folds"] == 2
    assert config["fit_weighted_ensemble"] is True
    assert "label_order" not in config
    assert "consider_labels_correlation" not in config
