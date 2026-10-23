from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_tridataset_initial_a1_cross_subject_autogluon_default.py"
)
SPEC = importlib.util.spec_from_file_location(
    "initial_a1_cross_subject_autogluon_default", SCRIPT
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_protocol_uses_three_initial_a1_v1_5_vector_files_and_subject_manifests():
    assert set(runner.DATASETS) == {"pamap2", "mhealth", "wisdm"}
    for dataset, paths in runner.DATASETS.items():
        assert paths["vectors"].name == (
            f"activity_attribute_vectors_initial_{dataset}_v1_5.json"
        )
        assert paths["subjects"].name == (
            f"{dataset}_subject80_20_balanced_seed42.json"
        )


def test_backend_matches_independent_refined_a2_default_ensemble_configuration():
    args = runner.build_parser().parse_args(
        ["check", "--dataset", "pamap2", "--gpu-id", "0"]
    )
    config = runner.backend_config(args)
    assert config["presets"] == "medium_quality"
    assert config["model_suite"] == "default"
    assert config["num_bag_folds"] == 2
    assert config["num_stack_levels"] == 0
    assert config["fit_weighted_ensemble"] is True
    assert "label_order" not in config
    assert "consider_labels_correlation" not in config
