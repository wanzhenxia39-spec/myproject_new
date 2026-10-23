from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_pamap2_a1_autogluon_soft_oof_catboost_orders.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pamap2_a1_soft_oof_catboost_orders", SCRIPT
)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def test_a1_catboost_order_soft_oof_uses_profile_vectors_and_both_orders(tmp_path):
    experiment.configure_base()
    _, _, table, indices = experiment.preflight()
    assert len(table.attribute_columns) == 9
    args = argparse.Namespace(
        gpu_id=0,
        run_root=tmp_path,
        run_name_prefix=experiment.base.RUN_NAME_PREFIX,
        split_indices=None,
        time_limit=300,
        presets="medium_quality",
        num_cpus=8,
        num_bag_folds=2,
        resume=True,
        dry_run=False,
    )
    configs = [
        experiment.base.experiment_config(args, variant, indices[:2])
        for variant in experiment.VARIANTS
    ]
    assert [config.attribute_order_variant for config in configs] == [
        "forward",
        "reverse",
    ]
    assert all(config.backend == "autogluon_soft_oof" for config in configs)
    assert all(config.attribute_vectors_root is None for config in configs)
    assert all(config.backend_config["num_bag_folds"] == 2 for config in configs)
    assert all(config.backend_config["time_limit"] == 300 for config in configs)
    assert all("a1_v1_5_catboost_seen_only" in str(config.run_name) for config in configs)


def test_catboost_ordering_protocol_is_seen_only_and_independent():
    assert "seen-class sensor samples only" in experiment.SELECTION_SCOPE
    assert "independent per-attribute CatBoost" in experiment.SELECTION_SCOPE
    assert experiment.WEIGHTS == {
        "validation_balanced_accuracy": 0.45,
        "validation_macro_f1": 0.30,
        "inverse_normalized_ordinal_mae": 0.20,
        "mean_dependency_nmi": 0.05,
    }
