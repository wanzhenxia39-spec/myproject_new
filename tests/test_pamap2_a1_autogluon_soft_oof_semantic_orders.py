from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_pamap2_a1_autogluon_soft_oof_semantic_dependency_orders.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pamap2_a1_soft_oof_semantic_orders", SCRIPT
)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def test_a1_soft_oof_uses_profile_vectors_and_both_dp_orders(tmp_path):
    experiment.configure_base()
    _, _, indices = experiment.validate_protocol()
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
        for variant in experiment.base.VARIANTS
    ]
    assert [config.attribute_order_variant for config in configs] == [
        "forward",
        "reverse",
    ]
    assert all(config.backend == "autogluon_soft_oof" for config in configs)
    assert all(config.attribute_vectors_root is None for config in configs)
    assert all(config.backend_config["num_bag_folds"] == 2 for config in configs)
    assert all(config.backend_config["time_limit"] == 300 for config in configs)
    assert all("a1_v1_5" in str(config.run_name) for config in configs)
