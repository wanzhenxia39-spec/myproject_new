from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_pamap2_autogluon_soft_oof_semantic_dependency_orders.py"
)
SPEC = importlib.util.spec_from_file_location("pamap2_soft_oof_semantic_orders", SCRIPT)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def test_soft_oof_uses_saved_semantic_dp_forward_and_reverse(tmp_path):
    experiment.configure_base()
    args = argparse.Namespace(
        gpu_id=0,
        run_root=tmp_path,
        run_name_prefix=experiment.base.RUN_NAME_PREFIX,
        split_indices=None,
        time_limit=600,
        presets="medium_quality",
        num_cpus=8,
        num_bag_folds=5,
        resume=True,
        dry_run=False,
    )
    configs = [
        experiment.base.experiment_config(args, variant, [1, 2])
        for variant in experiment.base.VARIANTS
    ]
    assert [config.attribute_order_variant for config in configs] == [
        "forward",
        "reverse",
    ]
    assert all(config.backend == "autogluon_soft_oof" for config in configs)
    assert all(
        config.attribute_order_config == str(experiment.SEMANTIC_ORDERS)
        for config in configs
    )
    assert all("semantic_dependency_dp" in str(config.run_name) for config in configs)
    assert experiment.base.ORDER_SUMMARY_METADATA["ranking_backend"] is None
    assert (
        experiment.base.ORDER_SUMMARY_METADATA["autogluon_used_for_order_generation"]
        is False
    )
