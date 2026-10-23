from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_pamap2_logistic_chain_probability_orders.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pamap2_logistic_probability_orders", SCRIPT
)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def test_saved_probability_chain_protocol_covers_all_twenty_splits():
    indices = experiment.validate_probability_chain_protocol()
    assert len(indices) == len(set(indices)) == 20
    assert experiment.file_hash(experiment.PROBABILITY_ORDERS) == (
        experiment.EXPECTED_ORDER_SHA256
    )


def test_lr_chain_configs_reuse_probability_order_file_and_both_directions(tmp_path):
    experiment.common.ORDERS = experiment.PROBABILITY_ORDERS
    values = argparse.Namespace(
        run_root=tmp_path,
        run_name_prefix=experiment.RUN_NAME_PREFIX,
        split_indices=None,
        max_iter=3000,
        c=1.0,
        resume=True,
        dry_run=False,
    )
    configs = [
        experiment.common.experiment_config(values, variant)
        for variant in experiment.common.VARIANTS
    ]
    assert [config.attribute_order_variant for config in configs] == [
        "forward",
        "reverse",
    ]
    assert all(config.backend == "logistic_regression_chain" for config in configs)
    assert all(
        config.attribute_order_config == str(experiment.PROBABILITY_ORDERS)
        for config in configs
    )
    assert all(config.prototype_metric == "ordinal_probability" for config in configs)
    assert all(config.attribute_weighting == "validation_accuracy" for config in configs)
