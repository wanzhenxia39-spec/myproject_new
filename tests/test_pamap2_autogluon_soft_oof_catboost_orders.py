from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_pamap2_autogluon_soft_oof_catboost_orders.py"
)
SPEC = importlib.util.spec_from_file_location("pamap2_soft_oof_catboost_orders", SCRIPT)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        gpu_id=0,
        run_root=tmp_path,
        run_name_prefix="test_soft_oof_catboost_order",
        split_indices=None,
        time_limit=600,
        presets="medium_quality",
        num_cpus=8,
        num_bag_folds=5,
        resume=True,
        dry_run=False,
    )


def test_configs_match_soft_oof_protocol_and_run_both_directions(tmp_path):
    values = args(tmp_path)
    configs = [
        experiment.experiment_config(values, variant, [1, 2])
        for variant in experiment.VARIANTS
    ]
    assert experiment.VARIANTS == ("forward", "reverse")
    assert [config.attribute_order_variant for config in configs] == [
        "forward",
        "reverse",
    ]
    assert all(config.backend == "autogluon_soft_oof" for config in configs)
    assert all(
        config.attribute_order_config == str(experiment.CATBOOST_ORDERS)
        for config in configs
    )
    assert all(config.prototype_metric == "ordinal_probability" for config in configs)
    assert all(config.attribute_weighting == "validation_accuracy" for config in configs)
    assert all(config.backend_config["num_bag_folds"] == 5 for config in configs)
    assert all(config.backend_config["time_limit"] == 600 for config in configs)
    assert all(
        config.backend_config["uncertainty_metric"] == "normalized_entropy"
        for config in configs
    )


def test_summarize_writes_paired_reverse_minus_forward_metrics(tmp_path):
    values = args(tmp_path)
    configs = [
        experiment.experiment_config(values, variant, [1])
        for variant in experiment.VARIANTS
    ]
    common = {
        "backend": "autogluon_soft_oof",
        "seen_activities": "seen",
        "unseen_activities": "unseen",
        "eval_samples": 200,
        "attribute_vectors_sha256": "vectors",
        "attribute_order_sha256": "orders",
        "attribute_exact_match": 0.2,
    }
    for config in configs:
        variant = config.attribute_order_variant
        directory = Path(config.run_root) / "pamap2" / str(config.run_name)
        directory.mkdir(parents=True)
        value = 0.5 if variant == "forward" else 0.6
        pd.DataFrame(
            [
                {
                    **common,
                    "split_index": 1,
                    "attribute_order_variant": variant,
                    "activity_top1_accuracy": value,
                    "activity_top3_recall": value,
                    "activity_macro_f1": value,
                }
            ]
        ).to_csv(directory / "summary.csv", index=False)

    output = experiment.summarize(values, configs, [1])
    paired = pd.read_csv(output / "paired_split_metrics.csv")
    assert paired.loc[
        0, "reverse_minus_forward_activity_top1_accuracy"
    ] == pytest.approx(0.1)
    payload = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert payload["ranking_backend"] == "catboost"
    assert payload["order_sequence"] == ["forward", "reverse"]
