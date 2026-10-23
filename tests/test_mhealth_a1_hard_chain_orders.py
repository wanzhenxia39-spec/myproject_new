from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts/run_mhealth_a1_hard_chain_orders.py"
SPEC = importlib.util.spec_from_file_location("mhealth_a1_hard_chain_orders", SCRIPT)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def test_materialized_split_vectors_are_exact_a1_copies(tmp_path, monkeypatch):
    a1 = tmp_path / "a1.json"
    payload = {
        "walking": {f"a{i}": ("Low", "Medium", "High")[i % 3] for i in range(9)}
    }
    a1.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    output = tmp_path / "vectors"
    monkeypatch.setattr(experiment, "A1", a1)
    monkeypatch.setattr(experiment, "A1_SPLIT_ROOT", output)
    splits = [argparse.Namespace(index=13), argparse.Namespace(index=16)]

    experiment.materialize_a1_split_vectors(splits)

    for split in splits:
        copied = output / f"split_{split.index}/activity_attribute_vectors_a2.json"
        assert copied.read_bytes() == a1.read_bytes()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["attribute_set"] == "initial_A1_v1_5"
    assert manifest["split_count"] == 2


def test_rank_attributes_places_constant_last_and_reverse_is_exact():
    labels = pd.DataFrame(
        {
            "constant": ["Low"] * 6,
            "strong": ["Low", "Low", "Medium", "Medium", "High", "High"],
            "weak": ["Low", "High", "Low", "High", "Low", "High"],
        }
    )
    report = {
        "per_attribute": {
            "constant": {"balanced_accuracy": 1.0, "macro_f1": 1.0, "ordinal_mae": 0.0},
            "strong": {"balanced_accuracy": 0.9, "macro_f1": 0.9, "ordinal_mae": 0.1},
            "weak": {"balanced_accuracy": 0.3, "macro_f1": 0.3, "ordinal_mae": 0.7},
        }
    }

    result = experiment.rank_attributes(labels, report)

    assert result["forward"][-1] == "constant"
    assert result["forward"].index("strong") < result["forward"].index("weak")
    assert result["reverse"] == result["forward"][::-1]


def test_runner_command_uses_a1_root_hard_chain_and_gpu1(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "A1_SPLIT_ROOT", tmp_path / "a1_vectors")
    monkeypatch.setattr(experiment, "ORDER_FILE", tmp_path / "orders.json")
    args = argparse.Namespace(
        time_limit=600,
        order_time_limit=300,
        presets="medium_quality",
        num_cpus=8,
        num_bag_folds=0,
        model_suite="default",
        verbosity=2,
        run_root=tmp_path / "runs",
        resume=True,
    )

    command = experiment.runner_command(args, "reverse", "train-test")

    assert command[command.index("--backend") + 1] == "autogluon_multilabel"
    assert command[command.index("--attribute-vectors-root") + 1] == str(tmp_path / "a1_vectors")
    assert command[command.index("--attribute-order-variant") + 1] == "reverse"
    assert command[command.index("--gpu-id") + 1] == "1"
    assert "--reuse-models" in command
