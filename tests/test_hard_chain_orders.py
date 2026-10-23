from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/run_mhealth_hard_chain_orders.py"
SPEC = importlib.util.spec_from_file_location("mhealth_hard_chain_orders", SCRIPT)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_saved_order_is_projected_to_latest_schema_without_inventing_old_names(
    tmp_path, monkeypatch
):
    old = tmp_path / "old_orders.json"
    a2 = tmp_path / "a2"
    write(
        old,
        {
            "dataset": "mhealth",
            "seed": 42,
            "splits": {
                "13": {
                    "forward": ["removed", "b", "a"],
                    "reverse": ["a", "b", "removed"],
                }
            },
        },
    )
    write(
        a2 / "split_13/a2_attribute_plan.json",
        {"attributes": [{"name": "a"}, {"name": "new"}, {"name": "b"}]},
    )
    write(
        a2 / "split_13/activity_attribute_vectors_a2.json",
        {"activity": {"a": "Low", "new": "Medium", "b": "High"}},
    )
    split_manifest = tmp_path / "splits.json"
    sample_manifest = tmp_path / "samples.json"
    write(split_manifest, {"dataset": "mhealth", "seed": 42})
    write(sample_manifest, {"dataset": "mhealth", "seed": 42})
    monkeypatch.setattr(experiment, "OLD_ORDERS", old)
    monkeypatch.setattr(experiment, "A2_ROOT", a2)
    monkeypatch.setattr(experiment, "SPLITS", split_manifest)
    monkeypatch.setattr(experiment, "SAMPLES", sample_manifest)

    result = experiment.build_compatible_orders([13])
    row = result["splits"]["13"]
    assert row["forward"] == ["b", "a", "new"]
    assert row["reverse"] == ["new", "a", "b"]
    assert row["retained_from_saved_forward"] == ["b", "a"]
    assert row["new_attributes_appended_in_a2_order"] == ["new"]
    assert row["removed_old_attributes"] == ["removed"]


def test_runner_accepts_explicit_hard_chain_order(tmp_path, monkeypatch):
    from zshar.datasets import DatasetSpec
    from zshar.pipeline import runner
    from zshar.splits import ActivitySplit

    dataset = DatasetSpec(
        "mhealth",
        "MHEALTH",
        {0: "seen", 1: "unseen"},
        tmp_path / "features",
        tmp_path / "names",
        tmp_path / "labels",
        tmp_path / "attrs",
        tmp_path / "meta",
        1,
        1,
        {},
    )
    split = ActivitySplit(13, (0,), (1,), ("seen",), ("unseen",))
    monkeypatch.setattr(
        runner,
        "_resolve_manifests",
        lambda *_: ([split], {}, tmp_path / "splits", tmp_path / "samples"),
    )
    write(
        tmp_path / "a2/split_13/activity_attribute_vectors_a2.json",
        {
            "seen": {"a": "Low", "b": "High"},
            "unseen": {"a": "High", "b": "Low"},
        },
    )
    order_path = tmp_path / "orders.json"
    write(
        order_path,
        {
            "dataset": "mhealth",
            "seed": 42,
            "splits": {"13": {"forward": ["a", "b"], "reverse": ["b", "a"]}},
        },
    )
    config = runner.ExperimentConfig(
        dataset="mhealth",
        backend="autogluon_multilabel",
        attribute_vectors_root=str(tmp_path / "a2"),
        attribute_order_config=str(order_path),
        attribute_order_variant="reverse",
        run_root=str(tmp_path / "runs"),
        run_name="test",
        dry_run=True,
    )

    output = runner.run_experiment(dataset, config)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["attribute_orders"] == {"13": ["b", "a"]}


def test_runner_rejects_order_that_does_not_match_latest_schema(tmp_path):
    from zshar.pipeline.runner import _load_split_attribute_orders

    order_path = tmp_path / "orders.json"
    write(
        order_path,
        {
            "dataset": "mhealth",
            "seed": 42,
            "splits": {"13": {"forward": ["old_a"], "reverse": ["old_a"]}},
        },
    )
    with pytest.raises(ValueError, match="does not match split 13 A2 schema"):
        _load_split_attribute_orders(
            order_path,
            dataset="mhealth",
            seed=42,
            variant="forward",
            split_attributes={13: ["new_a"]},
        )
