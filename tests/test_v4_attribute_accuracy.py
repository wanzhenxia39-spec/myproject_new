import csv
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

from zshar.evaluation.attribute_summary import export_attribute_accuracy


def _metrics(root, index, values, count=200, vector_hash="saved-vectors"):
    directory = root / f"split_{index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps({
        "sample_count": count,
        "per_attribute": {name: {"accuracy": value} for name, value in values.items()},
        "attribute_vectors": {"sha256": vector_hash},
    }))


def test_attribute_coverage_and_equal_split_weighting(tmp_path):
    _metrics(tmp_path, 13, {"cadence": 0.1, "impact": 0.3})
    _metrics(tmp_path, 48, {"cadence": 0.8})
    summary = export_attribute_accuracy(tmp_path, expected_split_indices=[13, 48], require_complete=True)
    assert summary["complete"]
    assert summary["attribute_mean_accuracy"] == pytest.approx(0.5)
    attributes = {row["attribute"]: row for row in summary["per_attribute"]}
    assert attributes["cadence"]["mean_accuracy"] == pytest.approx(0.45)
    assert attributes["cadence"]["present_split_count"] == 2
    assert attributes["impact"]["mean_accuracy"] == 0.3
    assert attributes["impact"]["present_split_count"] == 1
    assert attributes["impact"]["coverage"] == 0.5
    assert json.loads(attributes["impact"]["absent_split_indices"]) == [48]
    with (tmp_path / "attribute_accuracy_by_split.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_partial_metrics_are_not_reported_as_twenty_split_completion(tmp_path):
    _metrics(tmp_path, 13, {"cadence": 0.8})
    summary = export_attribute_accuracy(tmp_path, expected_split_indices=[13, 48])
    assert summary["complete"] is False
    assert summary["completed_split_count"] == 1
    assert summary["missing_split_indices"] == [48]
    assert json.loads(summary["per_attribute"][0]["unevaluated_split_indices"]) == [48]
    with pytest.raises(ValueError, match="Missing test metrics"):
        export_attribute_accuracy(tmp_path, expected_split_indices=[13, 48], require_complete=True)


def test_summary_rejects_stale_vectors_and_wrong_test_count(tmp_path):
    _metrics(tmp_path, 13, {"cadence": 0.8})
    inputs = {13: {"sha256": "changed-vectors", "attributes": ["cadence"]}}
    with pytest.raises(ValueError, match="different vectors"):
        export_attribute_accuracy(tmp_path, expected_split_indices=[13], vector_inputs=inputs)
    with pytest.raises(ValueError, match="sample count"):
        export_attribute_accuracy(tmp_path, expected_split_indices=[13], expected_sample_count=120)


def _launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/attributes_v4_autogluon_common.py"
    spec = importlib.util.spec_from_file_location("v4_autogluon_common_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dataset,gpu,samples", [("pamap2", 0, 50), ("mhealth", 1, 30), ("uschad", 0, 30), ("wisdm", 1, 30), ("dsads", 0, 30)])
def test_v4_protocol_uses_requested_vectors_manifests_and_default_ensemble(dataset, gpu, samples):
    launcher = _launcher()
    args = launcher.build_parser(dataset).parse_args([])
    assert args.gpu_id == gpu
    assert args.attribute_vectors_root == launcher.PROJECT / "configs/attrIbutes_v4" / dataset
    args.run_name = launcher.default_run_name(dataset, args.time_limit)
    config = launcher.experiment_config(dataset, args, [13, 48])
    assert config.scenario == "zsl" and config.seed == 42 and config.backend == "autogluon"
    assert config.samples_per_activity == samples
    assert config.attribute_vectors_root.endswith(f"configs/attrIbutes_v4/{dataset}")
    assert config.sample_manifest.endswith(
        f"{dataset}_unseen{launcher.UNSEEN_COUNTS[dataset]}_random20_eval{samples}_samples_seed42.json")
    assert config.prototype_metric == "ordinal_probability"
    assert config.attribute_weighting == "validation_accuracy"
    assert config.backend_config["model_suite"] == "default"
    assert config.backend_config["fit_weighted_ensemble"] is True
    assert config.backend_config["num_bag_folds"] == config.backend_config["num_stack_levels"] == 0


def test_resume_accepts_unordered_serialized_splits_but_rejects_changed_inputs(tmp_path):
    launcher = _launcher()
    args = launcher.build_parser("pamap2").parse_args(["--resume"])
    args.run_name = "unit-test"
    config = launcher.experiment_config("pamap2", args, [13, 48])
    saved = launcher.normalize(asdict(config))
    saved["split_indices"] = [48, 13]
    saved["reuse_models"] = False
    (tmp_path / "config.json").write_text(json.dumps(saved))
    snapshot = {"vectors": {13: {"sha256": "original"}}}
    (tmp_path / "v4_input_manifest.json").write_text(json.dumps(launcher.normalize(snapshot)))
    launcher.validate_existing_run(tmp_path, config, snapshot)
    with pytest.raises(ValueError, match="inputs changed"):
        launcher.validate_existing_run(tmp_path, config, {"vectors": {13: {"sha256": "changed"}}})


def test_missing_vectors_fail_fast_without_wait_flag(tmp_path):
    from types import SimpleNamespace
    launcher = _launcher()
    spec = SimpleNamespace(key="mhealth")
    splits = [SimpleNamespace(index=13)]
    args = launcher.build_parser("mhealth").parse_args([])
    with pytest.raises(ValueError, match="0/1 ready"):
        launcher.wait_for_vectors(tmp_path, spec=spec, splits=splits, args=args)


@pytest.mark.parametrize("dataset,unseen_count,sample_count", [("uschad", 4, 120), ("wisdm", 6, 180), ("dsads", 6, 180)])
def test_new_dataset_protocol_has_correct_unseen_count_and_test_size(dataset, unseen_count, sample_count):
    launcher = _launcher()
    _, splits, split_path, sample_path = launcher.read_protocol(dataset)
    assert len(splits) == 20
    assert {len(split.unseen_activities) for split in splits} == {unseen_count}
    assert launcher.UNSEEN_COUNTS[dataset] * launcher.EVAL_COUNTS[dataset] == sample_count
    assert f"unseen{unseen_count}" in split_path.name and f"unseen{unseen_count}" in sample_path.name


def _refinement_launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/refinement_xxapi_common.py"
    spec = importlib.util.spec_from_file_location("v4_refinement_common_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dataset,unseen_count", [("uschad", 4), ("wisdm", 6), ("dsads", 6)])
def test_generation_uses_latest_policy_and_dataset_specific_manifest(dataset, unseen_count):
    from zshar.datasets import load_dataset_spec
    launcher = _refinement_launcher()
    assert dataset in launcher.SUPPORTED_DATASETS
    args = launcher.build_parser(dataset).parse_args([])
    assert args.root == launcher.PROJECT / "configs/attrIbutes_v4" / dataset
    assert args.model == "gpt-6.1-sol" and args.temperature == 0.7
    assert args.min_attributes == 9 and args.max_attributes == 15
    assert launcher.split_manifest_path(load_dataset_spec(dataset)).name == (
        f"{dataset}_unseen{unseen_count}_random20_splits_seed42.json")


def test_dataset_extension_preserves_resume_but_not_changed_llm_protocol():
    launcher = _refinement_launcher()
    previous = {"launcher_sha256": "old", "model": "gpt-6.1-sol", "temperature": 0.7,
                "implementation_sha256": {"planner1.py": "same"}}
    current = {**previous, "launcher_sha256": "new-dataset-support"}
    assert launcher.configuration_matches(previous, current)
    assert not launcher.configuration_matches(previous, {**current, "temperature": 0.0})
    assert not launcher.configuration_matches(previous, {
        **current, "implementation_sha256": {"planner1.py": "changed"}})
