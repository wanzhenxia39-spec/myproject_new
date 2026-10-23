"""Protocol, disk-placement, and resume regressions for the v4 backend queue."""
import csv
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest


def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/run_v4_nonchain_backend_benchmark.py"
    spec = importlib.util.spec_from_file_location("v4_nonchain_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_registered_eligible_backends_are_included():
    from zshar.models import CLASSIFIER_BACKENDS
    excluded = {
        "autogluon", "autogluon_multilabel", "autogluon_soft_oof",
        "catboost_classifier_chain", "catboost_soft_oof", "generic_classifier_chain",
        "logistic_regression_chain",
    }
    assert set(launcher().BACKENDS) == set(CLASSIFIER_BACKENDS) - excluded


def test_output_guard_rejects_system_disk_and_symlink_escape(tmp_path):
    module = launcher()
    assert module.data_path(tmp_path) == tmp_path.resolve()
    with pytest.raises(ValueError, match="/data/wanzhen"):
        module.data_path(Path("/home/wanzhen/forbidden-test-output"))
    link = tmp_path / "system_link"
    link.symlink_to("/home/wanzhen", target_is_directory=True)
    with pytest.raises(ValueError, match="/data/wanzhen"):
        module.data_path(link / "forbidden-test-output")


@pytest.mark.parametrize("dataset,gpu", [("pamap2", 0), ("mhealth", 1)])
@pytest.mark.parametrize("mode", ["train-test", "evaluate"])
def test_cli_command_preserves_full_training_and_resume_protocol(dataset, gpu, mode, monkeypatch):
    from zshar import cli
    module = launcher()
    args = module.build_parser().parse_args([dataset, mode, "--resume"])
    args.gpu_id = gpu
    args.output_root = module.DEFAULT_OUTPUT.resolve()
    captured = []
    monkeypatch.setattr(cli, "run_experiment", lambda spec, config: captured.append(config))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(gpu))
    for backend in module.BACKENDS:
        expected = module.experiment_config(args, backend)
        invocation = module.command(args, expected)
        parsed = cli.build_parser().parse_args(invocation[4:])
        parsed.handler(parsed)
        assert module.v4.normalize(asdict(captured[-1])) == module.v4.normalize(asdict(expected))
        assert expected.attribute_vectors_root.endswith(f"configs/attrIbutes_v4/{dataset}")
        assert expected.scenario == "zsl" and expected.seed == 42
        if backend in module.NEURAL:
            assert expected.backend_config["device"] == "cuda:0"
    # Neither seed-43 nor generalized/cross-subject manifests can enter the queue.
    _, splits, split_path, sample_path = module.v4.read_protocol(dataset)
    assert len(splits) == 20
    assert split_path.parent.name == sample_path.parent.name == "zero_shot_split"
    assert split_path.name.endswith("seed42.json") and sample_path.name.endswith("seed42.json")


def test_partial_stale_or_duplicate_results_cannot_skip_training(tmp_path):
    module = launcher()
    indices = [13, 48]
    snapshot = {"dataset": "pamap2", "vectors": {
        index: {"sha256": "current", "attributes": ["cadence"]} for index in indices}}
    rows = []
    for index in indices:
        split_dir = tmp_path / f"split_{index:03d}"
        (split_dir / "models").mkdir(parents=True)
        (split_dir / "models/attribute_backend_manifest.json").write_text(json.dumps({"classifier_backend": "catboost"}))
        (split_dir / "metrics.json").write_text(json.dumps({
            "sample_count": 200, "attribute_vectors": {"sha256": "current"},
            "per_attribute": {"cadence": {"accuracy": 0.5}},
        }))
        (split_dir / "details.csv").write_text("placeholder")
        rows.append({
            "split_index": index, "dataset": "pamap2", "backend": "catboost", "scenario": "zsl",
            "eval_samples": 200, "attribute_vectors_sha256": "current",
            **{name: 0.5 for name in module.METRICS},
        })

    def save(items):
        with (tmp_path / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(items)

    save(rows)
    assert module.complete_rows(tmp_path, "catboost", indices, snapshot)
    save(rows[:1])
    assert module.complete_rows(tmp_path, "catboost", indices, snapshot) is None
    save([rows[0], rows[0]])
    assert module.complete_rows(tmp_path, "catboost", indices, snapshot) is None
    for changed in ({"eval_samples": 120}, {"attribute_vectors_sha256": "old"}, {"activity_macro_f1": "nan"}):
        save([rows[0], {**rows[1], **changed}])
        assert module.complete_rows(tmp_path, "catboost", indices, snapshot) is None
    save(rows)
    (tmp_path / "split_048/metrics.json").write_text("{}")
    assert module.complete_rows(tmp_path, "catboost", indices, snapshot) is None


def test_resume_rejects_replaced_v4_vectors(tmp_path):
    module = launcher()
    args = module.build_parser().parse_args(["mhealth", "--resume"])
    config = module.experiment_config(args, "mlp_multitask")
    (tmp_path / "config.json").write_text(json.dumps(module.v4.normalize(asdict(config))))
    snapshot = {"vectors": {13: {"sha256": "original"}}}
    (tmp_path / "v4_input_manifest.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="inputs changed"):
        module.v4.validate_existing_run(tmp_path, config, {"vectors": {13: {"sha256": "replaced"}}})
