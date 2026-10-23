import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


PROJECT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(f"seed43_test_{name}", PROJECT / f"scripts/{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_seed43_protocol_and_classifier_defaults():
    launcher = module("attributes_v4_autogluon_common")
    args = launcher.build_parser("wisdm", seed=43).parse_args([])
    spec, splits, split_path, sample_path = launcher.read_protocol("wisdm", seed=43)
    assert args.gpu_id == 1 and args.seed == 43
    assert args.attribute_vectors_root == PROJECT / "configs/attrIbutes_v4/wisdm/seed43"
    args.run_name = launcher.default_run_name("wisdm", args.time_limit, seed=43)
    config = launcher.experiment_config("wisdm", args, [s.index for s in splits])
    assert config.seed == 43 and config.scenario == "zsl" and config.backend == "autogluon"
    assert config.split_manifest == str(split_path) and config.sample_manifest == str(sample_path)
    assert split_path.name == "wisdm_unseen6_random20_splits_seed43.json"
    assert sample_path.name == "wisdm_unseen6_random20_eval30_samples_seed43.json"
    assert config.samples_per_activity == 30 and len(splits) == 20
    assert config.backend_config["model_suite"] == "default"
    assert config.backend_config["fit_weighted_ensemble"] is True
    assert config.backend_config["num_bag_folds"] == config.backend_config["num_stack_levels"] == 0
    assert config.prototype_metric == "ordinal_probability" and config.attribute_weighting == "validation_accuracy"
    assert config.attribute_vectors_root.endswith("/wisdm/seed43")
    assert config.run_name.endswith("seed43")
    _, previous, _, _ = launcher.read_protocol("wisdm", seed=42)
    assert not ({s.index for s in previous} & {s.index for s in splits})
    launcher.verify_test_windows(spec, splits, sample_path)


def test_seed43_generation_routes_gap_and_comparison_to_same_seed(tmp_path, monkeypatch):
    launcher = module("refinement_xxapi_common")
    args = launcher.build_parser("wisdm", seed=43).parse_args([])
    assert args.root == PROJECT / "configs/attrIbutes_v4/wisdm/seed43"
    assert args.model == "gpt-6.1-sol" and args.temperature == 0.7 and not args.stream
    assert args.proxy_mode == "direct"
    calls = {}

    def gap(**kwargs):
        calls["gap"] = kwargs

    monkeypatch.setattr(launcher, "run_gap_analysis", gap)
    monkeypatch.setattr(launcher, "compare_refinement_gap", lambda **kw: calls.update(compare=kw))
    monkeypatch.setattr(launcher, "valid_plan", lambda *a, **kw: True)
    monkeypatch.setattr(launcher, "valid_vectors", lambda *a, **kw: True)
    monkeypatch.setattr(launcher.planner1, "resolve_split_from_gap_csv", lambda *a: {})
    monkeypatch.setattr(launcher, "configure_proxy_routing", lambda *a: None)
    assert launcher.main(["--stage", "compare", "--root", str(tmp_path)], dataset="wisdm", seed=43) == 0
    assert calls["gap"]["seed"] == calls["compare"]["seed"] == 43
    assert calls["gap"]["split_manifest"].name.endswith("seed43.json")
    assert calls["gap"]["output"].name.endswith("seed43.csv")
    assert calls["compare"]["split_manifest"].name == "selected_seed43_splits.json"
    saved = json.loads((tmp_path / "experiment_config.json").read_text())
    selected = json.loads(calls["compare"]["split_manifest"].read_text())
    assert saved["seed"] == selected["seed"] == 43
    assert len(selected["splits"]) == 20


def test_wrong_partition_cannot_reuse_plan_even_if_split_index_matches(tmp_path):
    launcher = module("attributes_v4_autogluon_common")
    source = PROJECT / "configs/attrIbutes_v4/wisdm/split_820/a2_attribute_plan.json"
    plan = json.loads(source.read_text())
    directory = tmp_path / "split_820"
    directory.mkdir()
    (directory / "a2_attribute_plan.json").write_text(json.dumps(plan))
    meta = plan["_meta"]
    seen, unseen = list(meta["seen"]), list(meta["unseen"])
    seen[0], unseen[0] = unseen[0], seen[0]
    split = SimpleNamespace(index=820, seen_activities=seen, unseen_activities=unseen)
    vectors, errors = launcher.scan_vectors(tmp_path, spec=SimpleNamespace(key="wisdm"), splits=[split])
    assert not vectors
    assert "seen/unseen classes disagree" in errors[820]
