import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/run_mhealth_uschad_wisdm_dsads_initial9_autogluon_default.py"
    spec = importlib.util.spec_from_file_location("dsads_initial9_rerun_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dsads_initial9_fixed_manifest_default_ensemble_and_separate_output(tmp_path):
    module = launcher()
    args = module.build_parser().parse_args([
        "--datasets", "dsads", "--run-root", str(tmp_path), "--run-name-suffix", "rerun_test"])
    job = next(job for job in module.JOBS if job.key == "dsads")
    spec, splits, indices = module.preflight(job)
    config = module.make_config(job, args, indices)
    assert len(splits) == 20 and config.seed == 42 and config.scenario == "zsl"
    assert spec.attributes_path.name == "activity_attribute_vectors_initial_dsads_v1_5.json"
    assert config.backend == "autogluon" and config.backend_config["model_suite"] == "default"
    assert config.backend_config["fit_weighted_ensemble"] is True
    assert config.backend_config["num_bag_folds"] == config.backend_config["num_stack_levels"] == 0
    assert config.samples_per_activity == 30
    assert config.split_manifest.startswith(str(module.PROJECT / "split/zero_shot_split"))
    assert config.sample_manifest.endswith("dsads_unseen6_random20_eval30_samples_seed42.json")
    assert config.run_name.endswith("_rerun_test") and not config.reuse_models
    assert config.prototype_metric == "ordinal_probability" and config.attribute_weighting == "validation_accuracy"


def test_resume_rejects_changed_initial_vectors_or_untracked_models(tmp_path):
    module = launcher()
    (tmp_path / "initial9_input_snapshot.json").write_text(json.dumps({"vectors": "original"}))
    module.validate_snapshot(tmp_path, {"vectors": "original"})
    with pytest.raises(ValueError, match="inputs changed"):
        module.validate_snapshot(tmp_path, {"vectors": "changed"})
    (tmp_path / "initial9_input_snapshot.json").unlink()
    (tmp_path / "split_820/models").mkdir(parents=True)
    with pytest.raises(ValueError, match="lack an input snapshot"):
        module.validate_snapshot(tmp_path, {"vectors": "original"})


def test_dataset_selection_does_not_run_other_datasets(tmp_path, monkeypatch):
    module = launcher()
    calls = []
    job = next(job for job in module.JOBS if job.key == "dsads")
    spec, splits, indices = module.preflight(job)

    def preflight(selected):
        calls.append(selected)
        return spec, splits, indices

    monkeypatch.setattr(module, "preflight", preflight)
    assert module.main(["check", "--datasets", "dsads", "--run-root", str(tmp_path),
                        "--run-name-suffix", "rerun_test"]) == 0
    assert [job.key for job in calls] == ["dsads"]
    assert not list(tmp_path.iterdir())


def test_resume_preserves_failed_split_models_and_keeps_completed_split(tmp_path):
    module = launcher()
    job = SimpleNamespace(key="dsads")
    config = SimpleNamespace(run_root=str(tmp_path), run_name="resume-test", dry_run=False, reuse_models=False)
    run = tmp_path / "dsads/resume-test"
    incomplete = run / "split_2849/models"
    incomplete.mkdir(parents=True)
    (incomplete / "partial.bin").write_bytes(b"preserve partial work")
    complete = run / "split_820/models"
    complete.mkdir(parents=True)
    (complete / module.MANAGER_MANIFEST).write_text("{}")
    with pytest.raises(ValueError, match="pass --resume"):
        module.validate_existing_run(job, config)
    config.reuse_models = True
    assert module.validate_existing_run(job, config) == run
    module.preserve_incomplete_models(run, resume=True)
    assert not incomplete.exists()
    backups = list((run / "split_2849").glob("models.incomplete_*"))
    assert len(backups) == 1
    assert (backups[0] / "partial.bin").read_bytes() == b"preserve partial work"
    assert (complete / module.MANAGER_MANIFEST).is_file()
