import importlib.util
import json
from pathlib import Path

import pytest


def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/wait_for_uschad_v4.py"
    spec = importlib.util.spec_from_file_location("dsads_dependency_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_run(root):
    (root / "split_013").mkdir()
    (root / "split_013/metrics.json").write_text("{}")
    (root / "attribute_accuracy_summary.json").write_text(json.dumps({
        "complete": True, "completed_split_count": 1, "expected_split_count": 1,
        "missing_split_indices": [], "attribute_mean_accuracy": 0.7,
    }))
    (root / "aggregate_metrics.json").write_text(json.dumps({"split_count": 1}))


def test_still_waits_until_process_exits_even_with_complete_results(tmp_path, monkeypatch):
    module = launcher()
    complete_run(tmp_path)
    states = iter([[123], []])
    monkeypatch.setattr(module, "active_uschad_processes", lambda: next(states))
    sleeps = []
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    module.wait_for_completion(tmp_path, [13], 60)
    assert sleeps == [60]


def test_failed_predecessor_does_not_start_dsads(tmp_path, monkeypatch):
    module = launcher()
    monkeypatch.setattr(module, "active_uschad_processes", lambda: [])
    with pytest.raises(RuntimeError, match="DSADS was not started"):
        module.wait_for_completion(tmp_path, [13], 60)


def test_stale_complete_summary_without_all_metrics_is_rejected(tmp_path):
    module = launcher()
    complete_run(tmp_path)
    (tmp_path / "split_013/metrics.json").unlink()
    assert module.completion_status(tmp_path, [13]) == (False, 0)


def test_detects_only_uschad_gpu0_training(tmp_path):
    module = launcher()
    for pid, script, flags in [
        (10, "run_uschad_v4_autogluon_default_ensemble.py", ["train-test", "--gpu-id", "0"]),
        (11, "run_uschad_v4_autogluon_default_ensemble.py", ["check"]),
        (12, "run_uschad_v4_autogluon_default_ensemble.py", ["train-test", "--gpu-id", "1"]),
        (13, "run_wisdm_v4_autogluon_default_ensemble.py", ["train-test", "--gpu-id", "0"]),
    ]:
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes("\0".join(["python", script, *flags, ""]).encode())
    assert module.active_uschad_processes(tmp_path) == [10]
