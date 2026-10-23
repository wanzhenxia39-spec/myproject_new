import fcntl
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/attributes_v4_autogluon_common.py"
    spec = importlib.util.spec_from_file_location("v4_gpu_wait_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_query_filters_by_physical_uuid_and_ignores_own_process(monkeypatch):
    m = launcher()
    results = iter([
        "GPU-target, 128\n",
        f"GPU-other, 77\nGPU-target, {os.getpid()}\nGPU-target, 88\n",
    ])
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=next(results))

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    state = m.query_gpu_state(1)
    assert "--id=1" in calls[0]
    assert state == {"gpu_uuid": "GPU-target", "memory_used_mib": 128.0, "other_compute_pids": [88]}


def test_invalid_gpu_query_cannot_mean_idle(monkeypatch):
    m = launcher()
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="GPU-target, N/A\n"))
    with pytest.raises(ValueError):
        m.query_gpu_state(0)


def test_gpu_wait_requires_consecutive_idle_checks_and_holds_lock(tmp_path, monkeypatch):
    m = launcher()
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    samples = iter([
        {"gpu_uuid": "GPU-target", "memory_used_mib": 10, "other_compute_pids": [88]},
        {"gpu_uuid": "GPU-target", "memory_used_mib": 2000, "other_compute_pids": []},
        {"gpu_uuid": "GPU-target", "memory_used_mib": 10, "other_compute_pids": []},
        {"gpu_uuid": "GPU-target", "memory_used_mib": 10, "other_compute_pids": [88]},
        {"gpu_uuid": "GPU-target", "memory_used_mib": 10, "other_compute_pids": []},
        {"gpu_uuid": "GPU-target", "memory_used_mib": 10, "other_compute_pids": []},
    ])
    sleeps = []
    monkeypatch.setattr(m, "query_gpu_state", lambda gpu: next(samples))
    monkeypatch.setattr(m.time, "sleep", sleeps.append)
    args = SimpleNamespace(gpu_id=0, gpu_poll_seconds=60, gpu_max_used_mib=1024, gpu_idle_checks=2)
    with m.wait_for_gpu_slot(args, dataset="uschad", run_dir=tmp_path):
        assert sleeps == [60] * 5
        state = json.loads((tmp_path / "gpu_wait_status.json").read_text())
        assert state["state"] == "training"
        with (tmp_path / "runtime/locks/gpu0_v4_autogluon.lock").open("a+") as second:
            with pytest.raises(BlockingIOError):
                fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (tmp_path / "runtime/locks/gpu0_v4_autogluon.lock").open("a+") as second:
        fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(second, fcntl.LOCK_UN)


def test_gpu_query_failure_is_retried_without_starting_early(tmp_path, monkeypatch):
    m = launcher()
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    calls, sleeps = [], []

    def query(gpu):
        calls.append(gpu)
        if len(calls) == 1:
            raise ValueError("query unavailable")
        return {"gpu_uuid": "GPU-target", "memory_used_mib": 0, "other_compute_pids": []}

    monkeypatch.setattr(m, "query_gpu_state", query)
    monkeypatch.setattr(m.time, "sleep", sleeps.append)
    args = SimpleNamespace(gpu_id=1, gpu_poll_seconds=60, gpu_max_used_mib=1024, gpu_idle_checks=2)
    with m.wait_for_gpu_slot(args, dataset="wisdm", run_dir=tmp_path):
        assert calls == [1, 1, 1]
        assert sleeps == [60, 60]
