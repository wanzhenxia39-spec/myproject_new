#!/usr/bin/env python3
"""Wait for the GPU-0 USC-HAD v4 run to finish successfully before DSADS."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

PROJECT = Path(__file__).resolve().parents[1]
RUN_NAME = "uschad_zsl_autogluon_default_ensemble_a2_v4_independent_t300_random20_eval30_seed42"


def active_uschad_processes(proc_root=Path("/proc")):
    active = []
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = (process / "cmdline").read_bytes().decode().strip("\0").split("\0")
        except (OSError, UnicodeError):
            continue
        if not any(Path(arg).name == "run_uschad_v4_autogluon_default_ensemble.py" for arg in argv[1:3]):
            continue
        if "check" in argv or "summarize" in argv:
            continue
        gpu = argv[argv.index("--gpu-id") + 1] if "--gpu-id" in argv else "0"
        if gpu == "0":
            active.append(int(process.name))
    return active


def completion_status(run_dir, indices):
    completed = sum((run_dir / f"split_{index:03d}/metrics.json").is_file() for index in indices)
    try:
        summary = json.loads((run_dir / "attribute_accuracy_summary.json").read_text())
        aggregate = json.loads((run_dir / "aggregate_metrics.json").read_text())
    except (OSError, ValueError):
        return False, completed
    success = (
        completed == len(indices)
        and summary.get("complete") is True
        and summary.get("completed_split_count") == len(indices)
        and summary.get("expected_split_count") == len(indices)
        and summary.get("missing_split_indices") == []
        and summary.get("attribute_mean_accuracy") is not None
        and aggregate.get("split_count") == len(indices)
    )
    return success, completed


def wait_for_completion(run_dir, indices, poll_seconds):
    while True:
        active = active_uschad_processes()
        success, completed = completion_status(run_dir, indices)
        print(f"USC-HAD: {completed}/{len(indices)} splits; GPU-0 process PIDs={active}; "
              f"complete attribute summary={success}", flush=True)
        if not active:
            if success:
                print("USC-HAD completed successfully; DSADS may now start on GPU 0.", flush=True)
                return
            raise RuntimeError(
                "USC-HAD is not running and its 20-split results are incomplete. "
                "DSADS was not started. Resume USC-HAD, then rerun this command.")
        time.sleep(poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    manifest = json.loads((PROJECT / "split/zero_shot_split/uschad_unseen4_random20_splits_seed42.json").read_text())
    indices = [int(split["split_index"]) for split in manifest["splits"]]
    if manifest.get("seed") != 42 or len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError("Expected 20 unique USC-HAD seed-42 splits")
    wait_for_completion(PROJECT / "runtime/zshar_runs/uschad" / RUN_NAME, indices, args.poll_seconds)


if __name__ == "__main__":
    main()
