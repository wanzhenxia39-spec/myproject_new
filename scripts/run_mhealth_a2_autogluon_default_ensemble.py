#!/usr/bin/env python3
"""Run the MHEALTH semantic-v3 A2 independent AutoGluon ensemble baseline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROTOCOL_SCRIPT = (
    PROJECT / "scripts/run_mhealth_a2_autogluon_soft_oof_catboost_forward.py"
)
PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "mhealth_a2_semantic_v3_protocol", PROTOCOL_SCRIPT
)
protocol = importlib.util.module_from_spec(PROTOCOL_SPEC)
assert PROTOCOL_SPEC.loader is not None
PROTOCOL_SPEC.loader.exec_module(protocol)

RUN_ROOT = PROJECT / "runtime/zshar_runs"
RUN_NAME = (
    "mhealth_zsl_autogluon_default_ensemble_a2_semantic_v3_"
    "independent_random20_eval30_seed42"
)


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(normalize(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def validate_existing_run(config: Any) -> Path:
    run_dir = Path(config.run_root) / protocol.DATASET / str(config.run_name)
    saved = run_dir / "config.json"
    if saved.is_file():
        previous = normalize(json.loads(saved.read_text(encoding="utf-8")))
        current = normalize(asdict(config))
        ignored = {"reuse_models", "evaluate_only", "dry_run"}
        if {key: value for key, value in previous.items() if key not in ignored} != {
            key: value for key, value in current.items() if key not in ignored
        }:
            raise ValueError(f"Existing run uses a different protocol: {saved}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "train-test"), default="train-test"
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--split-indices", help="Optional comma-separated subset")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    from zshar.pipeline.runner import ExperimentConfig, run_experiment

    args = build_parser().parse_args(argv)
    if args.gpu_id < 0 or args.time_limit <= 0 or args.num_cpus <= 0:
        raise ValueError("GPU id must be nonnegative and CPU/time values positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    split_payload = protocol.read_json(protocol.SPLITS)
    available = [int(item["split_index"]) for item in split_payload.get("splits", [])]
    if len(available) != 20 or len(set(available)) != 20:
        raise ValueError("Expected exactly 20 distinct MHEALTH seed-42 splits")
    selected = protocol.selected_indices(args, available)
    spec, _, _, verified = protocol.preflight(selected)
    if verified != available:
        raise ValueError("Split manifest changed during protocol validation")

    config = ExperimentConfig(
        dataset=protocol.DATASET,
        scenario="zsl",
        backend="autogluon",
        backend_config={
            "time_limit": args.time_limit,
            "presets": "medium_quality",
            "num_gpus": 1,
            "num_cpus": args.num_cpus,
            "num_bag_folds": 0,
            "num_stack_levels": 0,
            "fit_weighted_ensemble": True,
            "model_suite": "default",
            "lightgbm_device": "cuda",
            "gpu_model_types": ["NN_TORCH", "FASTAI", "CAT", "XGB"],
            "verbosity": 2,
        },
        seed=42,
        split_manifest=str(protocol.SPLITS),
        sample_manifest=str(protocol.SAMPLES),
        attribute_vectors_root=str(protocol.A2_ROOT),
        split_indices=set(selected),
        samples_per_activity=protocol.EVAL_SAMPLES_PER_ACTIVITY,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        run_root=str(args.run_root.expanduser().resolve()),
        run_name=args.run_name,
        reuse_models=args.resume,
        dry_run=args.dry_run,
    )
    run_dir = validate_existing_run(config)
    print(
        "MHEALTH semantic-v3 A2 independent AutoGluon default ensemble\n"
        f"  physical GPU: {args.gpu_id} (logical device 0)\n"
        f"  selected splits: {selected}\n"
        f"  split manifest: {protocol.SPLITS}\n"
        f"  sample manifest: {protocol.SAMPLES}\n"
        f"  A2 root: {protocol.A2_ROOT}\n"
        f"  time limit per attribute: {args.time_limit}s\n"
        "  AutoGluon bagging folds: 0 (default independent ensemble)\n"
        f"  run: {run_dir}",
        flush=True,
    )
    if args.mode == "check":
        print("Check only: no classifier was trained or evaluated.", flush=True)
        return 0

    protocol.base.common.preserve_partial_models(
        run_dir, resume=args.resume, dry_run=args.dry_run
    )
    output = run_experiment(spec, config)
    print(f"Completed MHEALTH AutoGluon default ensemble: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
