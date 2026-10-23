#!/usr/bin/env python3
"""Run the MHEALTH initial-A1 ZSL AutoGluon default-ensemble baseline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATASET = "mhealth"
DATASET_CONFIG = PROJECT / "configs/datasets/mhealth.json"
ATTRIBUTES = (
    PROJECT / "configs/attributes/activity_attribute_vectors_initial_mhealth_v1_5.json"
)
ONTOLOGY = PROJECT / "configs/attributes/initial_attributes.json"
SPLITS = (
    PROJECT
    / "split/zero_shot_split/mhealth_unseen4_random20_splits_seed42.json"
)
SAMPLES = (
    PROJECT
    / "split/zero_shot_split/mhealth_unseen4_random20_eval30_samples_seed42.json"
)
RUN_ROOT = PROJECT / "runtime/zshar_runs"
SEED = 42
SAMPLES_PER_ACTIVITY = 30
EXPECTED_SPLITS = 20
MANAGER_MANIFEST = "attribute_backend_manifest.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def selected_indices(raw: str | None, available: Sequence[int]) -> list[int]:
    if not raw:
        return list(available)
    selected = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"Unknown --split-indices: {unknown}")
    if not selected:
        raise ValueError("--split-indices selected no splits")
    return selected


def preflight() -> tuple[Any, list[int]]:
    from zshar.data import load_activity_vectors, load_dataset_table
    from zshar.datasets import load_dataset_spec
    from zshar.splits import (
        load_sample_manifest,
        load_split_manifest,
        sample_evaluation_rows,
    )

    for path in (DATASET_CONFIG, ATTRIBUTES, ONTOLOGY, SPLITS, SAMPLES):
        if not path.is_file():
            raise FileNotFoundError(path)

    spec = load_dataset_spec(DATASET, DATASET_CONFIG)
    if spec.attributes_path.resolve() != ATTRIBUTES.resolve():
        raise ValueError(f"MHEALTH profile must point to the initial A1 vectors: {ATTRIBUTES}")

    vectors, attributes = load_activity_vectors(ATTRIBUTES)
    expected_attributes = [
        str(item["name"]) for item in read_json(ONTOLOGY).get("attributes", [])
    ]
    if set(vectors) != set(spec.activity_names):
        raise ValueError("Initial A1 vector activities differ from the MHEALTH profile")
    if attributes != expected_attributes or len(attributes) != 9:
        raise ValueError("MHEALTH initial vectors must match the ordered 9-D ontology")
    observed_values = {
        str(value) for vector in vectors.values() for value in vector.values()
    }
    if not observed_values.issubset({"Low", "Medium", "High"}):
        raise ValueError(f"Unexpected attribute levels: {sorted(observed_values)}")

    split_payload = read_json(SPLITS)
    sample_payload = read_json(SAMPLES)
    if (
        split_payload.get("dataset") != DATASET
        or int(split_payload.get("seed", -1)) != SEED
        or int(split_payload.get("unseen_count", -1)) != 4
    ):
        raise ValueError(f"Invalid MHEALTH seed-42 split manifest: {SPLITS}")
    splits = load_split_manifest(SPLITS, spec)
    indices = [int(split.index) for split in splits]
    if len(indices) != EXPECTED_SPLITS or len(set(indices)) != EXPECTED_SPLITS:
        raise ValueError("Expected exactly 20 distinct MHEALTH seed-42 splits")
    if (
        sample_payload.get("dataset") != DATASET
        or int(sample_payload.get("seed", -1)) != SEED
        or int(sample_payload.get("samples_per_activity", -1))
        != SAMPLES_PER_ACTIVITY
        or set(map(int, sample_payload.get("samples", {}))) != set(indices)
    ):
        raise ValueError(f"Invalid MHEALTH seed-42 eval30 sample manifest: {SAMPLES}")

    sample_manifest = load_sample_manifest(SAMPLES)
    table = load_dataset_table(spec)
    for split in splits:
        by_activity = sample_payload["samples"][str(split.index)]
        if set(map(str, by_activity)) != set(split.unseen_activities):
            raise ValueError(
                f"MHEALTH split {split.index} test classes differ from unseen classes"
            )
        if len(by_activity) != 4 or any(
            len(ids) != SAMPLES_PER_ACTIVITY
            or len(set(map(int, ids))) != SAMPLES_PER_ACTIVITY
            for ids in by_activity.values()
        ):
            raise ValueError(f"MHEALTH split {split.index} has invalid fixed samples")
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="zsl",
            samples_per_activity=SAMPLES_PER_ACTIVITY,
            seed=SEED,
            manifest_ids=sample_manifest[split.index],
        )
        if (
            len(evaluation) != 120
            or not evaluation.activity_name.value_counts().eq(30).all()
        ):
            raise ValueError(
                f"MHEALTH split {split.index} does not resolve to 4 x 30 samples"
            )
    return spec, indices


def default_run_name(time_limit: int) -> str:
    return (
        "mhealth_zsl_autogluon_default_ensemble_initial_a1_v1_5_"
        f"independent_t{time_limit}_random20_eval30_seed42"
    )


def experiment_config(args: argparse.Namespace, selected: Sequence[int]) -> Any:
    from zshar.pipeline.runner import ExperimentConfig

    return ExperimentConfig(
        dataset=DATASET,
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
            "lightgbm_device": args.lightgbm_device,
            "gpu_model_types": ["NN_TORCH", "FASTAI", "CAT", "XGB"],
            "verbosity": args.verbosity,
        },
        seed=SEED,
        split_manifest=str(SPLITS),
        sample_manifest=str(SAMPLES),
        split_indices=set(selected),
        samples_per_activity=SAMPLES_PER_ACTIVITY,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        run_root=str(args.run_root.expanduser().resolve()),
        run_name=args.run_name or default_run_name(args.time_limit),
        reuse_models=args.resume,
        evaluate_only=False,
        dry_run=args.dry_run,
    )


def validate_existing_run(config: Any) -> Path:
    run_dir = Path(config.run_root) / DATASET / str(config.run_name)
    saved = run_dir / "config.json"
    if saved.is_file():
        previous = normalize(read_json(saved))
        current = normalize(asdict(config))
        ignored = {"reuse_models", "evaluate_only", "dry_run"}
        if {key: value for key, value in previous.items() if key not in ignored} != {
            key: value for key, value in current.items() if key not in ignored
        }:
            raise ValueError(f"Existing run uses a different protocol: {saved}")
    return run_dir


def preserve_partial_models(run_dir: Path, *, resume: bool, dry_run: bool) -> None:
    partial = [
        model_dir
        for model_dir in run_dir.glob("split_*/models")
        if not (model_dir / MANAGER_MANIFEST).is_file()
    ]
    if not partial:
        return
    if not resume:
        raise ValueError(
            "Incomplete model directories exist; rerun with --resume: "
            + ", ".join(map(str, partial))
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in partial:
        destination = path.with_name(f"models.incomplete_{stamp}")
        counter = 1
        while destination.exists():
            destination = path.with_name(f"models.incomplete_{stamp}_{counter}")
            counter += 1
        if dry_run:
            print(f"Dry run: would preserve incomplete model directory {path}")
        else:
            shutil.move(str(path), str(destination))
            print(f"Preserved incomplete model directory: {destination}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "train-test"), default="train-test"
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--lightgbm-device", choices=("cpu", "gpu", "cuda"), default="cuda")
    parser.add_argument("--verbosity", type=int, choices=range(5), default=2)
    parser.add_argument("--split-indices", help="Optional comma-separated subset")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_id < 0 or args.time_limit <= 0 or args.num_cpus <= 0:
        raise ValueError("GPU id must be nonnegative; time/CPU values must be positive")
    if args.mode == "check" and (args.resume or args.dry_run):
        raise ValueError("--resume/--dry-run do not apply to check mode")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    spec, available = preflight()
    selected = selected_indices(args.split_indices, available)
    config = experiment_config(args, selected)
    run_dir = validate_existing_run(config)
    print(
        "MHEALTH initial-A1 independent AutoGluon default ensemble\n"
        f"  source vectors: {ATTRIBUTES}\n"
        f"  source sha256: {sha256(ATTRIBUTES)}\n"
        f"  physical GPU: {args.gpu_id} (logical device 0)\n"
        f"  selected splits: {selected}\n"
        f"  split manifest: {SPLITS}\n"
        f"  sample manifest: {SAMPLES}\n"
        f"  time limit per attribute: {args.time_limit}s\n"
        "  bagging folds: 0; stacking levels: 0; weighted ensemble: enabled\n"
        f"  output: {run_dir}",
        flush=True,
    )
    if args.mode == "check":
        print("Check only: no classifier was trained or evaluated.", flush=True)
        return 0

    preserve_partial_models(run_dir, resume=args.resume, dry_run=args.dry_run)
    from zshar.pipeline.runner import run_experiment

    output = run_experiment(spec, config)
    print(f"Completed MHEALTH initial-A1 AutoGluon baseline: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
