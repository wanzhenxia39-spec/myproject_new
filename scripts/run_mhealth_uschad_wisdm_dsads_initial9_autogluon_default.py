#!/usr/bin/env python3
"""Sequential initial-9D AutoGluon baselines: MHEALTH -> USC-HAD -> WISDM -> DSADS."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence


PROJECT = Path("/home/wanzhen/myproject_new")
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from zshar.workspace_paths import normalize_migrated_paths

SPLIT_ROOT = PROJECT / "split/zero_shot_split"
RUN_ROOT = Path("/home/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs")
ONTOLOGY = PROJECT / "configs/attributes/initial_attributes.json"
SEED = 42
EXPECTED_SPLITS = 20
MANAGER_MANIFEST = "attribute_backend_manifest.json"


@dataclass(frozen=True)
class DatasetJob:
    key: str
    display_name: str
    unseen_count: int
    samples_per_activity: int
    attributes: Path
    dataset_config: Path
    split_manifest: Path
    sample_manifest: Path


JOBS = (
    DatasetJob(
        key="mhealth",
        display_name="MHEALTH",
        unseen_count=4,
        samples_per_activity=30,
        attributes=PROJECT / "configs/attributes/activity_attribute_vectors_initial_mhealth_v1_5.json",
        dataset_config=PROJECT / "configs/datasets/mhealth.json",
        split_manifest=SPLIT_ROOT / "mhealth_unseen4_random20_splits_seed42.json",
        sample_manifest=SPLIT_ROOT / "mhealth_unseen4_random20_eval30_samples_seed42.json",
    ),
    DatasetJob(
        key="uschad",
        display_name="USC-HAD",
        unseen_count=4,
        samples_per_activity=30,
        attributes=PROJECT / "configs/attributes/activity_attribute_vectors_initial_uschad_v1_5.json",
        dataset_config=PROJECT / "configs/datasets/uschad.json",
        split_manifest=SPLIT_ROOT / "uschad_unseen4_random20_splits_seed42.json",
        sample_manifest=SPLIT_ROOT / "uschad_unseen4_random20_eval30_samples_seed42.json",
    ),
    DatasetJob(
        key="wisdm",
        display_name="WISDM",
        unseen_count=6,
        samples_per_activity=30,
        attributes=PROJECT / "configs/attributes/activity_attribute_vectors_initial_wisdm_v1_5.json",
        dataset_config=PROJECT / "configs/datasets/wisdm.json",
        split_manifest=SPLIT_ROOT / "wisdm_unseen6_random20_splits_seed42.json",
        sample_manifest=SPLIT_ROOT / "wisdm_unseen6_random20_eval30_samples_seed42.json",
    ),
    DatasetJob(
        key="dsads",
        display_name="DSADS",
        unseen_count=6,
        samples_per_activity=30,
        attributes=PROJECT / "configs/attributes/activity_attribute_vectors_initial_dsads_v1_5.json",
        dataset_config=PROJECT / "configs/datasets/dsads.json",
        split_manifest=SPLIT_ROOT / "dsads_unseen6_random20_splits_seed42.json",
        sample_manifest=SPLIT_ROOT / "dsads_unseen6_random20_eval30_samples_seed42.json",
    ),
)


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


def select_indices(raw: str | None, available: Sequence[int]) -> list[int]:
    if not raw:
        return list(available)
    selected = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"Unknown --split-indices for this dataset: {unknown}")
    if not selected:
        raise ValueError("--split-indices selected no splits")
    return selected


def preflight(job: DatasetJob) -> tuple[Any, list[Any], list[int]]:
    from zshar.data import load_activity_vectors, load_dataset_table
    from zshar.datasets import load_dataset_spec
    from zshar.splits import (
        load_sample_manifest,
        load_split_manifest,
        sample_evaluation_rows,
    )

    for path in (
        job.attributes,
        job.dataset_config,
        job.split_manifest,
        job.sample_manifest,
        ONTOLOGY,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    spec = load_dataset_spec(job.key, job.dataset_config)
    if spec.attributes_path.resolve() != job.attributes.resolve():
        raise ValueError(
            f"{job.display_name} profile does not point to its initial 9-D vectors"
        )
    vectors, attributes = load_activity_vectors(job.attributes)
    expected_attributes = [
        str(item["name"]) for item in read_json(ONTOLOGY).get("attributes", [])
    ]
    if set(vectors) != set(spec.activity_names):
        raise ValueError(f"{job.display_name} initial-vector activities mismatch")
    if attributes != expected_attributes or len(attributes) != 9:
        raise ValueError(f"{job.display_name} does not use the ordered 9-D ontology")
    values = {str(value) for vector in vectors.values() for value in vector.values()}
    if not values.issubset({"Low", "Medium", "High"}):
        raise ValueError(f"{job.display_name} has invalid attribute levels: {values}")

    split_payload = read_json(job.split_manifest)
    sample_payload = read_json(job.sample_manifest)
    if (
        split_payload.get("dataset") != job.key
        or int(split_payload.get("seed", -1)) != SEED
        or int(split_payload.get("unseen_count", -1)) != job.unseen_count
    ):
        raise ValueError(f"Invalid split manifest: {job.split_manifest}")
    splits = load_split_manifest(job.split_manifest, spec)
    indices = [int(split.index) for split in splits]
    if len(indices) != EXPECTED_SPLITS or len(set(indices)) != EXPECTED_SPLITS:
        raise ValueError(f"Expected 20 distinct {job.display_name} splits")
    if (
        sample_payload.get("dataset") != job.key
        or int(sample_payload.get("seed", -1)) != SEED
        or int(sample_payload.get("samples_per_activity", -1))
        != job.samples_per_activity
        or set(map(int, sample_payload.get("samples", {}))) != set(indices)
    ):
        raise ValueError(f"Invalid sample manifest: {job.sample_manifest}")

    sample_manifest = load_sample_manifest(job.sample_manifest)
    table = load_dataset_table(spec)
    expected_rows = job.unseen_count * job.samples_per_activity
    for split in splits:
        raw_by_activity = sample_payload["samples"][str(split.index)]
        if set(map(str, raw_by_activity)) != set(split.unseen_activities):
            raise ValueError(
                f"{job.display_name} split {split.index} test/unseen classes differ"
            )
        if len(raw_by_activity) != job.unseen_count or any(
            len(ids) != job.samples_per_activity
            or len(set(map(int, ids))) != job.samples_per_activity
            for ids in raw_by_activity.values()
        ):
            raise ValueError(
                f"{job.display_name} split {split.index} has invalid fixed samples"
            )
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="zsl",
            samples_per_activity=job.samples_per_activity,
            seed=SEED,
            manifest_ids=sample_manifest[split.index],
        )
        if (
            len(evaluation) != expected_rows
            or not evaluation.activity_name.value_counts()
            .eq(job.samples_per_activity)
            .all()
        ):
            raise ValueError(
                f"{job.display_name} split {split.index} does not resolve to "
                f"{job.unseen_count} x {job.samples_per_activity} samples"
            )
    return spec, splits, indices


def run_name(job: DatasetJob, time_limit: int) -> str:
    return (
        f"{job.key}_zsl_autogluon_default_ensemble_initial9_"
        f"independent_t{time_limit}_random20_eval{job.samples_per_activity}_seed42"
    )


def make_config(
    job: DatasetJob,
    args: argparse.Namespace,
    selected: Sequence[int],
) -> Any:
    from zshar.pipeline.runner import ExperimentConfig

    return ExperimentConfig(
        dataset=job.key,
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
        split_manifest=str(job.split_manifest),
        sample_manifest=str(job.sample_manifest),
        split_indices=set(selected),
        samples_per_activity=job.samples_per_activity,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        run_root=str(args.run_root.expanduser().resolve()),
        run_name=run_name(job, args.time_limit) + (f"_{args.run_name_suffix}" if args.run_name_suffix else ""),
        reuse_models=args.resume,
        evaluate_only=False,
        dry_run=args.dry_run,
    )


def validate_existing_run(job: DatasetJob, config: Any) -> Path:
    run_dir = Path(config.run_root) / job.key / str(config.run_name)
    saved_config = run_dir / "config.json"
    if saved_config.is_file():
        previous = normalize_migrated_paths(normalize(read_json(saved_config)))
        current = normalize_migrated_paths(normalize(asdict(config)))
        ignored = {"reuse_models", "evaluate_only", "dry_run"}
        for payload in (previous, current):
            if payload.get("split_indices") is not None:
                payload["split_indices"] = sorted(payload["split_indices"])
        if {key: value for key, value in previous.items() if key not in ignored} != {
            key: value for key, value in current.items() if key not in ignored
        }:
            raise ValueError(f"Existing run uses a different protocol: {saved_config}")
    partial = [
        model_dir
        for model_dir in run_dir.glob("split_*/models")
        if not (model_dir / MANAGER_MANIFEST).is_file()
    ]
    if partial and not config.dry_run and not config.reuse_models:
        raise ValueError(
            "Interrupted partial model directories found; pass --resume to preserve "
            "and retrain them: " + ", ".join(map(str, partial))
        )
    return run_dir


def preserve_incomplete_models(run_dir: Path, *, resume: bool) -> None:
    partial = [path for path in run_dir.glob("split_*/models")
               if not (path / MANAGER_MANIFEST).is_file()]
    if partial and not resume:
        raise ValueError("Pass --resume to preserve incomplete models before retraining")
    for path in partial:
        destination = path.with_name(f"models.incomplete_{time.time_ns()}")
        path.rename(destination)
        print(f"Preserved incomplete models: {destination}", flush=True)


def input_snapshot(job, spec):
    paths = (job.attributes, ONTOLOGY, job.dataset_config, job.split_manifest,
             job.sample_manifest, spec.features_path, spec.feature_names_path, spec.labels_path)
    return {str(path.resolve()): sha256(path) for path in paths}


def validate_snapshot(run_dir, snapshot):
    path = run_dir / "initial9_input_snapshot.json"
    if path.is_file() and normalize_migrated_paths(read_json(path)) != normalize_migrated_paths(snapshot):
        raise ValueError("Initial A1 vectors or experiment inputs changed; choose a new --run-name-suffix")
    if not path.exists() and any(run_dir.glob("split_*/models")):
        raise ValueError("Existing models lack an input snapshot; choose a new --run-name-suffix")


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
    parser.add_argument("--split-indices", help="Optional comma-separated subset for every dataset")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=[job.key for job in JOBS],
                        default=[job.key for job in JOBS], help="Run only these datasets, in this order")
    parser.add_argument("--split-root", type=Path, default=SPLIT_ROOT)
    parser.add_argument("--run-name-suffix", default="", help="Separate a fresh rerun from saved results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_id < 0 or args.time_limit <= 0 or args.num_cpus <= 0:
        raise ValueError("GPU id must be nonnegative; time/CPU values must be positive")
    if args.mode == "check" and (args.resume or args.dry_run):
        raise ValueError("--resume/--dry-run do not apply to check mode")
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("--datasets must not repeat a dataset")
    if args.run_name_suffix and (Path(args.run_name_suffix).name != args.run_name_suffix
                                 or args.run_name_suffix in (".", "..")):
        raise ValueError("--run-name-suffix must be a single path component")
    job_by_key = {job.key: job for job in JOBS}
    jobs = [replace(job_by_key[key],
                    split_manifest=args.split_root.resolve() / job_by_key[key].split_manifest.name,
                    sample_manifest=args.split_root.resolve() / job_by_key[key].sample_manifest.name)
            for key in args.datasets]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    print(
        "Initial-9D AutoGluon default-ensemble protocol\n"
        f"  sequence: {' -> '.join(job.display_name for job in jobs)}\n"
        f"  physical GPU: {args.gpu_id} (logical device 0)\n"
        "  independent attributes; bagging=0; stacking=0; weighted ensemble enabled\n"
        f"  time limit per attribute: {args.time_limit}s\n"
        f"  output root: {args.run_root.expanduser().resolve()}",
        flush=True,
    )

    from zshar.pipeline.runner import run_experiment

    # A single loop is intentional: a failure prevents every later dataset
    # from starting, so only one dataset can occupy physical GPU0 at a time.
    for stage, job in enumerate(jobs, 1):
        print(f"\n=== Dataset stage {stage}/{len(jobs)}: {job.display_name} ===", flush=True)
        spec, _, available = preflight(job)
        selected = select_indices(args.split_indices, available)
        config = make_config(job, args, selected)
        run_dir = validate_existing_run(job, config)
        snapshot = input_snapshot(job, spec)
        validate_snapshot(run_dir, snapshot)
        print(
            f"  vectors: {job.attributes}\n"
            f"  vector sha256: {sha256(job.attributes)}\n"
            f"  selected splits: {selected}\n"
            f"  split manifest: {job.split_manifest}\n"
            f"  sample manifest: {job.sample_manifest}\n"
            f"  output: {run_dir}",
            flush=True,
        )
        if args.mode == "check":
            print(f"[{job.key}] check passed; no training/evaluation.", flush=True)
            continue
        if args.dry_run:
            run_experiment(spec, config)
            continue
        from zshar.evaluation.attribute_summary import export_attribute_accuracy
        from zshar.data import load_activity_vectors
        _, attributes = load_activity_vectors(job.attributes)
        sources = {index: {"sha256": sha256(job.attributes), "attributes": attributes}
                   for index in selected}
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / ".training.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another process is training in this run directory")
            validate_snapshot(run_dir, snapshot)
            validate_existing_run(job, config)
            preserve_incomplete_models(run_dir, resume=args.resume)
            snapshot_path = run_dir / "initial9_input_snapshot.json"
            temporary = snapshot_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
            temporary.replace(snapshot_path)
            try:
                output = run_experiment(spec, config)
            except BaseException:
                try:
                    export_attribute_accuracy(run_dir, expected_split_indices=selected, vector_inputs=sources,
                                              expected_sample_count=job.unseen_count * job.samples_per_activity)
                except Exception as exc:
                    print(f"Partial attribute summary unavailable: {exc}", file=sys.stderr)
                raise
            summary = export_attribute_accuracy(
                run_dir, expected_split_indices=selected, vector_inputs=sources,
                expected_sample_count=job.unseen_count * job.samples_per_activity, require_complete=True)
            print(f"[{job.key}] mean attribute accuracy: {summary['attribute_mean_accuracy']:.6f}", flush=True)
        print(f"[{job.key}] completed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
