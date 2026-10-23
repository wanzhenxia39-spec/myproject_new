#!/usr/bin/env python3
"""Run the MHEALTH A1 benchmark on a selected physical GPU.

The protocol is fixed to the 20 seed-42 strict-ZSL activity splits, their
4-unseen-class x 30-sample evaluation manifest, and the latest nine-dimensional
MHEALTH A1 vectors.  Backends run sequentially and are compared with the same
ordinal-probability prototype matcher used by the existing backend benchmark.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT / "scripts"
SRC = PROJECT / "src"
for search_path in (str(SCRIPT_DIR), str(SRC)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

import run_mhealth_a1_hard_chain_orders as a1_protocol


SPLITS = PROJECT / "split/zero_shot_split/mhealth_unseen4_random20_splits_seed42.json"
SAMPLES = PROJECT / "split/zero_shot_split/mhealth_unseen4_random20_eval30_samples_seed42.json"
A1 = PROJECT / "configs/attributes/activity_attribute_vectors_initial_mhealth_v1_5.json"
A1_BINDINGS = PROJECT / "configs/attributes/initial_attributes_group.json"
DATASET_CONFIG = PROJECT / "configs/datasets/mhealth.json"
PROTOCOL_ROOT = PROJECT / "runtime/mhealth_a1_backend_benchmark_seed42"
A1_SPLIT_ROOT = PROTOCOL_ROOT / "attribute_vectors"
RUN_ROOT = PROJECT / "runtime/zshar_runs"
COMPARISON_ROOT = RUN_ROOT / "mhealth/mhealth_a1_v1_5_backend_benchmark_seed42"
MANAGER_MANIFEST = "attribute_backend_manifest.json"
METRICS = (
    "activity_top1_accuracy",
    "activity_top3_recall",
    "activity_macro_f1",
)
BACKENDS = (
    "lightgbm",
    "xgboost",
    "catboost",
    "logistic_regression",
    "linear_svm",
    "extra_trees",
    "random_forest",
    "neuralnet_fastai",
    "neuralnet_torch",
    "ordinal_logistic",
    "coral_mlp",
    "mlp_independent",
    "mlp_multitask",
)
AVAILABLE_BACKENDS = (*BACKENDS, "autogluon")
REQUIRED_MODULES = {
    "autogluon": "autogluon.tabular",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "neuralnet_fastai": "fastai",
    "neuralnet_torch": "torch",
    "ordinal_logistic": "torch",
    "coral_mlp": "torch",
    "mlp_independent": "torch",
    "mlp_multitask": "torch",
}


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def backend_config(
    backend: str, num_cpus: int, *, autogluon_time_limit: int = 600,
    autogluon_bag_folds: int = 0,
) -> dict[str, Any]:
    if backend == "autogluon":
        return {
            "time_limit": autogluon_time_limit,
            "presets": "medium_quality",
            "num_gpus": 1,
            "num_cpus": num_cpus,
            "num_bag_folds": autogluon_bag_folds,
            "num_stack_levels": 0,
            "fit_weighted_ensemble": True,
            "model_suite": "default",
            "lightgbm_device": "cuda",
            "gpu_model_types": ["NN_TORCH", "FASTAI", "CAT", "XGB"],
            "verbosity": 2,
        }
    config: dict[str, Any] = {"random_state": 42}
    if backend == "lightgbm":
        config.update({"n_jobs": num_cpus, "device": "cuda"})
    elif backend == "xgboost":
        config.update({"n_jobs": num_cpus, "tree_method": "hist", "device": "cuda"})
    elif backend == "catboost":
        # CUDA_VISIBLE_DEVICES maps the selected physical GPU to logical device 0.
        config.update({"n_jobs": num_cpus, "task_type": "GPU", "devices": "0"})
    elif backend in {"extra_trees", "random_forest"}:
        config["n_jobs"] = num_cpus
    elif backend == "linear_svm":
        # ordinal_probability requires calibrated class probabilities.
        config["calibrate_proba"] = True
    elif backend in {
        "neuralnet_fastai",
        "neuralnet_torch",
        "ordinal_logistic",
        "coral_mlp",
        "mlp_independent",
        "mlp_multitask",
    }:
        config["device"] = "cuda:0"
    return config


def configured_backend(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    return backend_config(
        backend, args.num_cpus,
        autogluon_time_limit=getattr(args, "autogluon_time_limit", 600),
        autogluon_bag_folds=getattr(args, "autogluon_bag_folds", 0),
    )


def run_name(backend: str, feature_subsets: bool = False) -> str:
    suffix = "_bound_feature_subsets" if feature_subsets else ""
    return f"mhealth_zsl_{backend}_a1_v1_5{suffix}_random20_eval30_seed42"


def parse_backends(value: str) -> list[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise argparse.ArgumentTypeError("At least one backend is required")
    unknown = sorted(set(requested).difference(AVAILABLE_BACKENDS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown backends {unknown}; choices are {list(AVAILABLE_BACKENDS)}"
        )
    if len(requested) != len(set(requested)):
        raise argparse.ArgumentTypeError("Backend list contains duplicates")
    return requested


def process_identity(pid: int | None) -> str | None:
    if pid is None:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def wait_for_process(pid: int | None, identity: str | None) -> None:
    if pid is None or identity is None:
        return
    print(f"Waiting for PID {pid} to exit before using the selected GPU.", flush=True)
    while process_identity(pid) == identity:
        time.sleep(30)
    print("Previous GPU queue exited; starting the backend benchmark.", flush=True)


def validate_dependencies(backends: Sequence[str]) -> None:
    import importlib.util

    missing = sorted(
        {module for backend, module in REQUIRED_MODULES.items()
         if backend in backends and importlib.util.find_spec(module) is None}
    )
    if missing:
        raise RuntimeError(
            "Missing backend dependencies in the active Python environment: "
            + ", ".join(missing)
        )


def configure_a1_protocol(protocol_root: Path) -> None:
    global PROTOCOL_ROOT, A1_SPLIT_ROOT
    PROTOCOL_ROOT = protocol_root
    A1_SPLIT_ROOT = protocol_root / "attribute_vectors"
    a1_protocol.PROTOCOL_ROOT = protocol_root
    a1_protocol.A1_SPLIT_ROOT = A1_SPLIT_ROOT


def preflight(
    backends: Sequence[str],
    protocol_root: Path,
    protocol_args: argparse.Namespace,
) -> tuple[Any, list[Any]]:
    configure_a1_protocol(protocol_root)
    validate_dependencies(backends)
    spec, splits, table, attributes = a1_protocol.preflight()
    if a1_protocol.SPLITS.resolve() != SPLITS.resolve():
        raise AssertionError("Imported A1 protocol uses a different split manifest")
    if a1_protocol.SAMPLES.resolve() != SAMPLES.resolve():
        raise AssertionError("Imported A1 protocol uses a different sample manifest")
    if a1_protocol.A1.resolve() != A1.resolve():
        raise AssertionError("Imported A1 protocol uses a different A1 vector file")
    if list(attributes) != list(table.attribute_columns) or len(attributes) != 9:
        raise ValueError("Expected the latest MHEALTH A1 schema to contain exactly 9 attributes")
    if protocol_args.feature_subsets:
        from zshar.features.groups import resolve_attribute_features

        mapping = resolve_attribute_features(
            table.feature_columns, dataset="mhealth", ontology=read_json(A1_BINDINGS),
            include_auxiliary=not protocol_args.primary_groups_only,
        )
        if set(mapping["attributes"]) != set(attributes):
            raise ValueError("Feature bindings must cover exactly the nine A1 attributes")
        print("Bound input dimensions: " + json.dumps({
            name: entry["dimension"] for name, entry in mapping["attributes"].items()
        }), flush=True)
    a1_protocol.materialize_a1_split_vectors(splits)
    payload = {
        "schema_version": 1,
        "dataset": "mhealth",
        "scenario": "zsl",
        "attribute_set": "initial_A1_v1_5_9d",
        "attribute_vectors": str(A1.resolve()),
        "attribute_vectors_sha256": a1_protocol.file_hash(A1),
        "split_manifest": str(SPLITS.resolve()),
        "sample_manifest": str(SAMPLES.resolve()),
        "seed": 42,
        "split_count": 20,
        "unseen_classes_per_split": 4,
        "evaluation_samples_per_unseen_class": 30,
        "prototype_metric": "ordinal_probability",
        "attribute_weighting": "validation_accuracy",
        "feature_subset_mode": bool(protocol_args.feature_subsets),
        "attribute_feature_bindings": (
            str(A1_BINDINGS.resolve()) if protocol_args.feature_subsets else None
        ),
        "include_auxiliary_groups": not protocol_args.primary_groups_only,
        "backends": list(backends),
        "backend_configs": {name: configured_backend(protocol_args, name) for name in backends},
    }
    write_json(protocol_root / "backend_benchmark_protocol.json", payload)
    del table
    return spec, splits


def expected_config(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    expected = {
        "dataset": "mhealth",
        "scenario": "zsl",
        "backend": backend,
        "backend_config": configured_backend(args, backend),
        "seed": 42,
        "split_manifest": str(SPLITS),
        "sample_manifest": str(SAMPLES),
        "attribute_vectors_root": str(A1_SPLIT_ROOT),
        "samples_per_activity": 30,
        "prototype_metric": "ordinal_probability",
        "attribute_weighting": "validation_accuracy",
    }
    if args.feature_subsets:
        expected.update(
            {
                "attribute_feature_bindings": str(A1_BINDINGS),
                "attribute_feature_include_auxiliary": not args.primary_groups_only,
            }
        )
    return expected


def validate_existing_config(args: argparse.Namespace, backend: str) -> None:
    path = (
        args.run_root
        / "mhealth"
        / run_name(backend, args.feature_subsets)
        / "config.json"
    )
    if not path.is_file():
        return
    actual = read_json(path)
    expected = expected_config(args, backend)
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Existing run uses a different protocol: {path}\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )


def runner_command(args: argparse.Namespace, backend: str, mode: str) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "zshar.cli",
        "run",
        "--dataset",
        "mhealth",
        "--dataset-config",
        str(DATASET_CONFIG),
        "--scenario",
        "zsl",
        "--backend",
        backend,
        "--backend-config",
        json.dumps(configured_backend(args, backend), separators=(",", ":")),
        "--seed",
        "42",
        "--split-manifest",
        str(SPLITS),
        "--sample-manifest",
        str(SAMPLES),
        "--attribute-vectors-root",
        str(A1_SPLIT_ROOT),
        "--samples-per-activity",
        "30",
        "--prototype-metric",
        "ordinal_probability",
        "--attribute-weighting",
        "validation_accuracy",
        "--gpu-id",
        str(args.gpu_id),
        "--run-root",
        str(args.run_root),
        "--run-name",
        run_name(backend, args.feature_subsets),
    ]
    if args.feature_subsets:
        command += ["--attribute-feature-bindings", str(A1_BINDINGS)]
        if args.primary_groups_only:
            command.append("--attribute-feature-primary-only")
    if mode == "evaluate":
        command += ["--evaluate-only", "--reuse-models"]
    elif args.resume:
        command.append("--reuse-models")
    return command


def backend_complete(
    args: argparse.Namespace, backend: str, expected_splits: set[int]
) -> bool:
    run_dir = args.run_root / "mhealth" / run_name(backend, args.feature_subsets)
    summary = run_dir / "summary.csv"
    if not summary.is_file():
        return False
    try:
        frame = pd.read_csv(summary)
        indices = set(pd.to_numeric(frame["split_index"], errors="raise").astype(int))
    except (KeyError, ValueError, pd.errors.EmptyDataError):
        return False
    if len(frame) != len(expected_splits) or indices != expected_splits:
        return False
    if set(frame["backend"].astype(str)) != {backend}:
        return False
    return all(
        (run_dir / f"split_{index:03d}" / "models" / MANAGER_MANIFEST).is_file()
        for index in expected_splits
    )


def archive_partial_models(args: argparse.Namespace, backend: str) -> None:
    run_dir = args.run_root / "mhealth" / run_name(backend, args.feature_subsets)
    existing = list(run_dir.glob("split_*/models"))
    if existing and not args.resume:
        raise FileExistsError(f"Existing models found under {run_dir}; use --resume")
    stamp = timestamp()
    for model_dir in existing:
        if (model_dir / MANAGER_MANIFEST).is_file():
            continue
        destination = model_dir.with_name(f"models_incomplete_{stamp}")
        suffix = 1
        while destination.exists():
            destination = model_dir.with_name(f"models_incomplete_{stamp}_{suffix}")
            suffix += 1
        shutil.move(str(model_dir), str(destination))
        print(f"Preserved incomplete {backend} model directory: {destination}", flush=True)


def execute(command: Sequence[str], gpu_id: int) -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(SRC)
        + (os.pathsep + os.environ["PYTHONPATH"] if os.getenv("PYTHONPATH") else ""),
    }
    print("$ " + shlex.join(command), flush=True)
    subprocess.run(list(command), cwd=PROJECT, env=environment, check=True)


def summarize(
    args: argparse.Namespace,
    splits: Sequence[Any],
    backends: Sequence[str],
    *,
    require_all: bool,
) -> Path:
    expected = {int(split.index) for split in splits}
    rows: list[dict[str, Any]] = []
    split_frames: list[pd.DataFrame] = []
    incomplete: list[str] = []
    for backend in backends:
        path = (
            args.run_root
            / "mhealth"
            / run_name(backend, args.feature_subsets)
            / "summary.csv"
        )
        if not backend_complete(args, backend, expected):
            incomplete.append(backend)
            continue
        frame = pd.read_csv(path).sort_values("split_index")
        split_frames.append(frame.assign(benchmark_backend=backend))
        row: dict[str, Any] = {"backend": backend, "split_count": 20}
        for metric in METRICS:
            values = pd.to_numeric(frame[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    if require_all and incomplete:
        raise RuntimeError(f"Incomplete backends: {incomplete}")
    if not rows:
        raise RuntimeError("No backend has a complete 20-split result")

    output = args.comparison_root
    output.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "comparison.csv", index=False)
    pd.concat(split_frames, ignore_index=True).to_csv(
        output / "all_split_metrics.csv", index=False
    )
    write_json(
        output / "comparison.json",
        {
            "dataset": "mhealth",
            "scenario": "zsl",
            "attribute_set": "initial_A1_v1_5_9d",
            "seed": 42,
            "split_count_per_backend": 20,
            "samples_per_unseen_activity": 30,
            "prototype_metric": "ordinal_probability",
            "attribute_weighting": "validation_accuracy",
            "feature_subset_mode": bool(args.feature_subsets),
            "attribute_feature_bindings": (
                str(A1_BINDINGS.resolve()) if args.feature_subsets else None
            ),
            "include_auxiliary_groups": not args.primary_groups_only,
            "completed_backends": [row["backend"] for row in rows],
            "incomplete_backends": incomplete,
            "metrics": rows,
        },
    )
    display = comparison.copy()
    for column in display.columns:
        if column.endswith("_mean") or column.endswith("_std"):
            display[column] = display[column].map(lambda value: f"{100.0 * value:.2f}%")
    print("\n" + display.to_string(index=False), flush=True)
    print(f"Comparison saved to: {output}", flush=True)
    if incomplete:
        print(f"Incomplete backends: {', '.join(incomplete)}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("check", "train-test", "evaluate", "report"),
        help="Validate, train/test, reevaluate saved models, or aggregate results.",
    )
    parser.add_argument(
        "--backends",
        type=parse_backends,
        default=list(BACKENDS),
        help="Comma-separated subset, including autogluon; default remains the 13 non-AutoGluon backends.",
    )
    parser.add_argument("--autogluon-time-limit", type=int, default=600,
                        help="AutoGluon training budget in seconds per attribute")
    parser.add_argument("--autogluon-bag-folds", type=int, default=0,
                        help="AutoGluon bagging folds: 0 (default) or at least 2")
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument(
        "--feature-subsets",
        action="store_true",
        help="use only each attribute's bound G1-G14 feature columns",
    )
    parser.add_argument(
        "--primary-groups-only",
        action="store_true",
        help="with --feature-subsets, exclude auxiliary groups",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first backend error instead of continuing with later backends.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.run_root = args.run_root.expanduser().resolve()
    if args.primary_groups_only and not args.feature_subsets:
        raise ValueError("--primary-groups-only requires --feature-subsets")
    if args.gpu_id < 0:
        raise ValueError("--gpu-id must be nonnegative")
    if args.autogluon_time_limit <= 0:
        raise ValueError("--autogluon-time-limit must be positive")
    if args.autogluon_bag_folds != 0 and args.autogluon_bag_folds < 2:
        raise ValueError("--autogluon-bag-folds must be 0 or at least 2")
    default_protocol = (
        PROJECT / "runtime/mhealth_a1_backend_feature_subsets_seed42"
        if args.feature_subsets
        else PROTOCOL_ROOT
    )
    default_comparison = (
        RUN_ROOT
        / "mhealth/mhealth_a1_v1_5_bound_feature_subsets_backend_benchmark_seed42"
        if args.feature_subsets
        else COMPARISON_ROOT
    )
    if args.backends == ["autogluon"]:
        feature_suffix = "_bound_feature_subsets" if args.feature_subsets else ""
        default_protocol = PROJECT / f"runtime/mhealth_a1_autogluon{feature_suffix}_seed42"
        default_comparison = RUN_ROOT / f"mhealth/mhealth_a1_v1_5_autogluon{feature_suffix}_seed42"
    args.protocol_root = (args.protocol_root or default_protocol).expanduser().resolve()
    args.comparison_root = (args.comparison_root or default_comparison).expanduser().resolve()
    if args.num_cpus <= 0:
        raise ValueError("--num-cpus must be positive")
    if args.wait_pid is not None and (args.wait_pid <= 0 or args.wait_pid == os.getpid()):
        raise ValueError("--wait-pid must identify another positive process")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    identity = process_identity(args.wait_pid)
    _, splits = preflight(args.backends, args.protocol_root, args)
    if args.mode == "check":
        print("Check only: no classifier was trained or evaluated.", flush=True)
        return 0
    if args.mode == "report":
        summarize(args, splits, args.backends, require_all=True)
        return 0

    lock_path = args.protocol_root / "backend_benchmark.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another benchmark owns {lock_path}") from exc
        wait_for_process(args.wait_pid, identity)
        expected_splits = {int(split.index) for split in splits}
        failures: dict[str, str] = {}
        for backend in args.backends:
            validate_existing_config(args, backend)
            if args.mode == "train-test" and args.resume and backend_complete(
                args, backend, expected_splits
            ):
                print(f"Skipping completed backend: {backend}", flush=True)
                continue
            if args.mode == "train-test":
                archive_partial_models(args, backend)
            try:
                execute(runner_command(args, backend, args.mode), args.gpu_id)
            except subprocess.CalledProcessError as exc:
                failures[backend] = f"exit code {exc.returncode}"
                print(f"Backend failed: {backend} ({failures[backend]})", flush=True)
                if args.fail_fast:
                    break

    if failures:
        try:
            summarize(args, splits, args.backends, require_all=False)
        except RuntimeError as exc:
            print(f"No final comparison yet: {exc}", flush=True)
        raise RuntimeError(
            "Backend failures; rerun the same command with --resume: "
            + json.dumps(failures, ensure_ascii=False)
        )
    summarize(args, splits, args.backends, require_all=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
