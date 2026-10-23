#!/usr/bin/env python3
"""Run 12 independent backends on PAMAP2 semantic-v3 A2 strict ZSL.

Every selected backend is trained and evaluated on all 20 seed-42 splits.  All
attribute classifiers are independent: predicted attributes are never fed into
later classifiers.  Backends execute sequentially so one physical GPU can be
shared safely, and completed backend/split artifacts can be reused with
``--resume``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROTOCOL_SCRIPT = (
    PROJECT / "scripts/run_pamap2_a2_autogluon_soft_oof_catboost_forward.py"
)
PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "pamap2_a2_semantic_v3_protocol", PROTOCOL_SCRIPT
)
protocol = importlib.util.module_from_spec(PROTOCOL_SPEC)
assert PROTOCOL_SPEC.loader is not None
PROTOCOL_SPEC.loader.exec_module(protocol)

DATASET = "pamap2"
DATASET_CONFIG = PROJECT / "configs/datasets/pamap2.json"
SPLITS = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_splits_seed42.json"
SAMPLES = (
    PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_eval50_samples_seed42.json"
)
A2_ROOT = PROJECT / "runtime/refinement_semantic_v3/pamap2_gpt_5_6_sol_11_15"
RUN_ROOT = PROJECT / "runtime/zshar_runs"
PROTOCOL_ROOT = PROJECT / "runtime/pamap2_a2_semantic_v3_independent_backends_seed42"
COMPARISON_NAME = (
    "pamap2_zsl_a2_semantic_v3_independent_backend_comparison_"
    "random20_eval50_seed42"
)
MANAGER_MANIFEST = "attribute_backend_manifest.json"

BACKENDS = (
    "catboost",
    "logistic_regression",
    "mlp_independent",
    "neuralnet_torch",
    "neuralnet_fastai",
    "xgboost",
    "extra_trees",
    "coral_mlp",
    "random_forest",
    "lightgbm",
    "ordinal_logistic",
    "linear_svm",
)
METRICS = (
    "activity_top1_accuracy",
    "activity_top3_recall",
    "activity_macro_f1",
    "attribute_exact_match",
)
REQUIRED_MODULES = {
    "catboost": "catboost",
    "mlp_independent": "torch",
    "neuralnet_torch": "torch",
    "neuralnet_fastai": "fastai",
    "xgboost": "xgboost",
    "coral_mlp": "torch",
    "lightgbm": "lightgbm",
    "ordinal_logistic": "torch",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_backends(value: str) -> list[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise argparse.ArgumentTypeError("At least one backend is required")
    unknown = sorted(set(requested).difference(BACKENDS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown backends {unknown}; choices are {list(BACKENDS)}"
        )
    if len(requested) != len(set(requested)):
        raise argparse.ArgumentTypeError("Backend list contains duplicates")
    return requested


def backend_config(backend: str, num_cpus: int) -> dict[str, Any]:
    config: dict[str, Any] = {"random_state": 42}
    if backend == "catboost":
        config.update(
            {
                "n_jobs": num_cpus,
                "task_type": "GPU",
                # CUDA_VISIBLE_DEVICES maps the selected physical GPU to device 0.
                "devices": "0",
                "allow_writing_files": False,
            }
        )
    elif backend == "lightgbm":
        config.update({"n_jobs": num_cpus, "device": "cuda"})
    elif backend == "xgboost":
        config.update({"n_jobs": num_cpus, "tree_method": "hist", "device": "cuda"})
    elif backend in {"extra_trees", "random_forest"}:
        config["n_jobs"] = num_cpus
    elif backend == "linear_svm":
        # The ordinal-probability prototype matcher requires probabilities.
        config["calibrate_proba"] = True
    elif backend in {
        "mlp_independent",
        "neuralnet_torch",
        "neuralnet_fastai",
        "coral_mlp",
        "ordinal_logistic",
    }:
        config["device"] = "cuda:0"
    elif backend != "logistic_regression":
        raise ValueError(f"Unsupported backend: {backend}")
    return config


def run_name(backend: str) -> str:
    return (
        f"pamap2_zsl_{backend}_a2_semantic_v3_independent_"
        "random20_eval50_seed42"
    )


def validate_dependencies(backends: Sequence[str]) -> None:
    missing = sorted(
        {
            module
            for backend, module in REQUIRED_MODULES.items()
            if backend in backends and importlib.util.find_spec(module) is None
        }
    )
    if missing:
        raise RuntimeError(
            "Missing backend dependencies in the active Python environment: "
            + ", ".join(missing)
        )


def preflight(
    backends: Sequence[str], num_cpus: int, gpu_id: int
) -> tuple[Any, list[Any], list[int]]:
    validate_dependencies(backends)
    if protocol.DATASET != DATASET:
        raise AssertionError("Imported protocol is not PAMAP2")
    if protocol.SPLITS.resolve() != SPLITS.resolve():
        raise AssertionError("Imported protocol uses a different split manifest")
    if protocol.SAMPLES.resolve() != SAMPLES.resolve():
        raise AssertionError("Imported protocol uses a different sample manifest")
    if protocol.A2_ROOT.resolve() != A2_ROOT.resolve():
        raise AssertionError("Imported protocol uses a different refined A2 root")

    split_payload = protocol.read_json(SPLITS)
    available = [int(item["split_index"]) for item in split_payload.get("splits", [])]
    if len(available) != 20 or len(set(available)) != 20:
        raise ValueError("Expected exactly 20 distinct PAMAP2 seed-42 splits")
    spec, splits, _, verified = protocol.preflight(available)
    if verified != available or len(splits) != 20:
        raise ValueError("PAMAP2 protocol validation did not return all 20 splits")

    write_json(
        PROTOCOL_ROOT / "protocol.json",
        {
            "schema_version": 1,
            "dataset": DATASET,
            "scenario": "zsl",
            "attribute_set": "semantic_v3_refined_A2_per_split_11_15d",
            "attribute_vectors_root": str(A2_ROOT.resolve()),
            "split_manifest": str(SPLITS.resolve()),
            "sample_manifest": str(SAMPLES.resolve()),
            "seed": 42,
            "split_count": 20,
            "unseen_classes_per_split": 4,
            "evaluation_samples_per_unseen_class": 50,
            "inference": "independent non-chain attribute classifiers",
            "prototype_metric": "ordinal_probability",
            "attribute_weighting": "validation_accuracy",
            "physical_gpu": gpu_id,
            "backends": list(backends),
            "backend_configs": {
                name: backend_config(name, num_cpus) for name in backends
            },
        },
    )
    return spec, splits, available


def expected_config(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "scenario": "zsl",
        "backend": backend,
        "backend_config": backend_config(backend, args.num_cpus),
        "seed": 42,
        "split_manifest": str(SPLITS),
        "sample_manifest": str(SAMPLES),
        "attribute_vectors_root": str(A2_ROOT),
        "samples_per_activity": 50,
        "prototype_metric": "ordinal_probability",
        "attribute_weighting": "validation_accuracy",
    }


def validate_existing_config(args: argparse.Namespace, backend: str) -> None:
    path = args.run_root / DATASET / run_name(backend) / "config.json"
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
        DATASET,
        "--dataset-config",
        str(DATASET_CONFIG),
        "--scenario",
        "zsl",
        "--backend",
        backend,
        "--backend-config",
        json.dumps(backend_config(backend, args.num_cpus), separators=(",", ":")),
        "--seed",
        "42",
        "--split-manifest",
        str(SPLITS),
        "--sample-manifest",
        str(SAMPLES),
        "--attribute-vectors-root",
        str(A2_ROOT),
        "--samples-per-activity",
        "50",
        "--prototype-metric",
        "ordinal_probability",
        "--attribute-weighting",
        "validation_accuracy",
        "--gpu-id",
        str(args.gpu_id),
        "--run-root",
        str(args.run_root),
        "--run-name",
        run_name(backend),
    ]
    if mode == "evaluate":
        command += ["--evaluate-only", "--reuse-models"]
    elif args.resume:
        command.append("--reuse-models")
    return command


def backend_complete(
    args: argparse.Namespace, backend: str, expected_splits: set[int]
) -> bool:
    run_dir = args.run_root / DATASET / run_name(backend)
    summary = run_dir / "summary.csv"
    if not summary.is_file():
        return False
    try:
        frame = pd.read_csv(summary)
        indices = set(pd.to_numeric(frame["split_index"], errors="raise").astype(int))
    except (KeyError, ValueError, pd.errors.EmptyDataError):
        return False
    if len(frame) != 20 or indices != expected_splits:
        return False
    if set(frame["backend"].astype(str)) != {backend}:
        return False
    return all(
        (run_dir / f"split_{index:03d}" / "models" / MANAGER_MANIFEST).is_file()
        for index in expected_splits
    )


def archive_partial_models(args: argparse.Namespace, backend: str) -> None:
    run_dir = args.run_root / DATASET / run_name(backend)
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
        print(f"Preserved incomplete {backend} models: {destination}", flush=True)


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
    backends: Sequence[str],
    expected_splits: set[int],
    *,
    require_all: bool,
) -> Path:
    rows: list[dict[str, Any]] = []
    split_frames: list[pd.DataFrame] = []
    incomplete: list[str] = []
    for backend in backends:
        run_dir = args.run_root / DATASET / run_name(backend)
        if not backend_complete(args, backend, expected_splits):
            incomplete.append(backend)
            continue
        frame = pd.read_csv(run_dir / "summary.csv").sort_values("split_index")
        split_frames.append(frame.assign(comparison_backend=backend))
        row: dict[str, Any] = {"backend": backend, "split_count": 20}
        for metric in METRICS:
            values = pd.to_numeric(frame[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    if require_all and incomplete:
        raise RuntimeError(f"Incomplete backends: {incomplete}")
    if not rows:
        raise RuntimeError("No backend has a complete 20-split result")

    output = args.run_root / DATASET / COMPARISON_NAME
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.concat(split_frames, ignore_index=True).to_csv(
        output / "all_split_metrics.csv", index=False
    )
    write_json(
        output / "comparison.json",
        {
            "dataset": DATASET,
            "scenario": "zsl",
            "attribute_set": "semantic_v3_refined_A2_per_split_11_15d",
            "attribute_vectors_root": str(A2_ROOT.resolve()),
            "seed": 42,
            "split_count_per_backend": 20,
            "samples_per_unseen_activity": 50,
            "inference": "independent non-chain attribute classifiers",
            "prototype_metric": "ordinal_probability",
            "attribute_weighting": "validation_accuracy",
            "completed_backends": [row["backend"] for row in rows],
            "incomplete_backends": incomplete,
            "metrics": rows,
        },
    )
    display = pd.DataFrame(rows)
    for column in display.columns:
        if column.endswith(("_mean", "_std", "_min", "_max")):
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
        default="train-test",
        nargs="?",
    )
    parser.add_argument("--backends", type=parse_backends, default=list(BACKENDS))
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately instead of continuing with later backends after an error",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.run_root = args.run_root.expanduser().resolve()
    if args.gpu_id < 0 or args.num_cpus <= 0:
        raise ValueError("GPU id must be nonnegative and --num-cpus must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    _, _, available = preflight(args.backends, args.num_cpus, args.gpu_id)
    expected_splits = set(available)
    for backend in args.backends:
        validate_existing_config(args, backend)
    print(
        "PAMAP2 semantic-v3 A2 independent backend benchmark\n"
        f"  physical GPU: {args.gpu_id} (logical device 0 for GPU backends)\n"
        f"  backends: {' -> '.join(args.backends)}\n"
        f"  splits: {available}\n"
        f"  split manifest: {SPLITS}\n"
        f"  sample manifest: {SAMPLES}\n"
        f"  A2 root: {A2_ROOT}\n"
        "  evaluation: 4 unseen activities x 50 fixed samples per split",
        flush=True,
    )
    if args.mode == "check":
        print("Check only: no classifier was trained or evaluated.", flush=True)
        return 0
    if args.mode == "report":
        summarize(args, args.backends, expected_splits, require_all=True)
        return 0

    failures: dict[str, str] = {}
    for position, backend in enumerate(args.backends, 1):
        run_dir = args.run_root / DATASET / run_name(backend)
        print(
            f"\nStage {position}/{len(args.backends)}: {backend} -> {run_dir}",
            flush=True,
        )
        if args.mode == "train-test" and backend_complete(
            args, backend, expected_splits
        ):
            print(f"Skipping completed 20-split backend: {backend}", flush=True)
            continue
        try:
            if args.mode == "train-test":
                archive_partial_models(args, backend)
            execute(runner_command(args, backend, args.mode), args.gpu_id)
            if not backend_complete(args, backend, expected_splits):
                raise RuntimeError(
                    f"{backend} exited without a complete 20-split result"
                )
        except Exception as exc:  # Continue so one backend does not block the queue.
            failures[backend] = f"{type(exc).__name__}: {exc}"
            print(f"Backend failed: {backend}: {failures[backend]}", flush=True)
            if args.fail_fast:
                raise

    try:
        summarize(
            args,
            args.backends,
            expected_splits,
            require_all=not failures,
        )
    except RuntimeError as exc:
        print(f"No final complete comparison yet: {exc}", flush=True)
    if failures:
        raise RuntimeError(
            "Backend failures; rerun the same command with --resume: "
            + json.dumps(failures, ensure_ascii=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
