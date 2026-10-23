#!/usr/bin/env python3
"""Run independent MHEALTH semantic-v3 A2 classifiers on 20 strict-ZSL splits.

The five attribute-classifier backends are deliberately non-chain: each
attribute model receives sensor features only.  Refined, split-specific A2
vectors must come from ``configs/attributes``; the launcher refuses any other
location.  Backends run sequentially and completed split models are reusable
with ``--resume``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

DATASET = "mhealth"
DATASET_CONFIG = PROJECT / "configs/datasets/mhealth.json"
SPLITS = PROJECT / "split/zero_shot_split/mhealth_unseen4_random20_splits_seed42.json"
SAMPLES = PROJECT / "split/zero_shot_split/mhealth_unseen4_random20_eval30_samples_seed42.json"
A2_ROOT = PROJECT / "configs/attributes/mhealth_semantic_v3_refined_a2_seed42"
RUN_ROOT = PROJECT / "runtime/zshar_runs"
PROTOCOL_ROOT = PROJECT / "runtime/mhealth_a2_independent_backends_seed42"
COMPARISON_NAME = (
    "mhealth_zsl_a2_semantic_v3_independent_backend_comparison_"
    "random20_eval30_seed42"
)
BACKENDS = ("catboost", "xgboost", "extra_trees", "random_forest", "lightgbm")
METRICS = (
    "activity_top1_accuracy",
    "activity_top3_recall",
    "activity_macro_f1",
    "attribute_exact_match",
)
REQUIRED_MODULES = {
    "catboost": "catboost",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}
MANAGER_MANIFEST = "attribute_backend_manifest.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
                # CUDA_VISIBLE_DEVICES maps physical GPU 0 to logical device 0.
                "devices": "0",
                "allow_writing_files": False,
            }
        )
    elif backend == "xgboost":
        config.update({"n_jobs": num_cpus, "tree_method": "hist", "device": "cuda"})
    elif backend == "lightgbm":
        config.update({"n_jobs": num_cpus, "device": "cuda"})
    elif backend in {"extra_trees", "random_forest"}:
        # These sklearn backends are CPU implementations; the experiment queue
        # remains assigned to GPU 0 for the GPU-capable stages.
        config["n_jobs"] = num_cpus
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return config


def run_name(backend: str) -> str:
    return (
        f"mhealth_zsl_{backend}_a2_semantic_v3_independent_"
        "random20_eval30_seed42"
    )


def vector_path(split_index: int) -> Path:
    return A2_ROOT / f"split_{split_index}" / "activity_attribute_vectors_a2.json"


def validate_dependencies(backends: Sequence[str]) -> None:
    missing = sorted(
        {
            module
            for backend, module in REQUIRED_MODULES.items()
            if backend in backends and importlib.util.find_spec(module) is None
        }
    )
    if missing:
        raise RuntimeError("Missing backend dependencies: " + ", ".join(missing))


def preflight(backends: Sequence[str], num_cpus: int, gpu_id: int) -> tuple[Any, list[int]]:
    from zshar.data import load_activity_vectors, load_dataset_table, replace_activity_vectors
    from zshar.datasets import load_dataset_spec
    from zshar.splits import load_sample_manifest, load_split_manifest, sample_evaluation_rows

    validate_dependencies(backends)
    for path in (DATASET_CONFIG, SPLITS, SAMPLES, A2_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    required_parent = (PROJECT / "configs/attributes").resolve()
    if A2_ROOT.resolve().parent != required_parent:
        raise ValueError(f"Refined A2 root must be directly under {required_parent}: {A2_ROOT}")

    split_payload = read_json(SPLITS)
    sample_payload = read_json(SAMPLES)
    if split_payload.get("dataset") != DATASET or int(split_payload.get("seed", -1)) != 42:
        raise ValueError(f"Invalid seed-42 split manifest: {SPLITS}")
    if (
        sample_payload.get("dataset") != DATASET
        or int(sample_payload.get("seed", -1)) != 42
        or int(sample_payload.get("samples_per_activity", -1)) != 30
    ):
        raise ValueError(f"Invalid seed-42 eval30 sample manifest: {SAMPLES}")

    spec = load_dataset_spec(DATASET)
    splits = load_split_manifest(SPLITS, spec)
    indices = [int(split.index) for split in splits]
    if len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError("Expected exactly 20 distinct MHEALTH seed-42 splits")
    if set(map(int, sample_payload.get("samples", {}))) != set(indices):
        raise ValueError("Sample manifest does not cover the exact 20 split indices")

    samples = load_sample_manifest(SAMPLES)
    base = load_dataset_table(spec)
    vector_hashes: dict[str, str] = {}
    attribute_counts: dict[str, int] = {}
    for split in splits:
        path = vector_path(split.index)
        if not path.is_file():
            raise FileNotFoundError(f"Missing refined A2 vector for split {split.index}: {path}")
        vectors, attributes = load_activity_vectors(path)
        if set(vectors) != set(spec.activity_names):
            raise ValueError(f"A2 activity mismatch for split {split.index}: {path}")
        levels = {value for row in vectors.values() for value in row.values()}
        if not levels <= {"Low", "Medium", "High"}:
            raise ValueError(f"Invalid A2 levels for split {split.index}: {sorted(levels)}")
        if not 11 <= len(attributes) <= 15:
            raise ValueError(
                f"Expected 11-15 refined attributes for split {split.index}, got {len(attributes)}"
            )
        table = replace_activity_vectors(base, path)
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="zsl",
            samples_per_activity=30,
            seed=42,
            manifest_ids=samples[split.index],
        )
        if len(evaluation) != 120 or not evaluation.activity_name.value_counts().eq(30).all():
            raise ValueError(f"Split {split.index} does not resolve to 4 x 30 fixed samples")
        vector_hashes[str(split.index)] = sha256(path)
        attribute_counts[str(split.index)] = len(attributes)

    write_json(
        PROTOCOL_ROOT / "protocol.json",
        {
            "schema_version": 1,
            "dataset": DATASET,
            "scenario": "zsl",
            "attribute_set": "semantic_v3_refined_A2_per_split_11_15d",
            "attribute_vectors_root": str(A2_ROOT.resolve()),
            "attribute_vector_sha256": vector_hashes,
            "attribute_counts": attribute_counts,
            "split_manifest": str(SPLITS.resolve()),
            "split_manifest_sha256": sha256(SPLITS),
            "sample_manifest": str(SAMPLES.resolve()),
            "sample_manifest_sha256": sha256(SAMPLES),
            "seed": 42,
            "split_count": 20,
            "unseen_classes_per_split": 4,
            "evaluation_samples_per_unseen_class": 30,
            "inference": "independent non-chain attribute classifiers",
            "prototype_metric": "ordinal_probability",
            "attribute_weighting": "validation_accuracy",
            "physical_gpu": gpu_id,
            "backends": list(backends),
            "backend_configs": {
                backend: backend_config(backend, num_cpus) for backend in backends
            },
        },
    )
    return spec, indices


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
        "samples_per_activity": 30,
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


def runner_command(args: argparse.Namespace, backend: str) -> list[str]:
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
        run_name(backend),
    ]
    if args.resume:
        command.append("--reuse-models")
    return command


def backend_complete(args: argparse.Namespace, backend: str, expected: set[int]) -> bool:
    run_dir = args.run_root / DATASET / run_name(backend)
    summary = run_dir / "summary.csv"
    if not summary.is_file():
        return False
    try:
        frame = pd.read_csv(summary)
        indices = set(pd.to_numeric(frame["split_index"], errors="raise").astype(int))
    except (KeyError, ValueError, pd.errors.EmptyDataError):
        return False
    if len(frame) != 20 or indices != expected or set(frame["backend"].astype(str)) != {backend}:
        return False
    return all(
        (run_dir / f"split_{index:03d}" / "models" / MANAGER_MANIFEST).is_file()
        for index in expected
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


def summarize(args: argparse.Namespace, backends: Sequence[str], expected: set[int]) -> Path:
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    incomplete: list[str] = []
    for backend in backends:
        run_dir = args.run_root / DATASET / run_name(backend)
        if not backend_complete(args, backend, expected):
            incomplete.append(backend)
            continue
        frame = pd.read_csv(run_dir / "summary.csv").sort_values("split_index")
        frames.append(frame.assign(comparison_backend=backend))
        row: dict[str, Any] = {"backend": backend, "split_count": 20}
        for metric in METRICS:
            values = pd.to_numeric(frame[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    if incomplete:
        raise RuntimeError(f"Incomplete backends: {incomplete}")

    output = args.run_root / DATASET / COMPARISON_NAME
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(output / "all_split_metrics.csv", index=False)
    write_json(
        output / "comparison.json",
        {
            "dataset": DATASET,
            "scenario": "zsl",
            "attribute_set": "semantic_v3_refined_A2_per_split_11_15d",
            "attribute_vectors_root": str(A2_ROOT.resolve()),
            "seed": 42,
            "split_count_per_backend": 20,
            "samples_per_unseen_activity": 30,
            "inference": "independent non-chain attribute classifiers",
            "prototype_metric": "ordinal_probability",
            "attribute_weighting": "validation_accuracy",
            "metrics": rows,
        },
    )
    print("\n" + pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Comparison saved to: {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("check", "train-test", "report"), default="train-test", nargs="?"
    )
    parser.add_argument("--backends", type=parse_backends, default=list(BACKENDS))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.run_root = args.run_root.expanduser().resolve()
    if args.gpu_id != 0:
        raise ValueError("This benchmark is reserved for physical GPU 0")
    if args.num_cpus <= 0:
        raise ValueError("--num-cpus must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    spec, available = preflight(args.backends, args.num_cpus, args.gpu_id)
    del spec
    expected = set(available)
    for backend in args.backends:
        validate_existing_config(args, backend)
    print(
        "MHEALTH semantic-v3 A2 independent backend benchmark\n"
        "  physical GPU: 0 (logical device 0 for GPU-capable backends)\n"
        f"  backends: {' -> '.join(args.backends)}\n"
        f"  splits: {available}\n"
        f"  split manifest: {SPLITS}\n"
        f"  sample manifest: {SAMPLES}\n"
        f"  refined A2 root: {A2_ROOT}\n"
        "  evaluation: 4 unseen activities x 30 fixed samples per split\n"
        "  classifier dependency: independent/non-chain",
        flush=True,
    )
    if args.mode == "check":
        print("Check only: no classifier was trained or evaluated.", flush=True)
        return 0
    if args.mode == "report":
        summarize(args, args.backends, expected)
        return 0

    failures: dict[str, str] = {}
    for position, backend in enumerate(args.backends, 1):
        run_dir = args.run_root / DATASET / run_name(backend)
        print(f"\nStage {position}/{len(args.backends)}: {backend} -> {run_dir}", flush=True)
        if backend_complete(args, backend, expected):
            print(f"Skipping completed 20-split backend: {backend}", flush=True)
            continue
        try:
            archive_partial_models(args, backend)
            execute(runner_command(args, backend), args.gpu_id)
            if not backend_complete(args, backend, expected):
                raise RuntimeError(f"{backend} exited without a complete 20-split result")
        except Exception as exc:
            failures[backend] = f"{type(exc).__name__}: {exc}"
            print(f"Backend failed: {backend}: {failures[backend]}", flush=True)
            if args.fail_fast:
                raise

    if failures:
        raise RuntimeError(
            "Backend failures; rerun the same command with --resume: "
            + json.dumps(failures, ensure_ascii=False)
        )
    summarize(args, args.backends, expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
