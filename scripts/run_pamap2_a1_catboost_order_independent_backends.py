#!/usr/bin/env python3
"""Run PAMAP2 A1-v1.5 independent baselines in CatBoost-derived order.

The two requested backends remain non-chain classifiers: every attribute uses
sensor features only.  The split-specific CatBoost forward order controls the
sequential training/evaluation order and is recorded in every run artifact.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATASET = "pamap2"
A1 = PROJECT / "configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5.json"
ONTOLOGY = PROJECT / "configs/attributes/initial_attributes.json"
SPLITS = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_splits_seed42.json"
SAMPLES = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_eval50_samples_seed42.json"
ORDERS = (
    PROJECT
    / "configs/attribute_orders/pamap2/initial_a1_v1_5_catboost_seen_only_seed42.json"
)
RUN_ROOT = PROJECT / "runtime/zshar_runs"
COMPARISON_NAME = (
    "pamap2_zsl_independent_catboost_logistic_regression_"
    "a1_v1_5_catboost_seen_only_forward_random20_eval50_seed42"
)
BACKENDS = ("catboost", "logistic_regression")
METRICS = (
    "activity_top1_accuracy",
    "activity_top3_recall",
    "activity_macro_f1",
    "attribute_exact_match",
)


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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_backends(value: str) -> list[str]:
    backends = [item.strip() for item in value.split(",") if item.strip()]
    if not backends or set(backends).difference(BACKENDS):
        raise argparse.ArgumentTypeError(f"Backends must be selected from {BACKENDS}")
    if len(backends) != len(set(backends)):
        raise argparse.ArgumentTypeError("Backends must not contain duplicates")
    return backends


def backend_config(backend: str, num_cpus: int) -> dict[str, Any]:
    if backend == "catboost":
        return {
            "random_state": 42,
            "n_jobs": num_cpus,
            "task_type": "GPU",
            # CUDA_VISIBLE_DEVICES maps physical GPU 1 to logical device 0.
            "devices": "0",
        }
    if backend == "logistic_regression":
        return {"random_state": 42}
    raise ValueError(f"Unsupported backend: {backend}")


def run_name(backend: str) -> str:
    return (
        f"pamap2_zsl_{backend}_independent_a1_v1_5_"
        "catboost_seen_only_forward_random20_eval50_seed42"
    )


def preflight() -> tuple[Any, list[Any], list[int]]:
    from zshar.data import load_activity_vectors, load_dataset_table
    from zshar.datasets import load_dataset_spec
    from zshar.splits import load_sample_manifest, load_split_manifest, sample_evaluation_rows

    for path in (A1, ONTOLOGY, SPLITS, SAMPLES, ORDERS):
        if not path.is_file():
            raise FileNotFoundError(path)
    spec = load_dataset_spec(DATASET)
    if spec.attributes_path.resolve() != A1.resolve():
        raise ValueError(f"PAMAP2 profile must use the new nine-dimensional A1: {A1}")
    vectors, attributes = load_activity_vectors(A1)
    expected = [str(item["name"]) for item in read_json(ONTOLOGY)["attributes"]]
    if set(vectors) != set(spec.activity_names) or attributes != expected or len(attributes) != 9:
        raise ValueError("PAMAP2 A1-v1.5 activity or attribute schema is invalid")

    splits = load_split_manifest(SPLITS, spec)
    indices = [int(split.index) for split in splits]
    if len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError("Expected exactly 20 distinct seed-42 PAMAP2 splits")
    sample_payload = read_json(SAMPLES)
    if (
        sample_payload.get("dataset") != DATASET
        or int(sample_payload.get("seed", -1)) != 42
        or int(sample_payload.get("samples_per_activity", -1)) != 50
        or set(map(int, sample_payload.get("samples", {}))) != set(indices)
    ):
        raise ValueError(f"Invalid seed-42 eval50 sample manifest: {SAMPLES}")
    sample_manifest = load_sample_manifest(SAMPLES)
    table = load_dataset_table(spec)
    for split in splits:
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="zsl",
            samples_per_activity=50,
            seed=42,
            manifest_ids=sample_manifest[split.index],
        )
        if len(evaluation) != 200 or not evaluation.activity_name.value_counts().eq(50).all():
            raise ValueError(f"Split {split.index} does not resolve to 4 x 50 samples")

    order_payload = read_json(ORDERS)
    entries = order_payload.get("splits", {})
    if (
        order_payload.get("dataset") != DATASET
        or order_payload.get("attribute_set") != "initial_A1_v1_5_9d"
        or int(order_payload.get("seed", -1)) != 42
        or order_payload.get("ranking_backend") != "catboost"
        or order_payload.get("attribute_vectors_sha256") != sha256(A1)
        or order_payload.get("split_manifest_sha256") != sha256(SPLITS)
        or set(map(int, entries)) != set(indices)
    ):
        raise ValueError(f"Invalid CatBoost order protocol: {ORDERS}")
    for split in splits:
        entry = entries[str(split.index)]
        order = list(map(str, entry.get("forward", [])))
        if (
            list(map(str, entry.get("seen_activities", []))) != list(split.seen_activities)
            or len(order) != 9
            or len(set(order)) != 9
            or set(order) != set(attributes)
        ):
            raise ValueError(f"Invalid forward order for split {split.index}")
    return spec, splits, indices


def selected_indices(args: argparse.Namespace, available: Sequence[int]) -> list[int]:
    if not args.split_indices:
        return list(available)
    selected = sorted({int(item) for item in args.split_indices.split(",") if item.strip()})
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"Unknown --split-indices: {unknown}")
    return selected


def experiment_config(args: argparse.Namespace, backend: str, selected: Sequence[int]):
    from zshar.pipeline.runner import ExperimentConfig

    return ExperimentConfig(
        dataset=DATASET,
        scenario="zsl",
        backend=backend,
        backend_config=backend_config(backend, args.num_cpus),
        seed=42,
        split_manifest=str(SPLITS),
        sample_manifest=str(SAMPLES),
        attribute_order_config=str(ORDERS),
        attribute_order_variant="forward",
        split_indices=set(selected),
        samples_per_activity=50,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        run_root=str(args.run_root.expanduser().resolve()),
        run_name=run_name(backend),
        reuse_models=args.resume,
        evaluate_only=False,
        dry_run=False,
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
    run_dir = Path(config.run_root) / DATASET / str(config.run_name)
    saved = run_dir / "config.json"
    if saved.is_file():
        previous = normalize(read_json(saved))
        current = normalize(asdict(config))
        ignored = {"reuse_models", "evaluate_only", "dry_run"}
        if {k: v for k, v in previous.items() if k not in ignored} != {
            k: v for k, v in current.items() if k not in ignored
        }:
            raise ValueError(f"Existing run uses a different protocol: {saved}")
    return run_dir


def preserve_partial_models(run_dir: Path, resume: bool) -> None:
    from zshar.models.serialization import MANAGER_MANIFEST

    for model_dir in run_dir.glob("split_*/models"):
        if (model_dir / MANAGER_MANIFEST).is_file():
            if not resume:
                raise FileExistsError(f"Existing models found under {run_dir}; use --resume")
            continue
        destination = model_dir.with_name("models_incomplete")
        suffix = 1
        while destination.exists():
            destination = model_dir.with_name(f"models_incomplete_{suffix}")
            suffix += 1
        shutil.move(str(model_dir), str(destination))


def summarize(args: argparse.Namespace, configs: Sequence[Any], selected: Sequence[int]) -> Path:
    rows: list[dict[str, Any]] = []
    split_frames: list[pd.DataFrame] = []
    expected = set(selected)
    for config in configs:
        path = Path(config.run_root) / DATASET / str(config.run_name) / "summary.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path).sort_values("split_index")
        if set(frame.split_index.astype(int)) != expected or len(frame) != len(expected):
            raise ValueError(f"Incomplete summary: {path}")
        if set(frame.backend.astype(str)) != {config.backend}:
            raise ValueError(f"Backend mismatch: {path}")
        if set(frame.attribute_order_variant.astype(str)) != {"forward"}:
            raise ValueError(f"Order mismatch: {path}")
        split_frames.append(frame.assign(comparison_backend=config.backend))
        for metric in METRICS:
            values = pd.to_numeric(frame[metric], errors="raise")
            rows.append(
                {
                    "backend": config.backend,
                    "attribute_order": "catboost_seen_only_forward",
                    "metric": metric,
                    "split_count": len(frame),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(frame) > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    output = args.run_root.expanduser().resolve() / DATASET / COMPARISON_NAME
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.concat(split_frames, ignore_index=True).to_csv(output / "all_split_metrics.csv", index=False)
    write_json(
        output / "comparison.json",
        {
            "dataset": DATASET,
            "scenario": "zsl",
            "attribute_set": "initial_A1_v1_5_9d",
            "backends": [config.backend for config in configs],
            "inference": "independent non-chain attribute classifiers",
            "attribute_order_variant": "forward",
            "attribute_order_config": str(ORDERS.resolve()),
            "attribute_order_sha256": sha256(ORDERS),
            "split_count": len(expected),
            "samples_per_unseen_activity": 50,
            "prototype_metric": "ordinal_probability",
            "attribute_weighting": "validation_accuracy",
            "metrics": rows,
        },
    )
    print("\n" + pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Comparison saved to: {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "train-test", "report"),
        default="train-test",
    )
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--backends", type=parse_backends, default=list(BACKENDS))
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--split-indices")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu_id < 0 or args.num_cpus <= 0:
        raise ValueError("GPU id must be nonnegative and --num-cpus must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    spec, _, available = preflight()
    selected = selected_indices(args, available)
    configs = [experiment_config(args, backend, selected) for backend in args.backends]
    run_dirs = [validate_existing_run(config) for config in configs]
    print(
        "PAMAP2 A1-v1.5 independent attribute backends in CatBoost forward order\n"
        f"  physical GPU: {args.gpu_id}\n"
        f"  backend sequence: {' -> '.join(args.backends)}\n"
        f"  selected splits: {selected}\n"
        f"  A1 vectors: {A1}\n"
        f"  order file: {ORDERS}\n"
        + "\n".join(
            f"  {config.backend} run: {run_dir}"
            for config, run_dir in zip(configs, run_dirs)
        ),
        flush=True,
    )
    if args.mode == "check":
        print("Check only: no classifier was trained or evaluated.", flush=True)
        return 0
    if args.mode == "report":
        summarize(args, configs, selected)
        return 0
    for run_dir in run_dirs:
        preserve_partial_models(run_dir, args.resume)
    from zshar.pipeline.runner import run_experiment

    for position, (config, run_dir) in enumerate(zip(configs, run_dirs), 1):
        print(
            f"\nStage {position}/{len(configs)}: {config.backend} -> {run_dir}",
            flush=True,
        )
        run_experiment(spec, config)
    summarize(args, configs, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
