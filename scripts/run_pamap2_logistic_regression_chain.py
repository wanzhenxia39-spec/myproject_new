#!/usr/bin/env python3
"""Run PAMAP2 seed-42 ZSL with both Logistic Regression chain orders.

The protocol is fixed to myproject_new's 20 split/sample manifests.  The A2
vectors and order file are the exact artifacts used to prepare the Logistic
Regression order.  It runs forward first and its exact reverse second, then
writes a paired comparison.  This launcher never prepares a new order.
"""
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
from typing import Any, Mapping


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
sys.path.insert(0, str(SRC))

SPLITS = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_splits_seed42.json"
SAMPLES = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_eval50_samples_seed42.json"
A2_ROOT = Path(
    "/home/wanzhen/LLMbasedZeroshotHARAgent/pamap2_pipeline_runs/"
    "pamap2_random20_unseen4_eval50_no_rerank_keep_models_seed42"
)
ORDERS = Path(
    "/home/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs/pamap2/"
    "pamap2_a2_logistic_order_preparation_seed42/attribute_orders_logistic.json"
)
RUN_ROOT = PROJECT / "runtime/zshar_runs"
RUN_NAME_PREFIX = "pamap2_zsl_logistic_regression_chain_a2_lr"
RUN_NAME_SUFFIX = "random20_eval50_seed42"
VARIANTS = ("forward", "reverse")
METRICS = (
    "activity_top1_accuracy",
    "activity_top3_recall",
    "activity_macro_f1",
    "attribute_exact_match",
)
MANAGER_MANIFEST = "attribute_backend_manifest.json"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def run_name(prefix: str, variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown order variant: {variant}")
    return f"{prefix}_{variant}_order_{RUN_NAME_SUFFIX}"


def resolve_vector(split_index: int) -> Path:
    candidates = (
        A2_ROOT / f"split_{split_index}/activity_attribute_vectors_a2.json",
        A2_ROOT / f"split_{split_index:03d}/activity_attribute_vectors_a2.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing A2 vectors for split {split_index}: "
        + " or ".join(map(str, candidates))
    )


def validate_protocol() -> list[int]:
    from zshar.data import load_activity_vectors

    split_payload = read_json(SPLITS)
    sample_payload = read_json(SAMPLES)
    order_payload = read_json(ORDERS)
    split_rows = split_payload.get("splits", [])
    indices = [int(row["split_index"]) for row in split_rows]
    if (
        split_payload.get("dataset") != "pamap2"
        or int(split_payload.get("seed", -1)) != 42
        or len(indices) != 20
        or len(set(indices)) != 20
    ):
        raise ValueError(f"Expected the fixed PAMAP2 20-split seed-42 manifest: {SPLITS}")
    if (
        sample_payload.get("dataset") != "pamap2"
        or int(sample_payload.get("seed", -1)) != 42
        or int(sample_payload.get("samples_per_activity", -1)) != 50
    ):
        raise ValueError(f"Expected the PAMAP2 seed-42 eval50 manifest: {SAMPLES}")
    raw_samples = sample_payload.get("samples", {})
    if set(map(int, raw_samples)) != set(indices):
        raise ValueError("Sample manifest split indices differ from the split manifest")
    for index in indices:
        activities = raw_samples[str(index)]
        if len(activities) != 4 or any(
            len(values) != 50 or len(set(map(int, values))) != 50
            for values in activities.values()
        ):
            raise ValueError(
                f"Split {index} must contain four unseen activities with 50 unique samples"
            )

    if (
        order_payload.get("dataset") != "pamap2"
        or int(order_payload.get("seed", -1)) != 42
        or order_payload.get("attribute_set") != "A2_refined_per_split"
    ):
        raise ValueError(f"Expected PAMAP2 seed-42 split-specific A2 orders: {ORDERS}")
    recorded_split_hash = order_payload.get("split_manifest_sha256")
    if recorded_split_hash != file_hash(SPLITS):
        raise ValueError(
            "The Logistic Regression order was prepared from a different split manifest"
        )
    order_splits = order_payload.get("splits", {})
    if set(map(int, order_splits)) != set(indices):
        raise ValueError("Order-file split indices differ from the split manifest")

    for index in indices:
        vector_path = resolve_vector(index)
        order = order_splits[str(index)]
        recorded_vector_hash = order.get("attribute_vectors_sha256")
        if recorded_vector_hash != file_hash(vector_path):
            raise ValueError(f"A2 vectors changed after preparing split {index}'s order")
        _, attributes = load_activity_vectors(vector_path)
        forward = list(map(str, order.get("forward", [])))
        reverse = list(map(str, order.get("reverse", [])))
        if (
            len(forward) != len(set(forward))
            or set(forward) != set(attributes)
            or reverse != forward[::-1]
        ):
            raise ValueError(f"Invalid Logistic Regression order for split {index}")
    return indices


def experiment_config(args: argparse.Namespace, variant: str):
    from zshar.pipeline.runner import ExperimentConfig

    selected = (
        {int(value) for value in args.split_indices.split(",") if value.strip()}
        if args.split_indices
        else None
    )
    return ExperimentConfig(
        dataset="pamap2",
        scenario="zsl",
        backend="logistic_regression_chain",
        backend_config={
            "random_state": 42,
            "max_iter": args.max_iter,
            "class_weight": "balanced",
            "C": args.c,
            "consider_labels_correlation": True,
            "chain_feature_encoding": "ordinal",
        },
        seed=42,
        split_manifest=str(SPLITS),
        sample_manifest=str(SAMPLES),
        attribute_vectors_root=str(A2_ROOT),
        attribute_order_config=str(ORDERS),
        attribute_order_variant=variant,
        split_indices=selected,
        samples_per_activity=50,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        run_root=str(args.run_root.resolve()),
        run_name=run_name(args.run_name_prefix, variant),
        reuse_models=args.resume,
        evaluate_only=False,
        dry_run=args.dry_run,
    )


def validate_existing_run(config: Any) -> Path:
    run_dir = Path(config.run_root) / "pamap2" / str(config.run_name)
    saved_config = run_dir / "config.json"
    if saved_config.is_file():
        previous = read_json(saved_config)
        current = asdict(config)
        ignored = {"reuse_models", "evaluate_only", "dry_run"}
        if {key: value for key, value in previous.items() if key not in ignored} != {
            key: value for key, value in current.items() if key not in ignored
        }:
            raise ValueError(
                f"Existing run uses a different experiment protocol: {saved_config}"
            )
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
            "Incomplete model directories exist; rerun with --resume to preserve and retrain "
            + ", ".join(str(path) for path in partial)
        )
    if dry_run:
        print("Dry run: incomplete model directories would be preserved:")
        for path in partial:
            print(f"  {path}")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in partial:
        destination = path.with_name(f"models.incomplete_{stamp}")
        counter = 1
        while destination.exists():
            destination = path.with_name(f"models.incomplete_{stamp}_{counter}")
            counter += 1
        shutil.move(str(path), str(destination))
        print(f"Preserved incomplete model directory: {destination}", flush=True)


def summarize(
    args: argparse.Namespace, configs: list[Any], expected_indices: list[int]
) -> Path:
    import pandas as pd

    expected = set(expected_indices)
    frames: dict[str, pd.DataFrame] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for config in configs:
        variant = str(config.attribute_order_variant)
        path = Path(config.run_root) / "pamap2" / str(config.run_name) / "summary.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed summary: {path}")
        frame = pd.read_csv(path).set_index("split_index").sort_index()
        if frame.index.has_duplicates or set(map(int, frame.index)) != expected:
            raise ValueError(
                f"Expected exactly {len(expected)} completed selected splits: {path}"
            )
        if set(frame["backend"].astype(str)) != {"logistic_regression_chain"}:
            raise ValueError(f"Unexpected backend in {path}")
        if set(frame["attribute_order_variant"].astype(str)) != {variant}:
            raise ValueError(f"Unexpected order variant in {path}")
        frames[variant] = frame
        for metric in METRICS:
            values = pd.to_numeric(frame[metric], errors="raise")
            aggregate_rows.append(
                {
                    "attribute_order": variant,
                    "metric": metric,
                    "split_count": len(frame),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )

    forward = frames["forward"]
    reverse = frames["reverse"]
    for column in (
        "seen_activities",
        "unseen_activities",
        "eval_samples",
        "attribute_vectors_sha256",
        "attribute_order_sha256",
    ):
        if not forward[column].equals(reverse[column]):
            raise ValueError(f"Forward/reverse protocols differ in {column}")

    paired = pd.DataFrame(index=forward.index)
    for metric in METRICS:
        paired[f"forward_{metric}"] = forward[metric]
        paired[f"reverse_{metric}"] = reverse[metric]
        paired[f"reverse_minus_forward_{metric}"] = reverse[metric] - forward[metric]

    output = (
        args.run_root.resolve()
        / "pamap2"
        / f"{args.run_name_prefix}_forward_reverse_comparison_{RUN_NAME_SUFFIX}"
    )
    output.mkdir(parents=True, exist_ok=True)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output / "comparison.csv", index=False)
    paired.to_csv(output / "paired_split_metrics.csv")
    write_json(
        output / "comparison.json",
        {
            "dataset": "pamap2",
            "scenario": "zsl",
            "attribute_set": "A2_refined_per_split",
            "backend": "logistic_regression_chain",
            "hard_labels_chained": True,
            "chain_feature_encoding": "ordinal",
            "split_count_per_order": len(expected),
            "order_sequence": list(VARIANTS),
            "attribute_order_config": str(ORDERS),
            "metrics": aggregate_rows,
        },
    )
    print("\n" + aggregate.to_string(index=False), flush=True)
    print(f"Forward/reverse comparison saved to: {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name-prefix", default=RUN_NAME_PREFIX)
    parser.add_argument("--split-indices", help="Optional comma-separated subset for a smoke test")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu_id < 0:
        raise ValueError("--gpu-id must be nonnegative")
    if args.max_iter <= 0 or args.c <= 0:
        raise ValueError("--max-iter and --c must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    all_indices = validate_protocol()
    configs = [experiment_config(args, variant) for variant in VARIANTS]
    selected = sorted(configs[0].split_indices) if configs[0].split_indices else all_indices
    if not set(selected).issubset(all_indices):
        raise ValueError(f"Unknown --split-indices: {sorted(set(selected) - set(all_indices))}")
    run_dirs = [validate_existing_run(config) for config in configs]
    for run_dir in run_dirs:
        preserve_partial_models(run_dir, resume=args.resume, dry_run=args.dry_run)
    print(
        "PAMAP2 Logistic Regression hard-label chain\n"
        "  execution order: forward -> reverse\n"
        f"  physical GPU visibility: {args.gpu_id}\n"
        "  estimator execution: scikit-learn CPU (CUDA_VISIBLE_DEVICES is pinned for protocol isolation)\n"
        f"  split manifest: {SPLITS}\n"
        f"  sample manifest: {SAMPLES}\n"
        f"  A2 root: {A2_ROOT}\n"
        f"  Logistic Regression order: {ORDERS}\n"
        f"  selected splits: {selected}\n"
        f"  forward run directory: {run_dirs[0]}\n"
        f"  reverse run directory: {run_dirs[1]}",
        flush=True,
    )

    from zshar.datasets import load_dataset_spec
    from zshar.pipeline.runner import run_experiment

    spec = load_dataset_spec("pamap2")
    for position, config in enumerate(configs, 1):
        print(
            f"\nStage {position}/{len(configs)}: "
            f"{config.attribute_order_variant} order",
            flush=True,
        )
        output = run_experiment(spec, config)
        print(f"Completed {config.attribute_order_variant}: {output}", flush=True)
    if not args.dry_run:
        summarize(args, configs, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
