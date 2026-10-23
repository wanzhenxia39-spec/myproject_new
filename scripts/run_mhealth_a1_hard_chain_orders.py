#!/usr/bin/env python3
"""Prepare split-specific A1 chain orders and run matched MHEALTH hard chains.

The nine-dimensional A1 prototypes are dataset-level.  This script creates an
exact per-split compatibility copy for the runner, estimates a forward order
from seen-only data for every split, defines reverse as its exact reversal, and
runs reverse followed by forward on physical GPU 1.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SPLITS = PROJECT / "split/zero_shot_split/mhealth_unseen4_random20_splits_seed42.json"
SAMPLES = PROJECT / "split/zero_shot_split/mhealth_unseen4_random20_eval30_samples_seed42.json"
A1 = PROJECT / "configs/attributes/activity_attribute_vectors_initial_mhealth_v1_5.json"
PROTOCOL_ROOT = PROJECT / "runtime/mhealth_a1_hard_chain_seed42"
A1_SPLIT_ROOT = PROTOCOL_ROOT / "attribute_vectors"
ORDER_WORK = PROTOCOL_ROOT / "order_preparation"
ORDER_FILE = ORDER_WORK / "attribute_orders.json"
RUN_ROOT = PROJECT / "runtime/zshar_runs"
VARIANTS = ("reverse", "forward")
METRICS = ("activity_top1_accuracy", "activity_top3_recall", "activity_macro_f1")
WEIGHTS = {
    "validation_balanced_accuracy": 0.45,
    "validation_macro_f1": 0.30,
    "inverse_normalized_ordinal_mae": 0.20,
    "mean_dependency_nmi": 0.05,
}
MANAGER_MANIFEST = "attribute_backend_manifest.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    print(f"Waiting for PID {pid} to exit before using physical GPU 1.", flush=True)
    while process_identity(pid) == identity:
        time.sleep(30)
    print("Previous GPU 1 process exited; continuing MHEALTH A1 experiment.", flush=True)


def validate_a1(vectors: Any, expected_activities: Sequence[str]) -> list[str]:
    if not isinstance(vectors, Mapping) or set(map(str, vectors)) != set(expected_activities):
        raise ValueError("A1 must contain exactly all 12 MHEALTH activities")
    rows = list(vectors.values())
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("A1 activity rows must be nonempty objects")
    attributes = list(map(str, rows[0]))
    if len(attributes) != 9 or len(set(attributes)) != 9:
        raise ValueError(f"Expected exactly nine distinct A1 attributes, found {attributes}")
    if any(list(map(str, row)) != attributes for row in rows):
        raise ValueError("All A1 activity rows must use the same ordered schema")
    if any(value not in {"Low", "Medium", "High"} for row in rows for value in row.values()):
        raise ValueError("A1 values must be Low, Medium, or High")
    return attributes


def preflight():
    from zshar.data import load_dataset_table, replace_activity_vectors
    from zshar.datasets import load_dataset_spec
    from zshar.splits import load_sample_manifest, load_split_manifest, sample_evaluation_rows

    for path in (SPLITS, SAMPLES, A1):
        if not path.is_file():
            raise FileNotFoundError(path)
    split_payload = read_json(SPLITS)
    sample_payload = read_json(SAMPLES)
    if split_payload.get("dataset") != "mhealth" or int(split_payload.get("seed", -1)) != 42:
        raise ValueError(f"Expected MHEALTH seed-42 splits: {SPLITS}")
    if (
        sample_payload.get("dataset") != "mhealth"
        or int(sample_payload.get("seed", -1)) != 42
        or int(sample_payload.get("samples_per_activity", -1)) != 30
    ):
        raise ValueError(f"Expected MHEALTH seed-42 eval30 samples: {SAMPLES}")

    spec = load_dataset_spec("mhealth", PROJECT / "configs/datasets/mhealth.json")
    splits = load_split_manifest(SPLITS, spec)
    if len(splits) != 20 or len({split.index for split in splits}) != 20:
        raise ValueError("Expected exactly 20 distinct MHEALTH splits")
    base = load_dataset_table(spec)
    table = replace_activity_vectors(base, A1)
    attributes = validate_a1(table.activity_vectors, spec.activity_names)
    samples = load_sample_manifest(SAMPLES)
    raw_samples = sample_payload.get("samples", {})
    if set(map(int, raw_samples)) != {split.index for split in splits}:
        raise ValueError("Sample manifest split indices do not match the split manifest")
    for split in splits:
        by_activity = raw_samples[str(split.index)]
        if set(map(str, by_activity)) != set(split.unseen_activities):
            raise ValueError(f"Split {split.index} sample activities do not match unseen classes")
        if len(by_activity) != 4 or any(
            len(ids) != 30 or len(set(map(int, ids))) != 30 for ids in by_activity.values()
        ):
            raise ValueError(f"Split {split.index} must have 4 unseen classes x 30 unique samples")
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="zsl",
            samples_per_activity=30,
            seed=42,
            manifest_ids=samples[split.index],
        )
        if len(evaluation) != 120 or not evaluation.activity_name.value_counts().eq(30).all():
            raise ValueError(f"Split {split.index} does not resolve to 120 balanced test samples")
        for activity, ids in by_activity.items():
            actual = evaluation.loc[
                evaluation.activity_name == activity, "window_index"
            ].astype(int)
            if set(actual) != set(map(int, ids)):
                raise ValueError(f"Split {split.index}/{activity} sample indices differ")
    print(
        "Validated MHEALTH A1 protocol: 12 activities x 9 attributes; "
        "20 seed-42 splits; each test split has 4 unseen classes x 30 fixed samples.",
        flush=True,
    )
    return spec, splits, table, attributes


def materialize_a1_split_vectors(splits: Sequence[Any]) -> None:
    """Create exact copies under the split-specific interface expected by the runner."""
    source_hash = file_hash(A1)
    files: dict[str, dict[str, Any]] = {}
    for split in splits:
        destination = A1_SPLIT_ROOT / f"split_{split.index}" / "activity_attribute_vectors_a2.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or file_hash(destination) != source_hash:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(A1, temporary)
            os.replace(temporary, destination)
        if file_hash(destination) != source_hash:
            raise AssertionError(f"A1 compatibility copy hash mismatch: {destination}")
        files[str(split.index)] = {"path": str(destination), "sha256": source_hash}
    write_json(
        A1_SPLIT_ROOT / "manifest.json",
        {
            "dataset": "mhealth",
            "seed": 42,
            "attribute_set": "initial_A1_v1_5",
            "source": str(A1.resolve()),
            "source_sha256": source_hash,
            "copy_policy": "byte-exact compatibility copies; no LLM generation",
            "split_count": len(splits),
            "splits": files,
        },
    )


def backend_config(args: argparse.Namespace, *, ranking: bool) -> dict[str, Any]:
    return {
        "time_limit": args.order_time_limit if ranking else args.time_limit,
        "presets": args.presets,
        "num_gpus": 1,
        "num_cpus": args.num_cpus,
        "num_bag_folds": args.num_bag_folds,
        "num_stack_levels": 0,
        "fit_weighted_ensemble": True,
        "model_suite": args.model_suite,
        "consider_labels_correlation": True,
        "lightgbm_device": "cuda",
        "gpu_model_types": ["NN_TORCH", "FASTAI", "CAT", "XGB"],
        "verbosity": args.verbosity,
    }


def rank_attributes(
    train_labels: pd.DataFrame, validation_report: Mapping[str, Any]
) -> dict[str, Any]:
    attributes = list(train_labels.columns)
    scores: dict[str, dict[str, Any]] = {}
    per_attribute = validation_report.get("per_attribute", {})
    for attribute in attributes:
        metric = per_attribute.get(attribute)
        if not isinstance(metric, Mapping):
            raise ValueError(f"Missing validation metrics for {attribute}")
        dependencies = [
            normalized_mutual_info_score(
                train_labels[attribute], train_labels[other], average_method="arithmetic"
            )
            for other in attributes
            if other != attribute
        ]
        components = {
            "validation_balanced_accuracy": float(metric["balanced_accuracy"]),
            "validation_macro_f1": float(metric["macro_f1"]),
            "inverse_normalized_ordinal_mae": 1.0 - float(metric["ordinal_mae"]),
            "mean_dependency_nmi": float(np.mean(dependencies)) if dependencies else 0.0,
        }
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in components.values()):
            raise ValueError(f"Invalid priority components for {attribute}: {components}")
        scores[attribute] = {
            **components,
            "constant": bool(train_labels[attribute].nunique() == 1),
            "priority": float(sum(WEIGHTS[key] * value for key, value in components.items())),
        }
    forward = sorted(
        attributes,
        key=lambda attribute: (
            scores[attribute]["constant"],
            -scores[attribute]["priority"],
            attributes.index(attribute),
        ),
    )
    return {"forward": forward, "reverse": forward[::-1], "scores": scores}


def archive_partial_model(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.incomplete_{stamp}")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.incomplete_{stamp}_{counter}")
        counter += 1
    shutil.move(str(path), str(backup))
    return backup


def preparation_signature(args: argparse.Namespace) -> dict[str, Any]:
    dataset_config = read_json(PROJECT / "configs/datasets/mhealth.json")
    feature_inputs = {}
    for value in dataset_config["files"].values():
        path = Path(value)
        feature_inputs[str(path)] = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    return {
        "algorithm_version": 1,
        "dataset": "mhealth",
        "seed": 42,
        "a1_sha256": file_hash(A1),
        "split_sha256": file_hash(SPLITS),
        "sample_sha256": file_hash(SAMPLES),
        "dataset_profile_sha256": file_hash(PROJECT / "configs/datasets/mhealth.json"),
        "feature_inputs": feature_inputs,
        "ranking_backend_config": backend_config(args, ranking=True),
        "ranking_weights": WEIGHTS,
    }


def prepare_orders(
    args: argparse.Namespace,
    splits: Sequence[Any],
    table: Any,
    attributes: Sequence[str],
) -> Path:
    from zshar.pipeline.training import load_or_train_attribute_models, split_train_validation

    ORDER_WORK.mkdir(parents=True, exist_ok=True)
    signature = preparation_signature(args)
    signature_path = ORDER_WORK / "preparation_config.json"
    with (ORDER_WORK / "prepare.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if signature_path.is_file() and read_json(signature_path) != signature:
            raise ValueError(
                f"Order-preparation inputs changed: {signature_path}. "
                "Use a new --protocol-root or restore the original settings."
            )
        write_json(signature_path, signature)
        entries: dict[str, Any] = {}
        for split in splits:
            split_work = ORDER_WORK / f"split_{split.index:03d}"
            record = split_work / "ranking.json"
            if record.is_file():
                saved = read_json(record)
                if set(saved.get("forward", [])) != set(attributes) or saved.get("reverse") != saved.get("forward", [])[::-1]:
                    raise ValueError(f"Invalid cached A1 order: {record}")
                entries[str(split.index)] = saved
                continue
            model_path = split_work / "models"
            if model_path.exists() and not (model_path / MANAGER_MANIFEST).is_file():
                if not args.resume:
                    raise ValueError(f"Incomplete ranking model: {model_path}; rerun with --resume")
                backup = archive_partial_model(model_path)
                print(f"Preserved incomplete ranking model: {backup}", flush=True)
            seen = table.frame[
                table.frame.activity_name.isin(split.seen_activities)
            ].copy()
            fitting, _ = split_train_validation(seen, seed=42)
            print(f"Preparing A1 order for split {split.index} on physical GPU 1.", flush=True)
            manager, report = load_or_train_attribute_models(
                reuse_models=args.resume,
                model_path=model_path,
                frame=seen,
                feature_columns=table.feature_columns,
                attribute_columns=list(attributes),
                backend="autogluon",
                backend_config={
                    key: value
                    for key, value in backend_config(args, ranking=True).items()
                    if key != "consider_labels_correlation"
                },
                seed=42,
            )
            result = rank_attributes(fitting[list(attributes)].astype(str), report)
            result.update(
                {
                    "split_index": split.index,
                    "seen_activities": list(split.seen_activities),
                    "a1_vectors": str(A1.resolve()),
                    "a1_sha256": file_hash(A1),
                    "validation_report": report,
                }
            )
            write_json(record, result)
            entries[str(split.index)] = result
            del manager
        payload = {
            "schema_version": 1,
            "dataset": "mhealth",
            "attribute_set": "initial_A1_v1_5_9d",
            "seed": 42,
            "selection_scope": "seen-class training data only; no unseen samples or labels",
            "ranking": WEIGHTS,
            "constant_attribute_policy": "place constants last in forward before exact reversal",
            "forward_definition": "descending weighted priority; ties follow A1 schema order",
            "reverse_definition": "exact reversal of the complete nine-attribute forward order",
            "a1_vectors": str(A1.resolve()),
            "a1_sha256": file_hash(A1),
            "preparation_config": signature,
            "generated_at": timestamp(),
            "splits": entries,
        }
        write_json(ORDER_FILE, payload)
    print(f"Saved/reused 20 split-specific A1 orders: {ORDER_FILE}", flush=True)
    return ORDER_FILE


def run_name(variant: str) -> str:
    return f"mhealth_zsl_autogluon_multilabel_hard_a1_{variant}_random20_eval30_seed42"


def runner_command(args: argparse.Namespace, variant: str, mode: str) -> list[str]:
    config = backend_config(args, ranking=False)
    command = [
        sys.executable,
        "-u",
        "-m",
        "zshar.cli",
        "run",
        "--dataset",
        "mhealth",
        "--dataset-config",
        str(PROJECT / "configs/datasets/mhealth.json"),
        "--scenario",
        "zsl",
        "--backend",
        "autogluon_multilabel",
        "--backend-config",
        json.dumps(
            {
                "consider_labels_correlation": True,
                "lightgbm_device": "cuda",
                "gpu_model_types": config["gpu_model_types"],
                "verbosity": args.verbosity,
            },
            separators=(",", ":"),
        ),
        "--seed",
        "42",
        "--split-manifest",
        str(SPLITS),
        "--sample-manifest",
        str(SAMPLES),
        "--attribute-vectors-root",
        str(A1_SPLIT_ROOT),
        "--attribute-order-config",
        str(ORDER_FILE),
        "--attribute-order-variant",
        variant,
        "--samples-per-activity",
        "30",
        "--prototype-metric",
        "ordinal_probability",
        "--attribute-weighting",
        "validation_accuracy",
        "--time-limit",
        str(args.time_limit),
        "--presets",
        args.presets,
        "--autogluon-model-suite",
        args.model_suite,
        "--num-cpus",
        str(args.num_cpus),
        "--num-gpus",
        "1",
        "--gpu-id",
        "1",
        "--num-bag-folds",
        str(args.num_bag_folds),
        "--num-stack-levels",
        "0",
        "--run-root",
        str(args.run_root),
        "--run-name",
        run_name(variant),
    ]
    if mode == "evaluate":
        command += ["--evaluate-only", "--reuse-models"]
    elif args.resume:
        command.append("--reuse-models")
    return command


def preserve_partial_models(args: argparse.Namespace, variant: str) -> None:
    directory = args.run_root / "mhealth" / run_name(variant)
    for model_dir in directory.glob("split_*/models"):
        if (model_dir / MANAGER_MANIFEST).is_file():
            if not args.resume:
                raise FileExistsError(f"Completed model exists: {model_dir}; use --resume")
            continue
        if not args.resume:
            raise FileExistsError(f"Partial model exists: {model_dir}; use --resume")
        backup = archive_partial_model(model_dir)
        print(f"Preserved incomplete classifier chain: {backup}", flush=True)


def execute(command: Sequence[str]) -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(SRC)
        + (os.pathsep + os.environ["PYTHONPATH"] if os.getenv("PYTHONPATH") else ""),
    }
    print("$ " + shlex.join(command), flush=True)
    subprocess.run(list(command), cwd=PROJECT, env=environment, check=True)


def summarize(args: argparse.Namespace, splits: Sequence[Any]) -> Path:
    expected = {split.index for split in splits}
    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        path = args.run_root / "mhealth" / run_name(variant) / "summary.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed summary: {path}")
        frame = pd.read_csv(path).set_index("split_index").sort_index()
        if frame.index.has_duplicates or set(map(int, frame.index)) != expected:
            raise ValueError(f"Expected exactly 20 completed splits: {path}")
        if set(frame["backend"].astype(str)) != {"autogluon_multilabel"}:
            raise ValueError(f"Unexpected backend in {path}")
        if set(frame["attribute_order_variant"].astype(str)) != {variant}:
            raise ValueError(f"Unexpected order variant in {path}")
        frames[variant] = frame
        for metric in METRICS:
            values = pd.to_numeric(frame[metric], errors="raise")
            rows.append(
                {
                    "attribute_order": variant,
                    "metric": metric,
                    "split_count": 20,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    reverse, forward = frames["reverse"], frames["forward"]
    for column in (
        "seen_activities",
        "unseen_activities",
        "eval_samples",
        "attribute_vectors_sha256",
        "attribute_order_sha256",
    ):
        if not reverse[column].equals(forward[column]):
            raise ValueError(f"Forward/reverse protocols differ in {column}")
    paired = pd.DataFrame(index=reverse.index)
    for metric in METRICS:
        paired[f"reverse_{metric}"] = reverse[metric]
        paired[f"forward_{metric}"] = forward[metric]
        paired[f"reverse_minus_forward_{metric}"] = reverse[metric] - forward[metric]

    output = args.run_root / "mhealth/mhealth_hard_chain_forward_reverse_a1_seed42"
    output.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "comparison.csv", index=False)
    paired.to_csv(output / "paired_split_metrics.csv")
    write_json(
        output / "comparison.json",
        {
            "dataset": "mhealth",
            "scenario": "zsl",
            "attribute_set": "initial_A1_v1_5_9d",
            "backend": "autogluon_multilabel",
            "hard_labels_chained": True,
            "split_count_per_order": 20,
            "order_sequence": list(VARIANTS),
            "metrics": rows,
        },
    )
    print("\n" + comparison.to_string(index=False), flush=True)
    print(f"Comparison saved to: {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("check", "prepare", "train-test", "evaluate", "report"),
        help="Validate only, prepare A1 orders, run both orders, reevaluate, or summarize.",
    )
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--time-limit", type=int, default=600, help="Seconds per chain attribute")
    parser.add_argument(
        "--order-time-limit", type=int, default=600, help="Seconds per independent order model"
    )
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--num-bag-folds", type=int, default=0)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--model-suite", choices=("default", "zshar_five"), default="default")
    parser.add_argument("--verbosity", type=int, choices=range(5), default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-pid", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global PROTOCOL_ROOT, A1_SPLIT_ROOT, ORDER_WORK, ORDER_FILE
    args = build_parser().parse_args(argv)
    args.run_root = args.run_root.expanduser().resolve()
    args.protocol_root = args.protocol_root.expanduser().resolve()
    PROTOCOL_ROOT = args.protocol_root
    A1_SPLIT_ROOT = PROTOCOL_ROOT / "attribute_vectors"
    ORDER_WORK = PROTOCOL_ROOT / "order_preparation"
    ORDER_FILE = ORDER_WORK / "attribute_orders.json"
    if min(args.time_limit, args.order_time_limit, args.num_cpus) <= 0:
        raise ValueError("Time limits and CPU count must be positive")
    if args.num_bag_folds < 0:
        raise ValueError("--num-bag-folds cannot be negative")
    if args.wait_pid is not None and (args.wait_pid <= 0 or args.wait_pid == os.getpid()):
        raise ValueError("--wait-pid must identify another positive process")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    identity = process_identity(args.wait_pid)
    spec, splits, table, attributes = preflight()
    materialize_a1_split_vectors(splits)
    if args.mode == "check":
        print(
            "Check only: A1 compatibility copies were verified; no AutoGluon model was trained.",
            flush=True,
        )
        return 0
    if args.mode == "report":
        summarize(args, splits)
        return 0

    wait_for_process(args.wait_pid, identity)
    prepare_orders(args, splits, table, attributes)
    if args.mode == "prepare":
        return 0
    print(
        "MHEALTH strict ZSL A1: hard-label AutoGluon multilabel chain; "
        "reverse 20 splits, then forward 20 splits; physical GPU 1.",
        flush=True,
    )
    del table, spec
    for variant in VARIANTS:
        command = runner_command(args, variant, args.mode)
        if args.mode == "train-test":
            preserve_partial_models(args, variant)
        execute(command)
    summarize(args, splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
