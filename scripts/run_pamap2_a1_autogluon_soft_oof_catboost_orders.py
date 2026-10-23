#!/usr/bin/env python3
"""Prepare A1 CatBoost orders and run PAMAP2 AutoGluon soft-OOF chains.

For each seed-42 split, nine independent CatBoost attribute classifiers are
trained on that split's seen-class sensor samples and the shared nine-
dimensional A1 v1.5 labels.  Their held-out validation scores define the
forward order; reverse is its exact reversal.  The same A1 vectors are then
used by AutoGluon soft-OOF chains in forward -> reverse order.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BASE_SCRIPT = PROJECT / "scripts/run_pamap2_autogluon_soft_oof_catboost_orders.py"
BASE_SPEC = importlib.util.spec_from_file_location(
    "pamap2_a1_soft_oof_catboost_common", BASE_SCRIPT
)
base = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(base)

A1 = PROJECT / "configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5.json"
DATASET_KEY = "pamap2"
DATASET_LABEL = "PAMAP2"
EVAL_SAMPLES_PER_ACTIVITY = 50
DEFAULT_GPU_ID = 0
ONTOLOGY = PROJECT / "configs/attributes/initial_attributes.json"
SPLITS = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_splits_seed42.json"
SAMPLES = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_eval50_samples_seed42.json"
PROTOCOL_ROOT = PROJECT / "runtime/pamap2_a1_catboost_seen_only_order_seed42"
ORDER_WORK = PROTOCOL_ROOT / "order_preparation"
ORDER_FILE = (
    PROJECT
    / "configs/attribute_orders/pamap2/initial_a1_v1_5_catboost_seen_only_seed42.json"
)
RUN_NAME_PREFIX = "pamap2_zsl_autogluon_soft_oof_a1_v1_5_catboost_seen_only"
ATTRIBUTE_SET = "initial_A1_v1_5_9d"
VARIANTS = ("forward", "reverse")
WEIGHTS = {
    "validation_balanced_accuracy": 0.45,
    "validation_macro_f1": 0.30,
    "inverse_normalized_ordinal_mae": 0.20,
    "mean_dependency_nmi": 0.05,
}
SELECTION_SCOPE = (
    "seen-class sensor samples only; independent per-attribute CatBoost models; "
    "no unseen samples, labels, or test metrics"
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def catboost_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "random_state": 42,
        "n_jobs": args.order_num_cpus,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "task_type": "GPU",
        # CUDA_VISIBLE_DEVICES maps the selected physical GPU to logical GPU 0.
        "devices": "0",
        "auto_class_weights": "Balanced",
        "allow_writing_files": False,
    }


def preflight() -> tuple[Any, list[Any], Any, list[int]]:
    from zshar.data import load_activity_vectors, load_dataset_table
    from zshar.datasets import load_dataset_spec
    from zshar.splits import load_sample_manifest, load_split_manifest, sample_evaluation_rows

    for path in (A1, ONTOLOGY, SPLITS, SAMPLES):
        if not path.is_file():
            raise FileNotFoundError(path)
    spec = load_dataset_spec(DATASET_KEY)
    if spec.attributes_path.resolve() != A1.resolve():
        raise ValueError(
            f"{DATASET_LABEL} dataset profile must point to the new A1 file: {A1}; "
            f"found {spec.attributes_path}"
        )
    vectors, attributes = load_activity_vectors(A1)
    ontology = read_json(ONTOLOGY)
    expected = [str(item["name"]) for item in ontology["attributes"]]
    if set(vectors) != set(spec.activity_names) or attributes != expected or len(attributes) != 9:
        raise ValueError(f"{DATASET_LABEL} A1 activities or nine-dimensional schema are invalid")

    splits = load_split_manifest(SPLITS, spec)
    indices = [split.index for split in splits]
    if len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError(f"Expected exactly 20 distinct {DATASET_LABEL} seed-42 splits")
    raw_samples = read_json(SAMPLES)
    if (
        raw_samples.get("dataset") != DATASET_KEY
        or int(raw_samples.get("seed", -1)) != 42
        or int(raw_samples.get("samples_per_activity", -1)) != EVAL_SAMPLES_PER_ACTIVITY
        or set(map(int, raw_samples.get("samples", {}))) != set(indices)
    ):
        raise ValueError(
            f"Invalid {DATASET_LABEL} seed-42 eval{EVAL_SAMPLES_PER_ACTIVITY} "
            f"sample manifest: {SAMPLES}"
        )
    sample_manifest = load_sample_manifest(SAMPLES)
    table = load_dataset_table(spec)
    if list(table.attribute_columns) != attributes:
        raise ValueError("Loaded PAMAP2 table does not use the expected nine-dimensional A1")
    for split in splits:
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="zsl",
            samples_per_activity=EVAL_SAMPLES_PER_ACTIVITY,
            seed=42,
            manifest_ids=sample_manifest[split.index],
        )
        expected_rows = len(split.unseen_activities) * EVAL_SAMPLES_PER_ACTIVITY
        if (
            len(evaluation) != expected_rows
            or not evaluation.activity_name.value_counts().eq(EVAL_SAMPLES_PER_ACTIVITY).all()
        ):
            raise ValueError(
                f"Split {split.index} does not resolve to "
                f"{len(split.unseen_activities)} x {EVAL_SAMPLES_PER_ACTIVITY} samples"
            )
    return spec, splits, table, indices


def selected_indices(args: argparse.Namespace, all_indices: Sequence[int]) -> list[int]:
    if not args.split_indices:
        return list(all_indices)
    selected = sorted({int(value) for value in args.split_indices.split(",") if value.strip()})
    unknown = sorted(set(selected).difference(all_indices))
    if unknown:
        raise ValueError(f"Unknown --split-indices: {unknown}")
    return selected


def preparation_signature(args: argparse.Namespace, selected: Sequence[int]) -> dict[str, Any]:
    dataset_config = PROJECT / f"configs/datasets/{DATASET_KEY}.json"
    profile = read_json(dataset_config)
    feature_inputs: dict[str, Any] = {}
    for value in profile.get("files", {}).values():
        path = Path(value)
        feature_inputs[str(path)] = {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
    return {
        "algorithm_version": 1,
        "dataset": DATASET_KEY,
        "seed": 42,
        "attribute_set": ATTRIBUTE_SET,
        "selection_scope": SELECTION_SCOPE,
        "validation_fraction": 0.3,
        "split_manifest_sha256": sha256(SPLITS),
        "sample_manifest_sha256": sha256(SAMPLES),
        "dataset_profile_sha256": sha256(dataset_config),
        "attribute_vectors_sha256": sha256(A1),
        "ontology_sha256": sha256(ONTOLOGY),
        "feature_inputs": feature_inputs,
        "selected_split_indices": list(selected),
        "ranking_backend": "catboost",
        "ranking_backend_config": catboost_config(args),
        "ranking_weights": WEIGHTS,
    }


def rank_split(args: argparse.Namespace, *, split: Any, table: Any) -> dict[str, Any]:
    from zshar.models.base import classification_metrics
    from zshar.models.boosting_backends import CatBoostBackend
    from zshar.pipeline.training import split_train_validation

    attributes = list(table.attribute_columns)
    seen = table.frame[table.frame.activity_name.isin(split.seen_activities)].copy()
    if set(seen.activity_name.unique()) != set(split.seen_activities):
        raise ValueError(f"Split {split.index} is missing seen-class training rows")
    fitting, validation = split_train_validation(seen, seed=42, validation_fraction=0.3)
    train_labels = fitting[attributes].astype(str)
    validation_labels = validation[attributes].astype(str)
    per_attribute: dict[str, Any] = {}
    print(
        f"Preparing A1 CatBoost order for split {split.index}: "
        f"train={len(fitting)}, validation={len(validation)}, attributes={len(attributes)}",
        flush=True,
    )
    for position, attribute in enumerate(attributes, 1):
        y_train = train_labels[attribute]
        if y_train.nunique() == 1:
            prediction = pd.Series(str(y_train.iloc[0]), index=validation.index, dtype="object")
            constant = True
        else:
            model = CatBoostBackend(label=attribute, **catboost_config(args))
            model.fit(
                fitting[table.feature_columns],
                y_train,
                validation[table.feature_columns],
                validation_labels[attribute],
            )
            prediction = model.predict(validation[table.feature_columns]).astype(str)
            prediction.index = validation.index
            constant = False
            del model
            gc.collect()
        metrics = classification_metrics(validation_labels[attribute], prediction.astype(str))
        dependencies = [
            normalized_mutual_info_score(
                train_labels[attribute], train_labels[other], average_method="arithmetic"
            )
            for other in attributes
            if other != attribute
        ]
        components = {
            "validation_balanced_accuracy": float(metrics["balanced_accuracy"]),
            "validation_macro_f1": float(metrics["macro_f1"]),
            "inverse_normalized_ordinal_mae": 1.0 - float(metrics["ordinal_mae"]),
            "mean_dependency_nmi": float(np.mean(dependencies)) if dependencies else 0.0,
        }
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in components.values()):
            raise ValueError(
                f"Invalid priority component for split {split.index}/{attribute}: {components}"
            )
        priority = float(sum(WEIGHTS[key] * value for key, value in components.items()))
        per_attribute[attribute] = {
            **components,
            "priority": priority,
            "constant": constant,
            "validation_metrics": metrics,
        }
        print(
            f"  [{position}/{len(attributes)}] {attribute}: priority={priority:.6f}"
            + (" (constant)" if constant else ""),
            flush=True,
        )
    forward = sorted(
        attributes,
        key=lambda name: (
            per_attribute[name]["constant"],
            -per_attribute[name]["priority"],
            attributes.index(name),
        ),
    )
    return {
        "split_index": split.index,
        "seen_activities": list(split.seen_activities),
        "unseen_activities": list(split.unseen_activities),
        "training_sample_count": len(fitting),
        "validation_sample_count": len(validation),
        "attribute_vectors": str(A1.resolve()),
        "attribute_vectors_sha256": sha256(A1),
        "attribute_schema": attributes,
        "forward": forward,
        "reverse": forward[::-1],
        "scores": per_attribute,
    }


def prepare_orders(
    args: argparse.Namespace, splits: Sequence[Any], table: Any, selected: Sequence[int]
) -> Path:
    selected_set = set(selected)
    chosen = [split for split in splits if split.index in selected_set]
    signature = preparation_signature(args, selected)
    signature_path = ORDER_WORK / "preparation_config.json"
    if signature_path.is_file() and read_json(signature_path) != signature:
        raise ValueError(
            f"Order-preparation settings differ from the cached protocol: {signature_path}. "
            "Use --rebuild-orders to replace the cached rankings."
        )
    base.write_json(signature_path, signature)
    entries: dict[str, Any] = {}
    for split in chosen:
        record = ORDER_WORK / f"split_{split.index:03d}" / "ranking.json"
        if record.is_file():
            saved = read_json(record)
            forward = list(map(str, saved.get("forward", [])))
            if (
                saved.get("attribute_vectors_sha256") != sha256(A1)
                or set(forward) != set(table.attribute_columns)
                or list(map(str, saved.get("reverse", []))) != forward[::-1]
            ):
                raise ValueError(f"Invalid cached CatBoost ranking: {record}")
            entries[str(split.index)] = saved
            print(f"Reusing A1 CatBoost order for split {split.index}: {record}", flush=True)
        else:
            result = rank_split(args, split=split, table=table)
            base.write_json(record, result)
            entries[str(split.index)] = result
    payload = {
        "schema_version": 1,
        "dataset": DATASET_KEY,
        "attribute_set": ATTRIBUTE_SET,
        "ontology_version": "1.5",
        "seed": 42,
        "selection_scope": SELECTION_SCOPE,
        "ranking_backend": "catboost",
        "ranking": WEIGHTS,
        "constant_attribute_policy": "place constants last in forward before exact reversal",
        "forward_definition": "descending weighted priority; ties follow the A1 schema order",
        "reverse_definition": "exact reversal of the complete forward order",
        "split_manifest": str(SPLITS.resolve()),
        "split_manifest_sha256": sha256(SPLITS),
        "attribute_vectors": str(A1.resolve()),
        "attribute_vectors_sha256": sha256(A1),
        "ontology": str(ONTOLOGY.resolve()),
        "ontology_sha256": sha256(ONTOLOGY),
        "preparation_config": signature,
        "generated_at": timestamp(),
        "splits": entries,
    }
    base.write_json(ORDER_FILE, payload)
    print(f"Saved/reused {len(entries)} A1 CatBoost orders: {ORDER_FILE}", flush=True)
    return ORDER_FILE


def validate_order_file(splits: Sequence[Any], indices: Sequence[int]) -> None:
    if not ORDER_FILE.is_file():
        raise FileNotFoundError(f"Missing A1 CatBoost order file: {ORDER_FILE}")
    payload = read_json(ORDER_FILE)
    entries = payload.get("splits", {})
    if (
        payload.get("dataset") != DATASET_KEY
        or payload.get("attribute_set") != ATTRIBUTE_SET
        or int(payload.get("seed", -1)) != 42
        or payload.get("ranking_backend") != "catboost"
        or payload.get("selection_scope") != SELECTION_SCOPE
        or payload.get("split_manifest_sha256") != sha256(SPLITS)
        or payload.get("attribute_vectors_sha256") != sha256(A1)
        or payload.get("ontology_sha256") != sha256(ONTOLOGY)
        or not set(indices).issubset(set(map(int, entries)))
    ):
        raise ValueError(f"Invalid or incomplete A1 CatBoost order file: {ORDER_FILE}")
    attributes = [str(item["name"]) for item in read_json(ONTOLOGY)["attributes"]]
    selected = set(indices)
    for split in splits:
        if split.index not in selected:
            continue
        entry = entries[str(split.index)]
        forward = list(map(str, entry.get("forward", [])))
        if (
            list(map(str, entry.get("seen_activities", []))) != list(split.seen_activities)
            or len(forward) != 9
            or len(set(forward)) != 9
            or set(forward) != set(attributes)
            or list(map(str, entry.get("reverse", []))) != forward[::-1]
        ):
            raise ValueError(f"Invalid A1 CatBoost order for split {split.index}")


def configure_base() -> None:
    base.DATASET = DATASET_KEY
    base.DATASET_LABEL = DATASET_LABEL
    base.SPLIT_MANIFEST = SPLITS
    base.SAMPLE_MANIFEST = SAMPLES
    base.SAMPLES_PER_ACTIVITY = EVAL_SAMPLES_PER_ACTIVITY
    base.CATBOOST_ORDERS = ORDER_FILE
    base.ATTRIBUTE_VECTORS_ROOT = None
    base.ATTRIBUTE_SET = ATTRIBUTE_SET
    base.ORDER_LABEL = "nine-dimensional A1 CatBoost seen-only"
    base.ORDER_SUMMARY_METADATA = {
        "order_method": "seen-only independent CatBoost validation ranking",
        "ranking_backend": "catboost",
        "selection_scope": SELECTION_SCOPE,
        "autogluon_used_for_order_generation": False,
    }
    base.RUN_NAME_PREFIX = RUN_NAME_PREFIX
    base.VARIANTS = VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "prepare", "train-test", "report"),
        default="train-test",
    )
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID)
    parser.add_argument("--run-root", type=Path, default=base.RUN_ROOT)
    parser.add_argument("--run-name-prefix", default=RUN_NAME_PREFIX)
    parser.add_argument("--split-indices", help="Optional comma-separated subset for a smoke test")
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--num-cpus", type=int, default=8, help="AutoGluon CPU allocation")
    parser.add_argument("--num-bag-folds", type=int, default=2)
    parser.add_argument("--order-num-cpus", type=int, default=112)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--rebuild-orders", action="store_true")
    parser.add_argument(
        "--reuse-orders-only",
        action="store_true",
        help=(
            "Require the existing 20-split A1 CatBoost order file and never fit "
            "or rewrite attribute rankings"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu_id < 0 or min(args.num_cpus, args.order_num_cpus, args.time_limit) <= 0:
        raise ValueError("GPU id must be nonnegative and CPU/time-limit values positive")
    if args.num_bag_folds < 2:
        raise ValueError("Soft OOF requires --num-bag-folds >= 2")
    if args.iterations <= 0 or args.learning_rate <= 0 or args.depth <= 0:
        raise ValueError("CatBoost ordering hyperparameters must be positive")
    if args.presets != "medium_quality":
        raise ValueError("This matched protocol requires --presets medium_quality")
    if args.reuse_orders_only and args.rebuild_orders:
        raise ValueError("--reuse-orders-only and --rebuild-orders are mutually exclusive")
    if args.reuse_orders_only and args.mode == "prepare":
        raise ValueError("prepare mode is incompatible with --reuse-orders-only")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    configure_base()
    spec, splits, table, all_indices = preflight()
    selected = selected_indices(args, all_indices)

    print(
        f"{DATASET_LABEL} nine-dimensional A1 + CatBoost-order + AutoGluon soft-OOF protocol\n"
        "  ordering: independent CatBoost models on seen-only data\n"
        f"  chain execution: {' -> '.join(VARIANTS)}\n"
        f"  physical GPU: {args.gpu_id}\n"
        f"  selected splits: {selected}\n"
        f"  A1 vectors: {A1}\n"
        f"  order file: {ORDER_FILE}",
        flush=True,
    )
    if args.rebuild_orders and ORDER_WORK.exists():
        import shutil

        shutil.rmtree(ORDER_WORK)
        if ORDER_FILE.exists():
            ORDER_FILE.unlink()
        print(f"Removed cached A1 CatBoost rankings: {ORDER_WORK}", flush=True)

    if args.mode in {"prepare", "train-test"} and not args.reuse_orders_only:
        prepare_orders(args, splits, table, selected)
    if args.mode == "prepare":
        return 0
    if args.mode == "check" and not ORDER_FILE.is_file() and not args.reuse_orders_only:
        print("Order file is not present yet; structural inputs are valid.", flush=True)
        return 0
    validate_order_file(splits, selected)

    configs = [base.experiment_config(args, variant, selected) for variant in VARIANTS]
    run_dirs = [base.validate_existing_run(config) for config in configs]
    print(
        "\n".join(
            f"  {variant} run: {run_dir}"
            for variant, run_dir in zip(VARIANTS, run_dirs)
        ),
        flush=True,
    )
    if args.mode == "check":
        return 0
    if args.mode == "report":
        base.summarize(args, configs, selected)
        return 0
    for run_dir in run_dirs:
        base.common.preserve_partial_models(run_dir, resume=args.resume, dry_run=args.dry_run)
    from zshar.pipeline.runner import run_experiment

    for position, config in enumerate(configs, 1):
        print(
            f"\nStage {position}/{len(configs)}: {config.attribute_order_variant} order",
            flush=True,
        )
        output = run_experiment(spec, config)
        print(f"Completed {config.attribute_order_variant}: {output}", flush=True)
    if not args.dry_run:
        base.summarize(args, configs, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
