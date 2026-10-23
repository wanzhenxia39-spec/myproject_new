#!/usr/bin/env python3
"""Run MHEALTH split-specific A2 CatBoost-order AutoGluon soft-OOF chains.

For every seed-42 ZSL split, this launcher validates the latest semantic-v3 A2
plan and vectors, fits independent CatBoost attribute classifiers using only
seen-class samples, and ranks the split-specific A2 attributes from held-out
validation metrics.  It then trains one forward AutoGluon soft-OOF probability
chain and evaluates the fixed 30-window-per-unseen-activity manifest.
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
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BASE_SCRIPT = PROJECT / "scripts/run_pamap2_autogluon_soft_oof_catboost_orders.py"
BASE_SPEC = importlib.util.spec_from_file_location("mhealth_a2_soft_oof_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(base)


DATASET = "mhealth"
DATASET_LABEL = "MHEALTH"
ATTRIBUTE_SET = "semantic_v3_refined_A2_per_split_11_15d"
SPLITS = PROJECT / "split/zero_shot_split/mhealth_unseen4_random20_splits_seed42.json"
SAMPLES = PROJECT / "split/zero_shot_split/mhealth_unseen4_random20_eval30_samples_seed42.json"
A2_ROOT = PROJECT / "runtime/refinement_semantic_v3/mhealth_gpt_5_6_sol_11_15"
PROTOCOL_ROOT = PROJECT / "runtime/mhealth_a2_semantic_v3_catboost_seen_only_order_seed42"
ORDER_WORK = PROTOCOL_ROOT / "order_preparation"
ORDER_FILE = ORDER_WORK / "attribute_orders_catboost.json"
RUN_ROOT = PROJECT / "runtime/zshar_runs"
RUN_NAME_PREFIX = "mhealth_zsl_autogluon_soft_oof_a2_semantic_v3_catboost_seen_only"
EVAL_SAMPLES_PER_ACTIVITY = 30
VARIANTS = ("forward",)
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


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def vector_path(split_index: int) -> Path:
    return A2_ROOT / f"split_{split_index}" / "activity_attribute_vectors_a2.json"


def plan_path(split_index: int) -> Path:
    return A2_ROOT / f"split_{split_index}" / "a2_attribute_plan.json"


def report_path(split_index: int) -> Path:
    return A2_ROOT / f"split_{split_index}" / "activity_attribute_vectors_a2.report.json"


def selected_indices(args: argparse.Namespace, available: Sequence[int]) -> list[int]:
    if not args.split_indices:
        return list(available)
    requested = [int(value.strip()) for value in args.split_indices.split(",") if value.strip()]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--split-indices must contain unique comma-separated split indices")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown --split-indices: {unknown}")
    wanted = set(requested)
    return [index for index in available if index in wanted]


def validate_a2_artifacts(split: Any, spec: Any) -> list[str]:
    from zshar.data import load_activity_vectors

    vectors_file = vector_path(split.index)
    plan_file = plan_path(split.index)
    report_file = report_path(split.index)
    for path in (vectors_file, plan_file, report_file):
        if not path.is_file():
            raise FileNotFoundError(
                f"split {split.index} is missing a required completed A2 artifact: {path}"
            )
    vectors, attributes = load_activity_vectors(vectors_file)
    plan = read_json(plan_file)
    report = read_json(report_file)
    plan_attributes = [str(item["name"]) for item in plan.get("attributes", [])]
    meta = plan.get("_meta", {})
    if (
        plan.get("dataset") != DATASET
        or int(plan.get("split_index", -1)) != split.index
        or meta.get("validation_decision") != "validated_semantic_a2"
    ):
        raise ValueError(f"split {split.index} does not contain a validated MHEALTH A2 plan")
    if not 11 <= len(attributes) <= 15 or len(attributes) != len(set(attributes)):
        raise ValueError(f"split {split.index} must contain 11-15 unique A2 attributes")
    if attributes != plan_attributes:
        raise ValueError(f"split {split.index} vector schema differs from its A2 plan")
    if set(vectors) != set(spec.activity_names):
        raise ValueError(f"split {split.index} A2 activities differ from MHEALTH")
    if (
        int(report.get("num_activities", -1)) != len(spec.activity_names)
        or int(report.get("num_attributes", -1)) != len(attributes)
    ):
        raise ValueError(f"split {split.index} A2 validation report has wrong dimensions")
    source = report.get("vector_source", {})
    if source.get("a2_plan_json") and Path(source["a2_plan_json"]).resolve() != plan_file.resolve():
        raise ValueError(f"split {split.index} report points to another A2 plan")
    return attributes


def preflight(selected: Sequence[int]) -> tuple[Any, list[Any], Any, list[int]]:
    from zshar.data import load_dataset_table, replace_activity_vectors
    from zshar.datasets import load_dataset_spec
    from zshar.splits import load_sample_manifest, load_split_manifest, sample_evaluation_rows

    for path in (SPLITS, SAMPLES, A2_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    split_payload = read_json(SPLITS)
    sample_payload = read_json(SAMPLES)
    if split_payload.get("dataset") != DATASET or int(split_payload.get("seed", -1)) != 42:
        raise ValueError(f"Expected MHEALTH seed-42 split manifest: {SPLITS}")
    if (
        sample_payload.get("dataset") != DATASET
        or int(sample_payload.get("seed", -1)) != 42
        or int(sample_payload.get("samples_per_activity", -1)) != EVAL_SAMPLES_PER_ACTIVITY
    ):
        raise ValueError(f"Expected MHEALTH seed-42 eval30 sample manifest: {SAMPLES}")

    spec = load_dataset_spec(DATASET)
    splits = load_split_manifest(SPLITS, spec)
    indices = [split.index for split in splits]
    if len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError("Expected exactly 20 distinct MHEALTH seed-42 splits")
    if set(map(int, sample_payload.get("samples", {}))) != set(indices):
        raise ValueError("Sample-manifest split indices differ from the split manifest")
    sample_manifest = load_sample_manifest(SAMPLES)
    base_table = load_dataset_table(spec)

    selected_set = set(selected)
    for split in splits:
        if split.index not in selected_set:
            continue
        validate_a2_artifacts(split, spec)
        table = replace_activity_vectors(base_table, vector_path(split.index))
        by_activity = sample_payload["samples"][str(split.index)]
        if set(map(str, by_activity)) != set(split.unseen_activities):
            raise ValueError(f"split {split.index} eval activities differ from unseen classes")
        if len(by_activity) != 4 or any(
            len(ids) != EVAL_SAMPLES_PER_ACTIVITY
            or len(set(map(int, ids))) != EVAL_SAMPLES_PER_ACTIVITY
            for ids in by_activity.values()
        ):
            raise ValueError(f"split {split.index} must have 4 unseen classes x 30 samples")
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="zsl",
            samples_per_activity=EVAL_SAMPLES_PER_ACTIVITY,
            seed=42,
            manifest_ids=sample_manifest[split.index],
        )
        if len(evaluation) != 120 or not evaluation.activity_name.value_counts().eq(30).all():
            raise ValueError(f"split {split.index} does not resolve to 120 balanced test rows")
    print(
        f"Validated MHEALTH A2 protocol for {len(selected_set)} selected split(s) "
        "from the 20-split semantic-v3 manifest; each test split has "
        "4 unseen classes x 30 fixed samples.",
        flush=True,
    )
    return spec, splits, base_table, indices


def catboost_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "random_state": 42,
        "n_jobs": args.order_num_cpus,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "task_type": "GPU",
        "devices": "0",
        "auto_class_weights": "Balanced",
        "allow_writing_files": False,
    }


def preparation_signature(args: argparse.Namespace) -> dict[str, Any]:
    dataset_config = PROJECT / "configs/datasets/mhealth.json"
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
        "dataset": DATASET,
        "seed": 42,
        "attribute_set": ATTRIBUTE_SET,
        "selection_scope": SELECTION_SCOPE,
        "validation_fraction": 0.3,
        "split_manifest_sha256": sha256(SPLITS),
        "sample_manifest_sha256": sha256(SAMPLES),
        "dataset_profile_sha256": sha256(dataset_config),
        "feature_inputs": feature_inputs,
        "ranking_backend": "catboost",
        "ranking_backend_config": catboost_config(args),
        "ranking_weights": WEIGHTS,
    }


def rank_split(args: argparse.Namespace, *, split: Any, base_table: Any) -> dict[str, Any]:
    from zshar.data import replace_activity_vectors
    from zshar.models.base import classification_metrics
    from zshar.models.boosting_backends import CatBoostBackend
    from zshar.pipeline.training import split_train_validation

    vectors_file = vector_path(split.index)
    table = replace_activity_vectors(base_table, vectors_file)
    attributes = list(table.attribute_columns)
    seen = table.frame[table.frame.activity_name.isin(split.seen_activities)].copy()
    if set(seen.activity_name.unique()) != set(split.seen_activities):
        raise ValueError(f"split {split.index} is missing seen-class training rows")
    fitting, validation = split_train_validation(seen, seed=42, validation_fraction=0.3)
    train_labels = fitting[attributes].astype(str)
    validation_labels = validation[attributes].astype(str)
    per_attribute: dict[str, Any] = {}
    print(
        f"Preparing A2 CatBoost seen-only order for split {split.index}: "
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
            + (" (constant; placed last)" if constant else ""),
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
        "attribute_vectors": str(vectors_file.resolve()),
        "attribute_vectors_sha256": sha256(vectors_file),
        "attribute_plan": str(plan_path(split.index).resolve()),
        "attribute_plan_sha256": sha256(plan_path(split.index)),
        "attribute_schema": attributes,
        "forward": forward,
        "reverse": forward[::-1],
        "scores": per_attribute,
    }


def valid_cached_order(path: Path, split: Any) -> tuple[bool, dict[str, Any] | None]:
    from zshar.data import load_activity_vectors

    try:
        saved = read_json(path)
        _, attributes = load_activity_vectors(vector_path(split.index))
        forward = list(map(str, saved.get("forward", [])))
        valid = (
            int(saved.get("split_index", -1)) == split.index
            and saved.get("attribute_vectors_sha256") == sha256(vector_path(split.index))
            and saved.get("attribute_plan_sha256") == sha256(plan_path(split.index))
            and len(forward) == len(set(forward))
            and set(forward) == set(attributes)
            and list(map(str, saved.get("reverse", []))) == forward[::-1]
        )
        return valid, saved if valid else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return False, None


def prepare_orders(
    args: argparse.Namespace, splits: Sequence[Any], base_table: Any, selected: Sequence[int]
) -> Path:
    if args.rebuild_orders and ORDER_WORK.exists():
        shutil.rmtree(ORDER_WORK)
        print(f"Removed cached A2 CatBoost rankings: {ORDER_WORK}", flush=True)
    signature = preparation_signature(args)
    signature_path = ORDER_WORK / "preparation_config.json"
    if signature_path.is_file() and read_json(signature_path) != signature:
        raise ValueError(
            f"CatBoost order settings differ from the cached protocol: {signature_path}. "
            "Use --rebuild-orders to replace the cached rankings."
        )
    write_json(signature_path, signature)

    selected_set = set(selected)
    entries: dict[str, Any] = {}
    for split in splits:
        record = ORDER_WORK / f"split_{split.index}" / "ranking.json"
        valid, saved = valid_cached_order(record, split) if record.is_file() else (False, None)
        if split.index in selected_set:
            if valid and saved is not None:
                entries[str(split.index)] = saved
                print(f"Reusing A2 CatBoost order for split {split.index}: {record}", flush=True)
            else:
                result = rank_split(args, split=split, base_table=base_table)
                write_json(record, result)
                entries[str(split.index)] = result
        elif valid and saved is not None:
            entries[str(split.index)] = saved

    previous_generated_at = None
    if ORDER_FILE.is_file():
        try:
            previous_generated_at = read_json(ORDER_FILE).get("generated_at")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            previous_generated_at = None
    payload = {
        "schema_version": 1,
        "dataset": DATASET,
        "attribute_set": ATTRIBUTE_SET,
        "seed": 42,
        "selection_scope": SELECTION_SCOPE,
        "ranking_backend": "catboost",
        "ranking": WEIGHTS,
        "constant_attribute_policy": "place constants last in forward order",
        "forward_definition": (
            "descending weighted seen-only CatBoost priority; ties follow each split A2 schema"
        ),
        "reverse_definition": "stored for audit only; this launcher executes forward only",
        "split_manifest": str(SPLITS.resolve()),
        "split_manifest_sha256": sha256(SPLITS),
        "a2_root": str(A2_ROOT.resolve()),
        "preparation_config": signature,
        # Keep the order-file hash stable across --resume runs when all cached
        # per-split rankings are unchanged.
        "generated_at": previous_generated_at or timestamp(),
        "splits": entries,
    }
    write_json(ORDER_FILE, payload)
    print(f"Saved/reused {len(entries)} split-specific A2 CatBoost orders: {ORDER_FILE}")
    return ORDER_FILE


def validate_order_file(splits: Sequence[Any], selected: Sequence[int]) -> None:
    if not ORDER_FILE.is_file():
        raise FileNotFoundError(f"Missing A2 CatBoost order file: {ORDER_FILE}")
    payload = read_json(ORDER_FILE)
    if (
        payload.get("dataset") != DATASET
        or payload.get("attribute_set") != ATTRIBUTE_SET
        or int(payload.get("seed", -1)) != 42
        or payload.get("ranking_backend") != "catboost"
        or payload.get("selection_scope") != SELECTION_SCOPE
        or payload.get("split_manifest_sha256") != sha256(SPLITS)
    ):
        raise ValueError(f"Invalid A2 CatBoost order metadata: {ORDER_FILE}")
    selected_set = set(selected)
    entries = payload.get("splits", {})
    for split in splits:
        if split.index not in selected_set:
            continue
        entry = entries.get(str(split.index))
        if not isinstance(entry, Mapping):
            raise ValueError(f"Missing CatBoost order for selected split {split.index}")
        record = ORDER_WORK / f"split_{split.index}" / "ranking.json"
        valid, _ = valid_cached_order(record, split)
        if not valid or entry != read_json(record):
            raise ValueError(f"Invalid or stale CatBoost order for split {split.index}")


def configure_soft_oof_base() -> None:
    base.DATASET = DATASET
    base.DATASET_LABEL = DATASET_LABEL
    base.SPLIT_MANIFEST = SPLITS
    base.SAMPLE_MANIFEST = SAMPLES
    base.SAMPLES_PER_ACTIVITY = EVAL_SAMPLES_PER_ACTIVITY
    base.CATBOOST_ORDERS = ORDER_FILE
    base.ATTRIBUTE_VECTORS_ROOT = A2_ROOT
    base.ATTRIBUTE_SET = ATTRIBUTE_SET
    base.ORDER_LABEL = "split-specific A2 CatBoost seen-only"
    base.ORDER_SUMMARY_METADATA = {
        "order_method": "seen-only independent CatBoost validation ranking",
        "ranking_backend": "catboost",
        "selection_scope": SELECTION_SCOPE,
        "autogluon_used_for_order_generation": False,
    }
    base.RUN_NAME_PREFIX = RUN_NAME_PREFIX
    base.VARIANTS = VARIANTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "prepare", "train-test", "report"),
        default="train-test",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name-prefix", default=RUN_NAME_PREFIX)
    parser.add_argument("--split-indices", help="Optional comma-separated split subset")
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--num-cpus", type=int, default=8)
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
            "Require the existing 20-split CatBoost order file and never fit or "
            "rewrite attribute rankings"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_id < 0 or min(args.num_cpus, args.order_num_cpus, args.time_limit) <= 0:
        raise ValueError("GPU id must be nonnegative and CPU/time-limit values positive")
    if args.num_bag_folds < 2:
        raise ValueError("Soft OOF requires --num-bag-folds >= 2")
    if args.presets != "medium_quality":
        raise ValueError("This matched protocol requires --presets medium_quality")
    if args.reuse_orders_only and args.rebuild_orders:
        raise ValueError("--reuse-orders-only and --rebuild-orders are mutually exclusive")
    if args.reuse_orders_only and args.mode == "prepare":
        raise ValueError("prepare mode is incompatible with --reuse-orders-only")
    if args.iterations <= 0 or args.learning_rate <= 0 or args.depth <= 0:
        raise ValueError("CatBoost ordering hyperparameters must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    configure_soft_oof_base()
    split_payload = read_json(SPLITS)
    declared_splits = split_payload.get("splits", [])
    all_indices = [int(item["split_index"]) for item in declared_splits]
    if len(all_indices) != 20 or len(set(all_indices)) != 20:
        raise ValueError("Expected exactly 20 distinct MHEALTH seed-42 splits")
    selected = selected_indices(args, all_indices)
    spec, splits, base_table, verified_indices = preflight(selected)
    if verified_indices != all_indices:
        raise ValueError("Split manifest changed during protocol validation")
    print(
        "MHEALTH semantic-v3 A2 + CatBoost-order + AutoGluon soft-OOF protocol\n"
        "  ordering: independent CatBoost models on seen-only data\n"
        "  chain execution: forward only\n"
        f"  physical GPU: {args.gpu_id} (logical device 0)\n"
        f"  selected splits: {selected}\n"
        f"  A2 root: {A2_ROOT}\n"
        f"  order file: {ORDER_FILE}",
        flush=True,
    )
    if args.mode == "check":
        if args.reuse_orders_only:
            validate_order_file(splits, selected)
        return 0
    if args.mode in {"prepare", "train-test"} and not args.reuse_orders_only:
        prepare_orders(args, splits, base_table, selected)
    if args.mode == "prepare":
        return 0
    validate_order_file(splits, selected)

    configs = [base.experiment_config(args, "forward", selected)]
    run_dirs = [base.validate_existing_run(config) for config in configs]
    if args.mode == "report":
        base.summarize(args, configs, selected)
        return 0
    for run_dir in run_dirs:
        base.common.preserve_partial_models(
            run_dir, resume=args.resume, dry_run=args.dry_run
        )
    from zshar.pipeline.runner import run_experiment

    output = run_experiment(spec, configs[0])
    print(f"Completed forward order: {output}", flush=True)
    if not args.dry_run:
        base.summarize(args, configs, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
