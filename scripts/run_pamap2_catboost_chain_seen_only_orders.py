#!/usr/bin/env python3
"""Prepare CatBoost seen-only orders and run PAMAP2 CatBoost hard chains.

For every seed-42 PAMAP2 split this launcher loads the old pipeline's refined
A2 vectors, fits independent CatBoost attribute classifiers on seen-class data
only, and ranks the attributes from their held-out validation metrics.  The
reverse order is the exact reversal of the forward order.  It then trains and
evaluates ``catboost_classifier_chain`` in forward -> reverse order with the
same fixed split/sample manifests and the same split-specific A2 vectors.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
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

SPLITS = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_splits_seed42.json"
SAMPLES = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_eval50_samples_seed42.json"
A2_ROOT = Path(
    "/home/wanzhen/LLMbasedZeroshotHARAgent/pamap2_pipeline_runs/"
    "pamap2_random20_unseen4_eval50_no_rerank_keep_models_seed42"
)
PROTOCOL_ROOT = PROJECT / "runtime/pamap2_catboost_chain_seen_only_order_seed42"
ORDER_WORK = PROTOCOL_ROOT / "order_preparation"
ORDER_FILE = ORDER_WORK / "attribute_orders_catboost.json"
RUN_ROOT = PROJECT / "runtime/zshar_runs"
RUN_NAME_PREFIX = "pamap2_zsl_catboost_classifier_chain_a2_catboost_seen_only"
RUN_NAME_SUFFIX = "random20_eval50_seed42"
VARIANTS = ("forward", "reverse")
METRICS = (
    "activity_top1_accuracy",
    "activity_top3_recall",
    "activity_macro_f1",
    "attribute_exact_match",
)
WEIGHTS = {
    "validation_balanced_accuracy": 0.45,
    "validation_macro_f1": 0.30,
    "inverse_normalized_ordinal_mae": 0.20,
    "mean_dependency_nmi": 0.05,
}
MANAGER_MANIFEST = "attribute_backend_manifest.json"


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


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_vector(split_index: int) -> Path:
    candidates = (
        A2_ROOT / f"split_{split_index}/activity_attribute_vectors_a2.json",
        A2_ROOT / f"split_{split_index:03d}/activity_attribute_vectors_a2.json",
    )
    existing = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if not existing:
        raise FileNotFoundError(
            f"Missing old-pipeline A2 vectors for split {split_index}: "
            + " or ".join(map(str, candidates))
        )
    if len(existing) > 1 and len({file_hash(path) for path in existing}) != 1:
        raise ValueError(f"Conflicting padded/unpadded A2 vectors for split {split_index}")
    return existing[0]


def selected_indices(args: argparse.Namespace, all_indices: Sequence[int]) -> list[int]:
    if not args.split_indices:
        return list(all_indices)
    selected = sorted({int(value) for value in args.split_indices.split(",") if value.strip()})
    unknown = sorted(set(selected).difference(all_indices))
    if unknown:
        raise ValueError(f"Unknown --split-indices: {unknown}")
    return selected


def preflight() -> tuple[Any, list[Any], Any, list[int]]:
    from zshar.data import load_activity_vectors, load_dataset_table, replace_activity_vectors
    from zshar.datasets import load_dataset_spec
    from zshar.splits import (
        load_sample_manifest,
        load_split_manifest,
        sample_evaluation_rows,
    )

    for path in (SPLITS, SAMPLES, A2_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    split_payload = read_json(SPLITS)
    sample_payload = read_json(SAMPLES)
    if (
        split_payload.get("dataset") != "pamap2"
        or int(split_payload.get("seed", -1)) != 42
    ):
        raise ValueError(f"Expected PAMAP2 seed-42 split manifest: {SPLITS}")
    if (
        sample_payload.get("dataset") != "pamap2"
        or int(sample_payload.get("seed", -1)) != 42
        or int(sample_payload.get("samples_per_activity", -1)) != 50
    ):
        raise ValueError(f"Expected PAMAP2 seed-42 eval50 sample manifest: {SAMPLES}")

    spec = load_dataset_spec("pamap2")
    splits = load_split_manifest(SPLITS, spec)
    if len(splits) != 20 or len({split.index for split in splits}) != 20:
        raise ValueError("Expected exactly 20 distinct PAMAP2 splits")
    indices = [split.index for split in splits]
    raw_samples = sample_payload.get("samples", {})
    if set(map(int, raw_samples)) != set(indices):
        raise ValueError("Sample-manifest split indices differ from split manifest")
    samples = load_sample_manifest(SAMPLES)
    base_table = load_dataset_table(spec)

    for split in splits:
        vector_path = resolve_vector(split.index)
        vectors, attributes = load_activity_vectors(vector_path)
        if set(vectors) != set(spec.activity_names):
            raise ValueError(f"Split {split.index} A2 activities differ from PAMAP2")
        if not attributes or len(attributes) != len(set(attributes)):
            raise ValueError(f"Split {split.index} has an empty or duplicate A2 schema")
        table = replace_activity_vectors(base_table, vector_path)
        by_activity = raw_samples[str(split.index)]
        if set(map(str, by_activity)) != set(split.unseen_activities):
            raise ValueError(f"Split {split.index} eval activities differ from unseen classes")
        if len(by_activity) != 4 or any(
            len(ids) != 50 or len(set(map(int, ids))) != 50
            for ids in by_activity.values()
        ):
            raise ValueError(f"Split {split.index} must have 4 unseen classes x 50 samples")
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="zsl",
            samples_per_activity=50,
            seed=42,
            manifest_ids=samples[split.index],
        )
        if len(evaluation) != 200 or not evaluation.activity_name.value_counts().eq(50).all():
            raise ValueError(f"Split {split.index} does not resolve to 200 balanced samples")
    print(
        "Validated PAMAP2 protocol: old split-specific A2 vectors; 20 seed-42 "
        "splits; each test split has 4 unseen classes x 50 fixed samples.",
        flush=True,
    )
    return spec, splits, base_table, indices


def catboost_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "random_state": 42,
        "n_jobs": args.num_cpus,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "task_type": "GPU",
        # CUDA_VISIBLE_DEVICES maps the selected physical GPU to logical device 0.
        "devices": "0",
        "auto_class_weights": "Balanced",
        "allow_writing_files": False,
    }


def preparation_signature(args: argparse.Namespace, indices: Sequence[int]) -> dict[str, Any]:
    dataset_config = PROJECT / "configs/datasets/pamap2.json"
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
        "dataset": "pamap2",
        "seed": 42,
        "selection_scope": "seen-class data only",
        "validation_fraction": 0.3,
        "split_manifest_sha256": file_hash(SPLITS),
        "sample_manifest_sha256": file_hash(SAMPLES),
        "dataset_profile_sha256": file_hash(dataset_config),
        "feature_inputs": feature_inputs,
        "selected_split_indices": list(indices),
        "a2_sha256": {str(index): file_hash(resolve_vector(index)) for index in indices},
        "ranking_backend": "catboost",
        "ranking_backend_config": catboost_config(args),
        "ranking_weights": WEIGHTS,
    }


def rank_split(
    args: argparse.Namespace,
    *,
    split: Any,
    base_table: Any,
) -> dict[str, Any]:
    from zshar.data import replace_activity_vectors
    from zshar.models.base import classification_metrics
    from zshar.models.boosting_backends import CatBoostBackend
    from zshar.pipeline.training import split_train_validation

    vector_path = resolve_vector(split.index)
    table = replace_activity_vectors(base_table, vector_path)
    attributes = list(table.attribute_columns)
    seen = table.frame[table.frame.activity_name.isin(split.seen_activities)].copy()
    if set(seen.activity_name.unique()) != set(split.seen_activities):
        raise ValueError(f"Split {split.index} is missing seen-class training rows")
    fitting, validation = split_train_validation(
        seen, seed=42, validation_fraction=0.3
    )
    train_labels = fitting[attributes].astype(str)
    validation_labels = validation[attributes].astype(str)
    per_attribute: dict[str, Any] = {}
    print(
        f"Preparing CatBoost seen-only order for split {split.index}: "
        f"train={len(fitting)}, validation={len(validation)}, attributes={len(attributes)}",
        flush=True,
    )
    for position, attribute in enumerate(attributes, 1):
        y_train = train_labels[attribute]
        if y_train.nunique() == 1:
            prediction = pd.Series(
                str(y_train.iloc[0]), index=validation.index, dtype="object"
            )
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
        metrics = classification_metrics(
            validation_labels[attribute], prediction.astype(str)
        )
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
        key=lambda attribute: (
            per_attribute[attribute]["constant"],
            -per_attribute[attribute]["priority"],
            attributes.index(attribute),
        ),
    )
    return {
        "split_index": split.index,
        "seen_activities": list(split.seen_activities),
        "unseen_activities": list(split.unseen_activities),
        "training_sample_count": len(fitting),
        "validation_sample_count": len(validation),
        "attribute_vectors": str(vector_path.resolve()),
        "attribute_vectors_sha256": file_hash(vector_path),
        "attribute_schema": attributes,
        "forward": forward,
        "reverse": forward[::-1],
        "scores": per_attribute,
    }


def validate_cached_order(record: Path, split: Any) -> dict[str, Any]:
    from zshar.data import load_activity_vectors

    saved = read_json(record)
    vector_path = resolve_vector(split.index)
    _, attributes = load_activity_vectors(vector_path)
    forward = list(map(str, saved.get("forward", [])))
    reverse = list(map(str, saved.get("reverse", [])))
    if (
        int(saved.get("split_index", -1)) != split.index
        or saved.get("attribute_vectors_sha256") != file_hash(vector_path)
        or len(forward) != len(set(forward))
        or set(forward) != set(attributes)
        or reverse != forward[::-1]
    ):
        raise ValueError(f"Invalid or stale cached CatBoost order: {record}")
    return saved


def prepare_orders(
    args: argparse.Namespace,
    splits: Sequence[Any],
    base_table: Any,
    selected: Sequence[int],
) -> Path:
    selected_set = set(selected)
    chosen = [split for split in splits if split.index in selected_set]
    signature = preparation_signature(args, selected)
    signature_path = ORDER_WORK / "preparation_config.json"
    if signature_path.is_file() and read_json(signature_path) != signature:
        raise ValueError(
            f"Order-preparation inputs differ from the cached protocol: {signature_path}. "
            "Use a new --protocol-root or restore the original CatBoost settings."
        )
    write_json(signature_path, signature)
    entries: dict[str, Any] = {}
    for split in chosen:
        record = ORDER_WORK / f"split_{split.index:03d}" / "ranking.json"
        if record.is_file():
            saved = validate_cached_order(record, split)
            entries[str(split.index)] = saved
            print(f"Reusing CatBoost order for split {split.index}: {record}", flush=True)
            continue
        result = rank_split(args, split=split, base_table=base_table)
        write_json(record, result)
        entries[str(split.index)] = result

    payload = {
        "schema_version": 1,
        "dataset": "pamap2",
        "attribute_set": "old_pipeline_A2_refined_per_split",
        "seed": 42,
        "selection_scope": "seen-class training data only; no unseen samples, labels, or test metrics",
        "ranking_backend": "catboost",
        "ranking": WEIGHTS,
        "constant_attribute_policy": "place constants last in forward before exact reversal",
        "forward_definition": "descending weighted priority; ties follow each split A2 schema order",
        "reverse_definition": "exact reversal of the complete split-specific forward order",
        "split_manifest": str(SPLITS.resolve()),
        "split_manifest_sha256": file_hash(SPLITS),
        "a2_root": str(A2_ROOT.resolve()),
        "preparation_config": signature,
        "generated_at": timestamp(),
        "splits": entries,
    }
    write_json(ORDER_FILE, payload)
    print(f"Saved/reused {len(entries)} split-specific CatBoost orders: {ORDER_FILE}", flush=True)
    return ORDER_FILE


def experiment_config(args: argparse.Namespace, variant: str, selected: Sequence[int]):
    from zshar.pipeline.runner import ExperimentConfig

    backend_config = {
        **catboost_config(args),
        "consider_labels_correlation": True,
        "chain_feature_encoding": "ordinal",
    }
    return ExperimentConfig(
        dataset="pamap2",
        scenario="zsl",
        backend="catboost_classifier_chain",
        backend_config=backend_config,
        seed=42,
        split_manifest=str(SPLITS),
        sample_manifest=str(SAMPLES),
        attribute_vectors_root=str(A2_ROOT),
        attribute_order_config=str(ORDER_FILE),
        attribute_order_variant=variant,
        split_indices=set(selected),
        samples_per_activity=50,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        run_root=str(args.run_root.resolve()),
        run_name=f"{args.run_name_prefix}_{variant}_order_{RUN_NAME_SUFFIX}",
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
        if previous.get("split_indices") is not None:
            previous["split_indices"] = sorted(map(int, previous["split_indices"]))
        if current.get("split_indices") is not None:
            current["split_indices"] = sorted(map(int, current["split_indices"]))
        ignored = {"reuse_models", "evaluate_only", "dry_run"}
        if {key: value for key, value in previous.items() if key not in ignored} != {
            key: value for key, value in current.items() if key not in ignored
        }:
            raise ValueError(f"Existing run uses a different protocol: {saved_config}")
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
            + ", ".join(str(path) for path in partial)
        )
    if dry_run:
        for path in partial:
            print(f"Dry run: would preserve incomplete model directory {path}")
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


def summarize(args: argparse.Namespace, configs: Sequence[Any], selected: Sequence[int]) -> Path:
    expected = set(selected)
    frames: dict[str, pd.DataFrame] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for config in configs:
        variant = str(config.attribute_order_variant)
        path = Path(config.run_root) / "pamap2" / str(config.run_name) / "summary.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed summary: {path}")
        frame = pd.read_csv(path).set_index("split_index").sort_index()
        if frame.index.has_duplicates or set(map(int, frame.index)) != expected:
            raise ValueError(f"Expected exactly {len(expected)} completed splits: {path}")
        if set(frame["backend"].astype(str)) != {"catboost_classifier_chain"}:
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

    forward, reverse = frames["forward"], frames["reverse"]
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
            "attribute_set": "old_pipeline_A2_refined_per_split",
            "backend": "catboost_classifier_chain",
            "ranking_backend": "catboost",
            "selection_scope": "seen-only",
            "hard_labels_chained": True,
            "chain_feature_encoding": "ordinal",
            "split_count_per_order": len(expected),
            "order_sequence": list(VARIANTS),
            "attribute_order_config": str(ORDER_FILE.resolve()),
            "metrics": aggregate_rows,
        },
    )
    print("\n" + aggregate.to_string(index=False), flush=True)
    print(f"Forward/reverse comparison saved to: {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("check", "prepare", "train-test", "report"),
        default="train-test",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    # Match the earlier PAMAP2 CatBoost runs; this controls CatBoost's
    # thread_count while model fitting itself remains on the selected GPU.
    parser.add_argument("--num-cpus", type=int, default=112)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name-prefix", default=RUN_NAME_PREFIX)
    parser.add_argument("--split-indices", help="Optional comma-separated subset for a smoke test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global PROTOCOL_ROOT, ORDER_WORK, ORDER_FILE
    PROTOCOL_ROOT = args.protocol_root.expanduser().resolve()
    ORDER_WORK = PROTOCOL_ROOT / "order_preparation"
    ORDER_FILE = ORDER_WORK / "attribute_orders_catboost.json"


def main() -> int:
    args = parse_args()
    if args.gpu_id < 0 or args.num_cpus <= 0:
        raise ValueError("--gpu-id must be nonnegative and --num-cpus must be positive")
    if args.iterations <= 0 or args.learning_rate <= 0 or args.depth <= 0:
        raise ValueError("CatBoost hyperparameters must be positive")
    configure_paths(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    spec, splits, base_table, all_indices = preflight()
    selected = selected_indices(args, all_indices)
    print(
        "PAMAP2 CatBoost hard-label chain protocol\n"
        "  order estimation: independent CatBoost on seen-only train/validation data\n"
        "  chain execution: forward -> exact reverse\n"
        f"  physical GPU: {args.gpu_id} (CatBoost logical device 0)\n"
        f"  selected splits: {selected}\n"
        f"  old A2 root: {A2_ROOT}\n"
        f"  order file: {ORDER_FILE}",
        flush=True,
    )
    if args.mode == "check":
        return 0
    if args.mode in {"prepare", "train-test"}:
        prepare_orders(args, splits, base_table, selected)
    elif not ORDER_FILE.is_file():
        raise FileNotFoundError(f"Prepare CatBoost orders first: {ORDER_FILE}")
    if args.mode == "prepare":
        return 0

    configs = [experiment_config(args, variant, selected) for variant in VARIANTS]
    run_dirs = [validate_existing_run(config) for config in configs]
    if args.mode == "report":
        summarize(args, configs, selected)
        return 0
    for run_dir in run_dirs:
        preserve_partial_models(run_dir, resume=args.resume, dry_run=args.dry_run)
    print(
        f"  forward run: {run_dirs[0]}\n"
        f"  reverse run: {run_dirs[1]}",
        flush=True,
    )
    from zshar.pipeline.runner import run_experiment

    for position, config in enumerate(configs, 1):
        print(
            f"\nStage {position}/{len(configs)}: {config.attribute_order_variant} order",
            flush=True,
        )
        output = run_experiment(spec, config)
        print(f"Completed {config.attribute_order_variant}: {output}", flush=True)
    if not args.dry_run:
        summarize(args, configs, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
