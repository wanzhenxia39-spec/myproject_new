#!/usr/bin/env python3
"""Run PAMAP2 and/or MHEALTH cross-subject AutoGluon soft-OOF chains.

Each dataset has 20 split-specific semantic-v3 A2 vector schemas.  Attribute
orders can either reuse the saved CatBoost-derived forward order or use a
deterministic random permutation for each semantic schema.  Cross-subject
training/test indices are held fixed while all 20 semantic schemas are
evaluated independently.  The backend called ``autogluon_oof_chain`` in the
experiment name is implemented by the repository's ``autogluon_soft_oof``
backend: native bagging OOF probabilities are chained during training and
predicted soft probabilities at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs")

DATASETS: dict[str, dict[str, Path]] = {
    "pamap2": {
        "vectors": PROJECT / "runtime/refinement_semantic_v3/pamap2_gpt_5_6_sol_11_15",
        "orders": PROJECT
        / "runtime/pamap2_a2_semantic_v3_catboost_seen_only_order_seed42"
        / "order_preparation/attribute_orders_catboost.json",
        "subjects": PROJECT / "split/subject_80_20/pamap2_subject80_20_balanced_seed42.json",
    },
    "mhealth": {
        "vectors": PROJECT / "runtime/refinement_semantic_v3/mhealth_gpt_5_6_sol_11_15",
        "orders": PROJECT
        / "runtime/mhealth_a2_semantic_v3_catboost_seen_only_order_seed42"
        / "order_preparation/attribute_orders_catboost.json",
        "subjects": PROJECT / "split/subject_80_20/mhealth_subject80_20_balanced_seed42.json",
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray, tuple, set)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_path(root: Path, split_index: int) -> Path:
    candidates = (
        root / f"split_{split_index}" / "activity_attribute_vectors_a2.json",
        root / f"split_{split_index:03d}" / "activity_attribute_vectors_a2.json",
    )
    existing = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if not existing:
        raise FileNotFoundError(f"No refined A2 vector found for split {split_index}: {candidates}")
    if len(existing) > 1 and len({sha256(path) for path in existing}) != 1:
        raise ValueError(f"Conflicting refined vectors for split {split_index}: {existing}")
    return existing[0]


def discover_semantic_split_indices(root: Path) -> list[int]:
    """Return the split indices backed by complete A2 vector files."""
    indices: set[int] = set()
    for path in root.glob("split_*/activity_attribute_vectors_a2.json"):
        suffix = path.parent.name.removeprefix("split_")
        if suffix.isdigit():
            indices.add(int(suffix))
    discovered = sorted(indices)
    if len(discovered) != 20:
        raise ValueError(
            f"Expected exactly 20 refined A2 semantic schemas under {root}, "
            f"found {len(discovered)}: {discovered}"
        )
    return discovered


def deterministic_random_order(
    attributes: Sequence[str], *, dataset: str, split_index: int, seed: int
) -> tuple[list[str], int]:
    """Shuffle one schema reproducibly without consulting sensor/test data."""
    material = f"{seed}:{dataset}:{split_index}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    order = list(map(str, attributes))
    random.Random(derived_seed).shuffle(order)
    return order, derived_seed


def backend_config(args: argparse.Namespace, order: Sequence[str]) -> dict[str, Any]:
    return {
        "time_limit": args.time_limit,
        "presets": "medium_quality",
        "num_gpus": 1,
        "num_cpus": args.num_cpus,
        "num_bag_folds": args.num_bag_folds,
        "num_stack_levels": 0,
        "fit_weighted_ensemble": True,
        "model_suite": "default",
        "consider_labels_correlation": True,
        "uncertainty_metric": "normalized_entropy",
        "lightgbm_device": "cuda",
        "gpu_model_types": ["NN_TORCH", "FASTAI", "CAT", "XGB"],
        "verbosity": 2,
        "label_order": list(order),
    }


def validate_dataset(
    dataset: str, args: argparse.Namespace
) -> tuple[Any, Any, dict[str, Any] | None, list[int]]:
    from zshar.data import load_activity_vectors, load_dataset_table
    from zshar.datasets import load_dataset_spec
    from zshar.splits import load_subject_manifest

    paths = DATASETS[dataset]
    required = [paths["vectors"], paths["subjects"]]
    if args.attribute_order == "catboost":
        required.append(paths["orders"])
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    spec = load_dataset_spec(dataset)
    subject_split = load_subject_manifest(paths["subjects"], spec)
    split_indices = discover_semantic_split_indices(paths["vectors"])
    order_payload: dict[str, Any] | None = None
    entries: dict[str, Any] | None = None
    if args.attribute_order == "catboost":
        order_payload = read_json(paths["orders"])
        if (
            str(order_payload.get("dataset")) != dataset
            or int(order_payload.get("seed", -1)) != 42
        ):
            raise ValueError(f"Order file provenance mismatch for {dataset}")
        entries = order_payload.get("splits")
        if not isinstance(entries, dict) or len(entries) != 20:
            raise ValueError(f"Expected exactly 20 saved order entries for {dataset}")
        if set(map(int, entries)) != set(split_indices):
            raise ValueError(f"CatBoost order/vector split mismatch for {dataset}")

    expected_activities = set(spec.activity_names)
    for split_index in split_indices:
        path = vector_path(paths["vectors"], split_index)
        vectors, attributes = load_activity_vectors(path)
        if set(vectors) != expected_activities:
            raise ValueError(f"Refined-vector activity mismatch for {dataset}/split {split_index}")
        if len(attributes) != len(set(attributes)):
            raise ValueError(f"Duplicate A2 attributes for {dataset}/split {split_index}")
        if entries is not None:
            entry = entries[str(split_index)]
            order = list(map(str, entry.get("forward", [])))
            if len(order) != len(set(order)) or set(order) != set(attributes):
                raise ValueError(f"Invalid forward order for {dataset}/split {split_index}")
            if entry.get("attribute_vectors_sha256") != sha256(path):
                raise ValueError(f"Refined-vector hash mismatch for {dataset}/split {split_index}")

    table = load_dataset_table(spec)
    available = set(table.frame["window_index"].astype(int))
    requested = set(subject_split.train_indices) | set(subject_split.test_indices)
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            f"Subject manifest for {dataset} references unavailable windows: {missing[:10]}"
        )
    if set(subject_split.train_subjects) & set(subject_split.test_subjects):
        raise AssertionError(f"Cross-subject leakage in {dataset} subject manifest")
    return spec, subject_split, order_payload, split_indices


def aggregate(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows).sort_values("semantic_split_index")
    frame.to_csv(run_dir / "summary.csv", index=False)
    names = (
        "activity_top1_accuracy",
        "activity_top3_recall",
        "activity_macro_f1",
        "attribute_exact_match",
    )
    payload: dict[str, Any] = {
        "semantic_split_count": len(frame),
        "standard_deviation": "sample (ddof=1; 0.0 when only one split is present)",
        "metrics": {},
    }
    csv_row: dict[str, Any] = {"semantic_split_count": len(frame)}
    for name in names:
        values = frame[name].to_numpy(dtype=float)
        stats = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        payload["metrics"][name] = stats
        for statistic, value in stats.items():
            if statistic != "count":
                csv_row[f"{name}_{statistic}"] = value
    write_json(run_dir / "aggregate_metrics.json", payload)
    pd.DataFrame([csv_row]).to_csv(run_dir / "aggregate_metrics.csv", index=False)


def run_dataset(dataset: str, args: argparse.Namespace) -> Path:
    from zshar.data import load_dataset_table, replace_activity_vectors
    from zshar.evaluation.cross_subject import evaluate_cross_subject
    from zshar.pipeline.training import load_or_train_attribute_models

    spec, subject_split, order_payload, all_indices = validate_dataset(dataset, args)
    selected = (
        all_indices
        if args.semantic_split_indices is None
        else [index for index in all_indices if index in args.semantic_split_indices]
    )
    if not selected:
        raise ValueError(f"No semantic splits selected for {dataset}")

    paths = DATASETS[dataset]
    order_tag = (
        "catboost_order"
        if args.attribute_order == "catboost"
        else f"random_order{args.random_order_seed}"
    )
    run_name = (
        f"{dataset}_cross_subject_autogluon_oof_chain_a2_semantic_v3_"
        f"{order_tag}_bag{args.num_bag_folds}_seed42"
    )
    run_dir = args.run_root.resolve() / dataset / run_name
    root_config = {
        "dataset": dataset,
        "scenario": "cross_subject",
        "requested_backend_name": "autogluon_oof_chain",
        "implementation_backend": "autogluon_soft_oof",
        "seed": 42,
        "physical_gpu": args.gpu_id,
        "logical_gpu": 0,
        "subject_manifest": str(paths["subjects"].resolve()),
        "subject_manifest_sha256": sha256(paths["subjects"]),
        "attribute_vectors_root": str(paths["vectors"].resolve()),
        "attribute_order_strategy": args.attribute_order,
        "attribute_order_config": (
            str(paths["orders"].resolve()) if args.attribute_order == "catboost" else None
        ),
        "attribute_order_config_sha256": (
            sha256(paths["orders"]) if args.attribute_order == "catboost" else None
        ),
        "attribute_order_variant": (
            "forward" if args.attribute_order == "catboost" else "deterministic_random"
        ),
        "random_order_seed": (
            args.random_order_seed if args.attribute_order == "random" else None
        ),
        "semantic_split_indices": selected,
        "time_limit": args.time_limit,
        "num_cpus": args.num_cpus,
        "num_bag_folds": args.num_bag_folds,
        "prototype_metric": "ordinal_probability",
        "attribute_weighting": "validation_accuracy",
    }
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists; use --resume: {run_dir}")
    if (run_dir / "config.json").exists():
        if read_json(run_dir / "config.json") != root_config:
            raise ValueError(f"Existing run configuration differs: {run_dir / 'config.json'}")
    else:
        write_json(run_dir / "config.json", root_config)

    base_table = load_dataset_table(spec)
    train_ids = set(subject_split.train_indices)
    test_ids = set(subject_split.test_indices)
    rows: list[dict[str, Any]] = []
    for position, split_index in enumerate(selected, 1):
        split_dir = run_dir / f"semantic_split_{split_index:03d}"
        result_path = split_dir / "result_row.json"
        if args.resume and result_path.is_file():
            row = read_json(result_path)
            rows.append(row)
            print(f"[{dataset} {position}/{len(selected)}] split {split_index}: already complete")
            continue

        refined_path = vector_path(paths["vectors"], split_index)
        table = replace_activity_vectors(base_table, refined_path)
        if args.attribute_order == "catboost":
            if order_payload is None:
                raise AssertionError("Validated CatBoost order payload is missing")
            order = list(map(str, order_payload["splits"][str(split_index)]["forward"]))
            derived_random_seed = None
        else:
            order, derived_random_seed = deterministic_random_order(
                table.attribute_columns,
                dataset=dataset,
                split_index=split_index,
                seed=args.random_order_seed,
            )
        training = table.frame[table.frame["window_index"].isin(train_ids)].copy()
        evaluation = table.frame[table.frame["window_index"].isin(test_ids)].copy()
        observed_train = set(training["user_id"].astype(int))
        observed_test = set(evaluation["user_id"].astype(int))
        if observed_train != set(subject_split.train_subjects):
            raise ValueError(f"Training subjects mismatch for {dataset}/split {split_index}")
        if observed_test != set(subject_split.test_subjects):
            raise ValueError(f"Test subjects mismatch for {dataset}/split {split_index}")
        if observed_train & observed_test:
            raise AssertionError(f"Subject leakage for {dataset}/split {split_index}")

        split_config = {
            "semantic_split_index": split_index,
            "attribute_vectors": str(refined_path.resolve()),
            "attribute_vectors_sha256": sha256(refined_path),
            "attribute_count": len(table.attribute_columns),
            "attribute_order": order,
            "attribute_order_strategy": args.attribute_order,
            "random_order_seed": (
                args.random_order_seed if args.attribute_order == "random" else None
            ),
            "derived_random_order_seed": derived_random_seed,
            "train_subjects": list(subject_split.train_subjects),
            "test_subjects": list(subject_split.test_subjects),
            "train_samples": len(training),
            "eval_samples": len(evaluation),
            "backend_config": backend_config(args, order),
        }
        protocol_path = split_dir / "protocol.json"
        if protocol_path.exists() and read_json(protocol_path) != split_config:
            raise ValueError(f"Existing split protocol differs: {protocol_path}")
        write_json(protocol_path, split_config)
        write_json(split_dir / "subject_split.json", asdict(subject_split))

        print(
            f"[{dataset} {position}/{len(selected)}] semantic split {split_index}: "
            f"train={len(training)}, test={len(evaluation)}, "
            f"attributes={len(table.attribute_columns)}",
            flush=True,
        )
        manager, validation_report = load_or_train_attribute_models(
            reuse_models=args.resume,
            evaluate_only=False,
            model_path=split_dir / "models",
            frame=training,
            feature_columns=table.feature_columns,
            attribute_columns=table.attribute_columns,
            backend="autogluon_soft_oof",
            backend_config=backend_config(args, order),
            seed=42,
        )
        metrics, details = evaluate_cross_subject(
            manager=manager,
            frame=evaluation,
            feature_columns=table.feature_columns,
            attribute_columns=table.attribute_columns,
            activity_vectors=table.activity_vectors,
            candidate_activities=spec.activity_names,
            metric="ordinal_probability",
            attribute_weighting="validation_accuracy",
            validation_report=validation_report,
        )
        row = {
            "dataset": dataset,
            "scenario": "cross_subject",
            "backend": "autogluon_soft_oof",
            "semantic_split_index": split_index,
            "train_subjects": "|".join(map(str, subject_split.train_subjects)),
            "test_subjects": "|".join(map(str, subject_split.test_subjects)),
            "train_samples": len(training),
            "eval_samples": len(evaluation),
            "attribute_count": len(table.attribute_columns),
            "activity_accuracy": metrics["activity"]["accuracy"],
            "activity_top1_accuracy": metrics["activity"]["top1_accuracy"],
            "activity_top3_recall": metrics["activity"]["top3_recall"],
            "activity_macro_f1": metrics["activity"]["macro_f1"],
            "attribute_exact_match": metrics["attribute_exact_match"],
        }
        details.to_csv(split_dir / "details.csv", index=False)
        write_json(split_dir / "metrics.json", metrics)
        write_json(split_dir / "validation_report.json", validation_report)
        write_json(result_path, row)
        rows.append(row)
        aggregate(run_dir, rows)

    aggregate(run_dir, rows)
    print(f"[{dataset}] completed {len(rows)} semantic schemas: {run_dir}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("check", "train-test"), default="train-test")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--num-bag-folds", type=int, default=2)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument(
        "--dataset",
        choices=("all", "pamap2", "mhealth"),
        default="all",
        help="Run both datasets or only the selected dataset.",
    )
    parser.add_argument(
        "--attribute-order",
        choices=("catboost", "random"),
        default="catboost",
        help="Reuse cached CatBoost forward orders or deterministically shuffle each A2 schema.",
    )
    parser.add_argument(
        "--random-order-seed",
        type=int,
        default=42,
        help="Base seed used only when --attribute-order=random.",
    )
    parser.add_argument(
        "--semantic-split-indices",
        help="Optional comma-separated subset used for debugging or partial reruns",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_id != 0:
        raise ValueError("This launcher is intentionally pinned to physical GPU 0")
    if min(args.num_cpus, args.time_limit) <= 0 or args.num_bag_folds < 2:
        raise ValueError("Positive CPU/time values and at least two OOF folds are required")
    args.semantic_split_indices = (
        None
        if not args.semantic_split_indices
        else {int(item.strip()) for item in args.semantic_split_indices.split(",") if item.strip()}
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    selected_datasets = (
        ("pamap2", "mhealth") if args.dataset == "all" else (args.dataset,)
    )
    for dataset in selected_datasets:
        _, subject_split, _, indices = validate_dataset(dataset, args)
        print(
            f"Validated {dataset}: semantic_schemas={len(indices)}, "
            f"train_subjects={list(subject_split.train_subjects)}, "
            f"test_subjects={list(subject_split.test_subjects)}, "
            f"attribute_order={args.attribute_order}, "
            f"physical_gpu={args.gpu_id}, logical_gpu=0",
            flush=True,
        )
    if args.mode == "check":
        print("Check complete; no model was trained and no result directory was written.")
        return 0

    for dataset in selected_datasets:
        run_dataset(dataset, args)
    print(f"Cross-subject queue completed: {', '.join(selected_datasets)}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
