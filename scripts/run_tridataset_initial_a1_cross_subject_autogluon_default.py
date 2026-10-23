#!/usr/bin/env python3
"""Run initial-A1-v1.5 cross-subject tests with AutoGluon default ensembles.

PAMAP2, MHEALTH and WISDM use the fixed subject-80/20 manifests and the
dataset-specific, pre-refinement nine-attribute vectors under
``configs/attributes``.  Every attribute is classified independently by an
AutoGluon default ensemble with two bagging folds; there is no attribute-order
or classifier-chain dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs")
MANAGER_MANIFEST = "attribute_backend_manifest.json"

DATASETS: dict[str, dict[str, Path]] = {
    "pamap2": {
        "vectors": PROJECT
        / "configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5.json",
        "subjects": PROJECT
        / "split/subject_80_20/pamap2_subject80_20_balanced_seed42.json",
    },
    "mhealth": {
        "vectors": PROJECT
        / "configs/attributes/activity_attribute_vectors_initial_mhealth_v1_5.json",
        "subjects": PROJECT
        / "split/subject_80_20/mhealth_subject80_20_balanced_seed42.json",
    },
    "wisdm": {
        "vectors": PROJECT
        / "configs/attributes/activity_attribute_vectors_initial_wisdm_v1_5.json",
        "subjects": PROJECT
        / "split/subject_80_20/wisdm_subject80_20_balanced_seed42.json",
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
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


def backend_config(args: argparse.Namespace) -> dict[str, Any]:
    """Match the refined-A2 independent AutoGluon default configuration."""
    return {
        "time_limit": args.time_limit,
        "presets": "medium_quality",
        "num_gpus": 1,
        "num_cpus": args.num_cpus,
        "num_bag_folds": args.num_bag_folds,
        "num_stack_levels": 0,
        "fit_weighted_ensemble": True,
        "model_suite": "default",
        "lightgbm_device": "cuda",
        "gpu_model_types": ["NN_TORCH", "FASTAI", "CAT", "XGB"],
        "verbosity": 2,
    }


def validate_dataset(dataset: str) -> tuple[Any, Any, Any]:
    from zshar.data import load_activity_vectors, load_dataset_table
    from zshar.datasets import load_dataset_spec
    from zshar.splits import load_subject_manifest

    paths = DATASETS[dataset]
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    spec = load_dataset_spec(dataset)
    if spec.attributes_path.resolve() != paths["vectors"].resolve():
        raise ValueError(
            "Dataset profile does not point to the required initial A1 v1.5 vectors: "
            f"{spec.attributes_path} != {paths['vectors']}"
        )

    vectors, attributes = load_activity_vectors(paths["vectors"])
    ontology = read_json(PROJECT / "configs/attributes/initial_attributes.json")
    expected_attributes = [str(item["name"]) for item in ontology["attributes"]]
    if set(vectors) != set(spec.activity_names):
        raise ValueError(f"Initial-vector activity mismatch for {dataset}")
    if attributes != expected_attributes or len(attributes) != 9:
        raise ValueError(
            f"Expected the ordered nine-attribute initial A1 v1.5 schema for {dataset}"
        )

    table = load_dataset_table(spec)
    subject_split = load_subject_manifest(paths["subjects"], spec)
    available = set(table.frame["window_index"].astype(int))
    requested = set(subject_split.train_indices) | set(subject_split.test_indices)
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            f"Subject manifest for {dataset} references unavailable windows: {missing[:10]}"
        )
    if set(subject_split.train_subjects) & set(subject_split.test_subjects):
        raise AssertionError(f"Cross-subject leakage in {dataset} subject manifest")
    return spec, subject_split, table


def preserve_incomplete_models(model_path: Path, *, resume: bool) -> None:
    """Move an interrupted model directory aside before a safe full retry."""
    if not model_path.exists() or (model_path / MANAGER_MANIFEST).is_file():
        return
    if not resume:
        raise FileExistsError(f"Incomplete model directory exists; use --resume: {model_path}")
    destination = model_path.with_name("models.incomplete")
    counter = 1
    while destination.exists():
        destination = model_path.with_name(f"models.incomplete_{counter}")
        counter += 1
    model_path.rename(destination)
    print(f"Preserved incomplete model directory: {destination}", flush=True)


def run_dataset(dataset: str, args: argparse.Namespace) -> Path:
    from zshar.evaluation.cross_subject import evaluate_cross_subject
    from zshar.pipeline.training import load_or_train_attribute_models

    spec, subject_split, table = validate_dataset(dataset)
    paths = DATASETS[dataset]
    run_name = (
        f"{dataset}_cross_subject_autogluon_default_ensemble_initial_a1_v1_5_"
        f"independent_bag{args.num_bag_folds}_seed42"
    )
    run_dir = args.run_root.expanduser().resolve() / dataset / run_name
    config = {
        "dataset": dataset,
        "scenario": "cross_subject",
        "backend": "autogluon",
        "model_suite": "default",
        "classifier_structure": "independent_per_attribute",
        "attribute_set": "initial_a1_v1_5_pre_refinement",
        "attribute_vectors": str(paths["vectors"].resolve()),
        "attribute_vectors_sha256": sha256(paths["vectors"]),
        "attribute_count": len(table.attribute_columns),
        "attribute_order_strategy": "none_independent_backend",
        "seed": 42,
        "physical_gpu": args.gpu_id,
        "logical_gpu": 0,
        "subject_manifest": str(paths["subjects"].resolve()),
        "subject_manifest_sha256": sha256(paths["subjects"]),
        "train_subjects": list(subject_split.train_subjects),
        "test_subjects": list(subject_split.test_subjects),
        "train_samples": len(subject_split.train_indices),
        "eval_samples": len(subject_split.test_indices),
        "prototype_metric": "ordinal_probability",
        "attribute_weighting": "validation_accuracy",
        "backend_config": backend_config(args),
    }
    config_path = run_dir / "config.json"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists; use --resume: {run_dir}")
    if config_path.is_file() and read_json(config_path) != config:
        raise ValueError(f"Existing run configuration differs: {config_path}")
    write_json(config_path, config)

    result_path = run_dir / "result.json"
    if args.resume and result_path.is_file():
        print(f"[{dataset}] already complete: {result_path}", flush=True)
        return run_dir

    train_ids = set(subject_split.train_indices)
    test_ids = set(subject_split.test_indices)
    training = table.frame[table.frame["window_index"].isin(train_ids)].copy()
    evaluation = table.frame[table.frame["window_index"].isin(test_ids)].copy()
    observed_train = set(training["user_id"].astype(int))
    observed_test = set(evaluation["user_id"].astype(int))
    if observed_train != set(subject_split.train_subjects):
        raise ValueError(f"Training subjects mismatch for {dataset}")
    if observed_test != set(subject_split.test_subjects):
        raise ValueError(f"Test subjects mismatch for {dataset}")
    if observed_train & observed_test:
        raise AssertionError(f"Cross-subject leakage for {dataset}")

    model_path = run_dir / "models"
    preserve_incomplete_models(model_path, resume=args.resume)
    print(
        f"[{dataset}] train={len(training)}, test={len(evaluation)}, "
        f"attributes={len(table.attribute_columns)}, backend=autogluon-default",
        flush=True,
    )
    manager, validation_report = load_or_train_attribute_models(
        reuse_models=args.resume,
        evaluate_only=False,
        model_path=model_path,
        frame=training,
        feature_columns=table.feature_columns,
        attribute_columns=table.attribute_columns,
        backend="autogluon",
        backend_config=backend_config(args),
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
    result = {
        "dataset": dataset,
        "scenario": "cross_subject",
        "backend": "autogluon",
        "model_suite": "default",
        "classifier_structure": "independent_per_attribute",
        "attribute_set": "initial_a1_v1_5_pre_refinement",
        "attribute_count": len(table.attribute_columns),
        "activity_accuracy": metrics["activity"]["accuracy"],
        "activity_top1_accuracy": metrics["activity"]["top1_accuracy"],
        "activity_top3_recall": metrics["activity"]["top3_recall"],
        "activity_macro_f1": metrics["activity"]["macro_f1"],
        "attribute_exact_match": metrics["attribute_exact_match"],
    }
    details.to_csv(run_dir / "details.csv", index=False)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "validation_report.json", validation_report)
    write_json(run_dir / "subject_split.json", asdict(subject_split))
    write_json(result_path, result)
    pd.DataFrame([result]).to_csv(run_dir / "summary.csv", index=False)
    print(f"[{dataset}] completed: {result}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "train-test"), default="train-test"
    )
    parser.add_argument(
        "--dataset", choices=("all", *DATASETS), default="all"
    )
    parser.add_argument("--gpu-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--num-bag-folds", type=int, default=2)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.num_cpus, args.time_limit) <= 0:
        raise ValueError("CPU count and per-attribute time limit must be positive")
    if args.num_bag_folds != 2:
        raise ValueError("This protocol requires exactly two AutoGluon bagging folds")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    selected = tuple(DATASETS) if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        _, subject_split, table = validate_dataset(dataset)
        print(
            f"Validated {dataset}: initial_attributes={len(table.attribute_columns)}, "
            f"train_subjects={list(subject_split.train_subjects)}, "
            f"test_subjects={list(subject_split.test_subjects)}, "
            "backend=autogluon, model_suite=default, attribute_order=none, "
            f"physical_gpu={args.gpu_id}, logical_gpu=0",
            flush=True,
        )
    if args.mode == "check":
        print("Check complete; no model was trained and no result directory was written.")
        return 0

    for dataset in selected:
        run_dataset(dataset, args)
    print(f"Cross-subject queue completed: {', '.join(selected)}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
