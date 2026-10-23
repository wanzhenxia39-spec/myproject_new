#!/usr/bin/env python3
"""Run PAMAP2 initial-A1 binary/ordinal-5 cross-subject tests on one GPU."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUN_ROOT = Path("/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs")
SUBJECT_MANIFEST = (
    PROJECT / "split/subject_80_20/pamap2_subject80_20_balanced_seed42.json"
)
ONTOLOGY = PROJECT / "configs/attributes/initial_attributes.json"
MANAGER_MANIFEST = "attribute_backend_manifest.json"

LEVEL_LABELS = {
    2: ("Low", "High"),
    5: ("Very Low", "Low", "Medium", "High", "Very High"),
}
VECTOR_PATHS = {
    2: PROJECT
    / "configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5_binary.json",
    5: PROJECT
    / "configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5_ordinal_5.json",
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
    return {
        "time_limit": args.time_limit,
        "presets": "medium_quality",
        "num_gpus": 1,
        "num_cpus": args.num_cpus,
        "num_bag_folds": 2,
        "num_stack_levels": 0,
        "fit_weighted_ensemble": True,
        "model_suite": "default",
        "lightgbm_device": "cuda",
        "gpu_model_types": ["NN_TORCH", "FASTAI", "CAT", "XGB"],
        "verbosity": 2,
    }


def dataset_spec(level_count: int) -> Any:
    from zshar.datasets import load_dataset_spec

    original = load_dataset_spec("pamap2")
    vector_path = VECTOR_PATHS[level_count].resolve()
    raw = dict(original.raw)
    raw["attributes"] = str(vector_path)
    return replace(original, attributes_path=vector_path, raw=raw)


def validate_protocol(level_count: int) -> tuple[Any, Any, Any]:
    from zshar.data import load_activity_vectors, load_dataset_table
    from zshar.splits import load_subject_manifest

    vector_path = VECTOR_PATHS[level_count]
    for path in (vector_path, SUBJECT_MANIFEST, ONTOLOGY):
        if not path.is_file():
            raise FileNotFoundError(path)

    spec = dataset_spec(level_count)
    vectors, attributes = load_activity_vectors(vector_path)
    expected_attributes = [
        str(item["name"]) for item in read_json(ONTOLOGY).get("attributes", [])
    ]
    if set(vectors) != set(spec.activity_names):
        raise ValueError(f"{level_count}-level vectors do not cover PAMAP2 activities")
    if attributes != expected_attributes or len(attributes) != 9:
        raise ValueError(
            f"{level_count}-level vectors must keep the ordered A1 v1.5 9-D schema"
        )

    expected_levels = set(LEVEL_LABELS[level_count])
    observed_all: set[str] = set()
    for attribute in attributes:
        observed = {str(vector[attribute]) for vector in vectors.values()}
        observed_all.update(observed)
        if not observed or not observed.issubset(expected_levels):
            raise ValueError(
                f"{attribute} contains labels outside the configured "
                f"{level_count}-level scale: {sorted(observed)}"
            )
    if observed_all != expected_levels:
        raise ValueError(
            f"The vectors must collectively cover the {level_count}-level scale; "
            f"observed={sorted(observed_all)}"
        )

    table = load_dataset_table(spec)
    subject_split = load_subject_manifest(SUBJECT_MANIFEST, spec)
    available = set(table.frame["window_index"].astype(int))
    train_ids = set(map(int, subject_split.train_indices))
    test_ids = set(map(int, subject_split.test_indices))
    if train_ids & test_ids:
        raise AssertionError("Cross-subject train/test window leakage")
    missing = sorted((train_ids | test_ids) - available)
    if missing:
        raise ValueError(f"Subject manifest references unavailable windows: {missing[:10]}")
    train_subjects = set(map(int, subject_split.train_subjects))
    test_subjects = set(map(int, subject_split.test_subjects))
    if train_subjects & test_subjects:
        raise AssertionError("Cross-subject train/test subject leakage")

    training = table.frame[table.frame["window_index"].isin(train_ids)]
    evaluation = table.frame[table.frame["window_index"].isin(test_ids)]
    if set(training["user_id"].astype(int)) != train_subjects:
        raise ValueError("Training rows do not match the subject manifest")
    if set(evaluation["user_id"].astype(int)) != test_subjects:
        raise ValueError("Evaluation rows do not match the subject manifest")
    if set(training["activity_name"].astype(str)) != set(spec.activity_names):
        raise ValueError("Training subjects do not cover all PAMAP2 activities")
    if set(evaluation["activity_name"].astype(str)) != set(spec.activity_names):
        raise ValueError("Evaluation subjects do not cover all PAMAP2 activities")

    print(
        f"Validated PAMAP2 {level_count}-level cross-subject protocol: "
        f"train_subjects={list(subject_split.train_subjects)}, "
        f"test_subjects={list(subject_split.test_subjects)}, "
        f"train_rows={len(training)}, test_rows={len(evaluation)}, "
        f"vectors={vector_path}",
        flush=True,
    )
    return spec, subject_split, table


def preserve_incomplete_models(model_path: Path, *, resume: bool) -> None:
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


def run_test(args: argparse.Namespace) -> Path:
    from zshar.evaluation.cross_subject import evaluate_cross_subject
    from zshar.pipeline.training import load_or_train_attribute_models

    level_count = int(args.level_count)
    spec, subject_split, table = validate_protocol(level_count)
    vector_path = VECTOR_PATHS[level_count].resolve()
    run_name = args.run_name or (
        "pamap2_cross_subject_autogluon_default_ensemble_initial_a1_v1_5_"
        f"{level_count}level_independent_bag2_seed42_gpu{args.gpu_id}"
    )
    run_dir = args.run_root.expanduser().resolve() / "pamap2" / run_name
    config = {
        "dataset": "pamap2",
        "scenario": "cross_subject",
        "backend": "autogluon",
        "model_suite": "default",
        "classifier_structure": "independent_per_attribute",
        "attribute_set": f"initial_a1_v1_5_pre_refinement_{level_count}level",
        "attribute_level_count": level_count,
        "attribute_level_order": list(LEVEL_LABELS[level_count]),
        "attribute_vectors": str(vector_path),
        "attribute_vectors_sha256": sha256(vector_path),
        "attribute_count": len(table.attribute_columns),
        "attribute_order_strategy": "none_independent_backend",
        "seed": 42,
        "physical_gpu": args.gpu_id,
        "logical_gpu": 0,
        "subject_manifest": str(SUBJECT_MANIFEST.resolve()),
        "subject_manifest_sha256": sha256(SUBJECT_MANIFEST),
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
        print(f"Already complete: {result_path}", flush=True)
        return run_dir

    train_ids = set(map(int, subject_split.train_indices))
    test_ids = set(map(int, subject_split.test_indices))
    training = table.frame[table.frame["window_index"].isin(train_ids)].copy()
    evaluation = table.frame[table.frame["window_index"].isin(test_ids)].copy()

    model_path = run_dir / "models"
    preserve_incomplete_models(model_path, resume=args.resume)
    print(
        f"Training PAMAP2 {level_count}-level cross-subject AutoGluon default "
        f"ensemble: train={len(training)}, test={len(evaluation)}, "
        f"attributes={len(table.attribute_columns)}, run={run_dir}",
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
        "dataset": "pamap2",
        "scenario": "cross_subject",
        "backend": "autogluon",
        "model_suite": "default",
        "classifier_structure": "independent_per_attribute",
        "attribute_set": f"initial_a1_v1_5_pre_refinement_{level_count}level",
        "attribute_level_count": level_count,
        "attribute_vectors": str(vector_path),
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
    print(f"Completed PAMAP2 {level_count}-level cross-subject test: {result}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "train-test"), default="train-test"
    )
    parser.add_argument("--level-count", type=int, choices=tuple(LEVEL_LABELS), required=True)
    parser.add_argument("--gpu-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--num-bag-folds", type=int, default=2)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_cpus <= 0 or args.time_limit != 300 or args.num_bag_folds != 2:
        raise ValueError(
            "This matched protocol requires positive CPUs, time-limit 300 and two bag folds"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    validate_protocol(args.level_count)
    if args.mode == "check":
        print("Check complete; no model was trained and no result directory was written.")
        return 0
    run_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
