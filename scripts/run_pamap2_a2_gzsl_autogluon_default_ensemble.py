#!/usr/bin/env python3
"""Run calibrated PAMAP2 GZSL with initial-A1 AutoGluon default ensembles.

This experiment uses the pre-refinement initial A1 v1.5 vectors while keeping
the fixed seed-42 GZSL split/sample manifests and all classifier/calibration
settings of the matched refined-A2 experiment. Each attribute is predicted
independently; no attribute order or chained predictions are used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MATCHED_SCRIPT = (
    PROJECT / "scripts/run_pamap2_a2_gzsl_autogluon_soft_oof_reuse_zsl_order.py"
)
MATCHED_SPEC = importlib.util.spec_from_file_location(
    "pamap2_gzsl_matched_protocol", MATCHED_SCRIPT
)
matched = importlib.util.module_from_spec(MATCHED_SPEC)
assert MATCHED_SPEC.loader is not None
MATCHED_SPEC.loader.exec_module(matched)

DATASET = matched.DATASET
DATASET_LABEL = matched.DATASET_LABEL
ATTRIBUTE_SET = "initial_A1_v1_5_9d"
SPLITS = PROJECT / "split/gzsl_20/pamap2_unseen4_random20_gzsl_splits_seed42.json"
SAMPLES = (
    PROJECT
    / "split/gzsl_20/pamap2_unseen4_random20_gzsl_eval20_seen20_samples_seed42.json"
)
A1_VECTORS = (
    PROJECT / "configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5.json"
)
A1_ONTOLOGY = PROJECT / "configs/attributes/initial_attributes.json"
RUN_ROOT = Path("/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs")
RUN_NAME = (
    "pamap2_gzsl_autogluon_default_ensemble_initial_a1_v1_5_"
    "independent_calibrated_grid_cal3_t300_random20_eval20_seen20_seed42_gpu0"
)
REPORT_METRICS = matched.REPORT_METRICS
MANAGER_MANIFEST = matched.MANAGER_MANIFEST


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


def backend_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "time_limit": args.time_limit,
        "presets": "medium_quality",
        "num_gpus": 1,
        "num_cpus": args.num_cpus,
        "num_bag_folds": 0,
        "num_stack_levels": 0,
        "fit_weighted_ensemble": True,
        "model_suite": "default",
        "lightgbm_device": "cuda",
        "gpu_model_types": ["NN_TORCH", "FASTAI", "CAT", "XGB"],
        "verbosity": 2,
    }


def preflight(selected: Sequence[int]) -> tuple[Any, list[Any], list[int]]:
    from zshar.data import load_activity_vectors, load_dataset_table
    from zshar.datasets import load_dataset_spec
    from zshar.splits import (
        load_sample_manifest,
        load_split_manifest,
        sample_evaluation_rows,
    )

    for path in (SPLITS, SAMPLES, A1_VECTORS, A1_ONTOLOGY):
        if not path.is_file():
            raise FileNotFoundError(path)

    split_payload = read_json(SPLITS)
    sample_payload = read_json(SAMPLES)
    if (
        split_payload.get("dataset") != DATASET
        or split_payload.get("evaluation_protocol") != "gzsl"
        or split_payload.get("training_label_space") != "seen_only"
        or split_payload.get("candidate_label_space") != "seen_union_unseen"
        or int(split_payload.get("seed", -1)) != 42
        or int(split_payload.get("num_splits", -1)) != 20
        or int(split_payload.get("unseen_count", -1))
        != matched.UNSEEN_ACTIVITY_COUNT
    ):
        raise ValueError(f"Invalid PAMAP2 seed-42 GZSL split manifest: {SPLITS}")
    if (
        sample_payload.get("dataset") != DATASET
        or sample_payload.get("evaluation_protocol") != "gzsl"
        or int(sample_payload.get("seed", -1)) != 42
        or int(sample_payload.get("seen_samples_per_activity", -1))
        != matched.EVAL_SAMPLES_PER_ACTIVITY
        or int(sample_payload.get("unseen_samples_per_activity", -1))
        != matched.EVAL_SAMPLES_PER_ACTIVITY
    ):
        raise ValueError(f"Invalid PAMAP2 GZSL eval20/seen20 manifest: {SAMPLES}")

    spec = load_dataset_spec(DATASET)
    if spec.attributes_path.resolve() != A1_VECTORS.resolve():
        raise ValueError(f"PAMAP2 profile must point to initial A1 vectors: {A1_VECTORS}")
    vectors, attributes = load_activity_vectors(A1_VECTORS)
    expected_attributes = [
        str(item["name"]) for item in read_json(A1_ONTOLOGY).get("attributes", [])
    ]
    if set(vectors) != set(spec.activity_names):
        raise ValueError("Initial A1 vector activities differ from the PAMAP2 profile")
    if attributes != expected_attributes or len(attributes) != 9:
        raise ValueError("PAMAP2 initial A1 vectors must match the ordered 9-D ontology")
    observed_values = {
        str(value) for vector in vectors.values() for value in vector.values()
    }
    if not observed_values.issubset({"Low", "Medium", "High"}):
        raise ValueError(f"Unexpected initial A1 levels: {sorted(observed_values)}")

    splits = load_split_manifest(SPLITS, spec)
    indices = [int(split.index) for split in splits]
    if len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError("Expected exactly 20 distinct PAMAP2 GZSL splits")
    if set(map(int, sample_payload.get("samples", {}))) != set(indices):
        raise ValueError("GZSL sample-manifest split indices differ from split manifest")

    sample_manifest = load_sample_manifest(SAMPLES)
    table = load_dataset_table(spec)
    selected_set = set(map(int, selected))
    for split in splits:
        if split.index not in selected_set:
            continue
        by_activity = sample_payload["samples"][str(split.index)]
        if (
            len(split.seen_activities) != matched.SEEN_ACTIVITY_COUNT
            or len(split.unseen_activities) != matched.UNSEEN_ACTIVITY_COUNT
        ):
            raise ValueError(
                f"split {split.index} must contain {matched.SEEN_ACTIVITY_COUNT} seen + "
                f"{matched.UNSEEN_ACTIVITY_COUNT} unseen activities"
            )
        if set(map(str, by_activity)) != set(spec.activity_names):
            raise ValueError(f"split {split.index} samples do not cover all activities")
        if len(by_activity) != matched.TOTAL_ACTIVITY_COUNT or any(
            len(ids) != matched.EVAL_SAMPLES_PER_ACTIVITY
            or len(set(map(int, ids))) != matched.EVAL_SAMPLES_PER_ACTIVITY
            for ids in by_activity.values()
        ):
            raise ValueError(
                f"split {split.index} must have 20 fixed samples for all "
                f"{matched.TOTAL_ACTIVITY_COUNT} activities"
            )
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="gzsl",
            samples_per_activity=matched.EVAL_SAMPLES_PER_ACTIVITY,
            seed=42,
            manifest_ids=sample_manifest[split.index],
        )
        expected_rows = (
            matched.TOTAL_ACTIVITY_COUNT * matched.EVAL_SAMPLES_PER_ACTIVITY
        )
        if len(evaluation) != expected_rows or not evaluation.activity_name.value_counts().eq(
            matched.EVAL_SAMPLES_PER_ACTIVITY
        ).all():
            raise ValueError(
                f"split {split.index} does not resolve to "
                f"{matched.TOTAL_ACTIVITY_COUNT} x 20 balanced GZSL rows"
            )
        evaluation_ids = set(evaluation.window_index.astype(int))
        training = table.frame[
            table.frame.activity_name.isin(split.seen_activities)
            & ~table.frame.window_index.astype(int).isin(evaluation_ids)
        ]
        if set(training.activity_name.astype(str)) != set(split.seen_activities):
            raise ValueError(f"split {split.index} is missing outer-seen training rows")
        if set(training.window_index.astype(int)) & evaluation_ids:
            raise AssertionError(f"split {split.index} has train/test window leakage")

    print(
        f"Validated PAMAP2 GZSL initial-A1 protocol for {len(selected_set)} split(s); "
        f"each split has {matched.SEEN_ACTIVITY_COUNT} seen + "
        f"{matched.UNSEEN_ACTIVITY_COUNT} unseen activities and 20 fixed test rows "
        "per class.",
        flush=True,
    )
    return spec, splits, indices


def experiment_config(args: argparse.Namespace, selected: Sequence[int]) -> Any:
    from zshar.pipeline.runner import ExperimentConfig

    return ExperimentConfig(
        dataset=DATASET,
        scenario="gzsl",
        backend="autogluon",
        backend_config=backend_config(args),
        seed=42,
        split_manifest=str(SPLITS),
        sample_manifest=str(SAMPLES),
        # None makes the runner use the dataset profile's pre-refinement A1 vectors.
        attribute_vectors_root=None,
        # Intentionally omit attribute_order_config: attributes are independent.
        split_indices=set(selected),
        samples_per_activity=matched.EVAL_SAMPLES_PER_ACTIVITY,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        gzsl_bias_correction="temperature_seen_penalty",
        gamma_grid=matched.gamma_grid(),
        calibration_mode="grid",
        calibration_class_folds=3,
        calibration_pseudo_unseen_count=matched.UNSEEN_ACTIVITY_COUNT,
        calibration_samples_per_activity=3,
        calibration_temperature_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        calibration_gamma_min=-1.0,
        calibration_gamma_max=1.0,
        calibration_gamma_coarse_step=0.05,
        calibration_gamma_fine_step=0.001,
        calibration_min_seen_retention=0.5,
        calibration_min_validation_ausuc=0.0,
        calibration_one_standard_error_rule=True,
        run_root=str(args.run_root.expanduser().resolve()),
        run_name=args.run_name,
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
        old = {key: value for key, value in previous.items() if key not in ignored}
        new = {key: value for key, value in current.items() if key not in ignored}
        old["split_indices"] = sorted(map(int, old.get("split_indices", [])))
        new["split_indices"] = sorted(map(int, new.get("split_indices", [])))
        if old != new:
            raise ValueError(f"Existing run uses a different protocol: {saved}")
    return run_dir


def preserve_partial_models(run_dir: Path, *, resume: bool) -> None:
    candidates = list(run_dir.glob("split_*/models")) + list(
        run_dir.glob("split_*/calibration/fold_*/models")
    )
    partial = [path for path in candidates if not (path / MANAGER_MANIFEST).is_file()]
    if not partial:
        return
    if not resume:
        raise ValueError(
            "Incomplete model directories exist; rerun with --resume: "
            + ", ".join(map(str, partial))
        )
    for path in partial:
        destination = path.with_name("models.incomplete")
        counter = 1
        while destination.exists():
            destination = path.with_name(f"models.incomplete_{counter}")
            counter += 1
        shutil.move(str(path), str(destination))
        print(f"Preserved incomplete model directory: {destination}", flush=True)


def summarize(config: Any, selected: Sequence[int]) -> Path:
    run_dir = Path(config.run_root) / DATASET / str(config.run_name)
    summary = run_dir / "summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    frame = pd.read_csv(summary).sort_values("split_index")
    expected = set(map(int, selected))
    if len(frame) != len(expected) or set(frame.split_index.astype(int)) != expected:
        raise ValueError(f"Incomplete GZSL summary: {summary}")
    if set(frame.backend.astype(str)) != {"autogluon"}:
        raise ValueError(f"Unexpected backend in {summary}")
    if "attribute_order_path" in frame and frame.attribute_order_path.notna().any():
        raise ValueError("Independent run unexpectedly recorded an attribute order")

    rows: list[dict[str, Any]] = []
    for metric in REPORT_METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "split_count": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(run_dir / "gzsl_report.csv", index=False)
    write_json(
        run_dir / "gzsl_report.json",
        {
            "dataset": DATASET,
            "scenario": "gzsl",
            "backend": "autogluon",
            "classifier": "independent AutoGluon default ensemble per attribute",
            "attribute_set": ATTRIBUTE_SET,
            "attribute_vectors_root": None,
            "attribute_vectors_path": str(A1_VECTORS),
            "attribute_order": None,
            "attribute_dependency": "independent",
            "calibration": "3-fold outer-seen-only pseudo-GZSL grid calibration",
            "bias_correction": "temperature_seen_penalty",
            "split_count": len(expected),
            "metrics": rows,
        },
    )
    print("\n" + report.to_string(index=False), flush=True)
    print(f"GZSL report saved to: {run_dir}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "train-test", "report"), default="train-test"
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--split-indices", help="Optional comma-separated split subset")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_id != 0:
        raise ValueError("This launcher is intentionally pinned to physical GPU 0")
    if args.num_cpus <= 0 or args.time_limit <= 0:
        raise ValueError("CPU count and time limit must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    split_payload = read_json(SPLITS)
    all_indices = [int(item["split_index"]) for item in split_payload.get("splits", [])]
    if len(all_indices) != 20 or len(set(all_indices)) != 20:
        raise ValueError("Expected exactly 20 distinct PAMAP2 GZSL splits")
    selected = matched.selected_indices(args, all_indices)
    spec, _, verified = preflight(selected)
    if verified != all_indices:
        raise ValueError("GZSL split manifest changed during protocol validation")
    config = experiment_config(args, selected)
    run_dir = validate_existing_run(config)

    print(
        "PAMAP2 initial-A1 v1.5 calibrated GZSL AutoGluon default ensemble\n"
        "  attribute dependency: independent (no attribute order)\n"
        "  classifier: AutoGluon default model suite + weighted ensemble\n"
        "  calibration: 3-fold outer-seen-only pseudo-GZSL (3 rows/activity)\n"
        "  bias correction: temperature softmax + calibrated seen-class penalty\n"
        "  physical GPU: 0 (logical device 0)\n"
        f"  split manifest: {SPLITS}\n"
        f"  sample manifest: {SAMPLES}\n"
        f"  selected splits: {selected}\n"
        f"  initial A1 vectors: {A1_VECTORS}\n"
        f"  run: {run_dir}",
        flush=True,
    )
    if args.mode == "check":
        print("Check only: no classifier or calibration model was trained.", flush=True)
        return 0
    if args.mode == "report":
        summarize(config, selected)
        return 0

    preserve_partial_models(run_dir, resume=args.resume)
    from zshar.pipeline.runner import run_experiment

    output = run_experiment(spec, config)
    print(
        f"Completed calibrated PAMAP2 initial-A1 GZSL AutoGluon default ensemble: "
        f"{output}",
        flush=True,
    )
    summarize(config, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
