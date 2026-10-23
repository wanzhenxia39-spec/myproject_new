#!/usr/bin/env python3
"""Run calibrated PAMAP2 GZSL with split-specific A2 soft-OOF chains.

The launcher uses the fixed seed-42 GZSL split/sample manifests, the refined
semantic-v3 A2 vectors, and the CatBoost forward orders previously selected
using ZSL seen-only data.  GZSL bias calibration uses only outer-seen data.
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

ZSL_PROTOCOL_SCRIPT = (
    PROJECT / "scripts/run_pamap2_a2_autogluon_soft_oof_catboost_forward.py"
)
ZSL_PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "pamap2_a2_semantic_v3_zsl_protocol", ZSL_PROTOCOL_SCRIPT
)
zsl_protocol = importlib.util.module_from_spec(ZSL_PROTOCOL_SPEC)
assert ZSL_PROTOCOL_SPEC.loader is not None
ZSL_PROTOCOL_SPEC.loader.exec_module(zsl_protocol)

DATASET = "pamap2"
DATASET_LABEL = "PAMAP2"
ATTRIBUTE_SET = "semantic_v3_refined_A2_per_split_11_15d"
SPLITS = PROJECT / "split/gzsl_20/pamap2_unseen4_random20_gzsl_splits_seed42.json"
SAMPLES = (
    PROJECT
    / "split/gzsl_20/pamap2_unseen4_random20_gzsl_eval20_seen20_samples_seed42.json"
)
A2_ROOT = PROJECT / "runtime/refinement_semantic_v3/pamap2_gpt_5_6_sol_11_15"
REUSED_ORDER_FILE = (
    PROJECT
    / "runtime/pamap2_a2_semantic_v3_catboost_seen_only_order_seed42"
    / "order_preparation/attribute_orders_catboost.json"
)
RUN_ROOT = Path("/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs")
RUN_NAME = (
    "pamap2_gzsl_autogluon_soft_oof_a2_semantic_v3_"
    "reuse_zsl_catboost_seen_only_forward_calibrated_grid_cal3_bag2_"
    "random20_eval20_seen20_seed42_gpu1"
)
EVAL_SAMPLES_PER_ACTIVITY = 20
SEEN_ACTIVITY_COUNT = 8
UNSEEN_ACTIVITY_COUNT = 4
TOTAL_ACTIVITY_COUNT = SEEN_ACTIVITY_COUNT + UNSEEN_ACTIVITY_COUNT
MANAGER_MANIFEST = "attribute_backend_manifest.json"
ORDER_REUSE_NOTE = (
    "The forward order is reused unchanged from the completed ZSL experiment. "
    "Its CatBoost ranking used seen-class data only and never used GZSL test rows."
)
REPORT_METRICS = (
    "seen_accuracy",
    "unseen_accuracy",
    "harmonic_mean",
    "overall_accuracy",
    "activity_top1_accuracy",
    "activity_top3_recall",
    "activity_macro_f1",
    "ausuc",
    "calibration_validation_ausuc",
    "temperature",
    "gamma",
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


def selected_indices(args: argparse.Namespace, available: Sequence[int]) -> list[int]:
    return zsl_protocol.selected_indices(args, available)


def preflight(
    selected: Sequence[int], *, require_order: bool = True
) -> tuple[Any, list[Any], list[int]]:
    from zshar.data import load_dataset_table, replace_activity_vectors
    from zshar.datasets import load_dataset_spec
    from zshar.splits import (
        load_sample_manifest,
        load_split_manifest,
        sample_evaluation_rows,
    )

    required_paths = [SPLITS, SAMPLES, A2_ROOT]
    if require_order:
        required_paths.append(REUSED_ORDER_FILE)
    for path in required_paths:
        if not path.exists():
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
        or int(split_payload.get("unseen_count", -1)) != UNSEEN_ACTIVITY_COUNT
    ):
        raise ValueError(f"Invalid PAMAP2 seed-42 GZSL split manifest: {SPLITS}")
    if (
        sample_payload.get("dataset") != DATASET
        or sample_payload.get("evaluation_protocol") != "gzsl"
        or int(sample_payload.get("seed", -1)) != 42
        or int(sample_payload.get("seen_samples_per_activity", -1))
        != EVAL_SAMPLES_PER_ACTIVITY
        or int(sample_payload.get("unseen_samples_per_activity", -1))
        != EVAL_SAMPLES_PER_ACTIVITY
    ):
        raise ValueError(f"Invalid PAMAP2 GZSL eval20/seen20 manifest: {SAMPLES}")

    spec = load_dataset_spec(DATASET)
    splits = load_split_manifest(SPLITS, spec)
    indices = [int(split.index) for split in splits]
    if len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError("Expected exactly 20 distinct PAMAP2 GZSL splits")
    if set(map(int, sample_payload.get("samples", {}))) != set(indices):
        raise ValueError("GZSL sample-manifest split indices differ from split manifest")
    sample_manifest = load_sample_manifest(SAMPLES)
    base_table = load_dataset_table(spec)

    selected_set = set(map(int, selected))
    for split in splits:
        if split.index not in selected_set:
            continue
        zsl_protocol.validate_a2_artifacts(split, spec)
        vector_path = A2_ROOT / f"split_{split.index}" / "activity_attribute_vectors_a2.json"
        table = replace_activity_vectors(base_table, vector_path)
        by_activity = sample_payload["samples"][str(split.index)]
        if (
            len(split.seen_activities) != SEEN_ACTIVITY_COUNT
            or len(split.unseen_activities) != UNSEEN_ACTIVITY_COUNT
        ):
            raise ValueError(
                f"split {split.index} must contain {SEEN_ACTIVITY_COUNT} seen + "
                f"{UNSEEN_ACTIVITY_COUNT} unseen activities"
            )
        if set(map(str, by_activity)) != set(spec.activity_names):
            raise ValueError(f"split {split.index} samples do not cover all activities")
        if len(by_activity) != TOTAL_ACTIVITY_COUNT or any(
            len(ids) != EVAL_SAMPLES_PER_ACTIVITY
            or len(set(map(int, ids))) != EVAL_SAMPLES_PER_ACTIVITY
            for ids in by_activity.values()
        ):
            raise ValueError(
                f"split {split.index} must have 20 fixed samples for all "
                f"{TOTAL_ACTIVITY_COUNT} activities"
            )
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="gzsl",
            samples_per_activity=EVAL_SAMPLES_PER_ACTIVITY,
            seed=42,
            manifest_ids=sample_manifest[split.index],
        )
        expected_rows = TOTAL_ACTIVITY_COUNT * EVAL_SAMPLES_PER_ACTIVITY
        if len(evaluation) != expected_rows or not evaluation.activity_name.value_counts().eq(
            EVAL_SAMPLES_PER_ACTIVITY
        ).all():
            raise ValueError(
                f"split {split.index} does not resolve to "
                f"{TOTAL_ACTIVITY_COUNT} x 20 balanced GZSL rows"
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
        f"Validated PAMAP2 GZSL A2 protocol for {len(selected_set)} split(s); "
        f"each split has {SEEN_ACTIVITY_COUNT} seen + {UNSEEN_ACTIVITY_COUNT} unseen "
        "activities and 20 fixed test rows per class.",
        flush=True,
    )
    return spec, splits, indices


def validate_reused_order(splits: Sequence[Any], selected: Sequence[int]) -> None:
    from zshar.data import load_activity_vectors

    payload = read_json(REUSED_ORDER_FILE)
    if (
        payload.get("dataset") != DATASET
        or payload.get("attribute_set") != ATTRIBUTE_SET
        or int(payload.get("seed", -1)) != 42
        or payload.get("ranking_backend") != "catboost"
    ):
        raise ValueError(f"Invalid reusable PAMAP2 A2 order: {REUSED_ORDER_FILE}")
    records = payload.get("splits", {})
    if len(records) != 20:
        raise ValueError("Reusable PAMAP2 order file must contain all 20 splits")
    selected_set = set(map(int, selected))
    for split in splits:
        if split.index not in selected_set:
            continue
        record = records.get(str(split.index))
        if not isinstance(record, dict):
            raise ValueError(f"Reusable order omits split {split.index}")
        vector_path = A2_ROOT / f"split_{split.index}" / "activity_attribute_vectors_a2.json"
        _, attributes = load_activity_vectors(vector_path)
        forward = list(map(str, record.get("forward", [])))
        if len(forward) != len(set(forward)) or set(forward) != set(attributes):
            raise ValueError(f"Incompatible forward order for split {split.index}")
        if record.get("attribute_vectors_sha256") != zsl_protocol.sha256(vector_path):
            raise ValueError(f"Order/vector hash mismatch for split {split.index}")
        if list(record.get("seen_activities", [])) != list(split.seen_activities):
            raise ValueError(f"Seen-activity mismatch in order for split {split.index}")
        if list(record.get("unseen_activities", [])) != list(split.unseen_activities):
            raise ValueError(f"Unseen-activity mismatch in order for split {split.index}")


def backend_config(args: argparse.Namespace) -> dict[str, Any]:
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
    }


def gamma_grid() -> list[float]:
    from zshar.evaluation.calibration import build_gamma_grid

    return build_gamma_grid(minimum=-1.0, maximum=1.0, coarse_step=0.05, fine_step=0.001)


def experiment_config(args: argparse.Namespace, selected: Sequence[int]) -> Any:
    from zshar.pipeline.runner import ExperimentConfig

    return ExperimentConfig(
        dataset=DATASET,
        scenario="gzsl",
        backend="autogluon_soft_oof",
        backend_config=backend_config(args),
        seed=42,
        split_manifest=str(SPLITS),
        sample_manifest=str(SAMPLES),
        attribute_vectors_root=str(A2_ROOT),
        attribute_order_config=str(REUSED_ORDER_FILE),
        attribute_order_variant="forward",
        split_indices=set(selected),
        samples_per_activity=EVAL_SAMPLES_PER_ACTIVITY,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        gzsl_bias_correction="temperature_seen_penalty",
        gamma_grid=gamma_grid(),
        calibration_mode="grid",
        calibration_class_folds=3,
        calibration_pseudo_unseen_count=UNSEEN_ACTIVITY_COUNT,
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
    if set(frame.backend.astype(str)) != {"autogluon_soft_oof"}:
        raise ValueError(f"Unexpected backend in {summary}")

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
            "backend": "autogluon_soft_oof",
            "attribute_set": ATTRIBUTE_SET,
            "attribute_vectors_root": str(A2_ROOT),
            "attribute_order": "reused ZSL seen-only CatBoost forward order",
            "attribute_order_reuse_note": ORDER_REUSE_NOTE,
            "chain_features": (
                f"{config.backend_config['num_bag_folds']}-fold bagged OOF "
                "probabilities plus normalized entropy"
            ),
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
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--num-bag-folds", type=int, default=2)
    parser.add_argument("--split-indices", help="Optional comma-separated split subset")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_id < 0:
        raise ValueError("GPU id must be non-negative")
    if args.num_cpus <= 0 or args.time_limit <= 0 or args.num_bag_folds < 2:
        raise ValueError("Positive CPU/time values and at least two OOF folds are required")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    split_payload = read_json(SPLITS)
    all_indices = [int(item["split_index"]) for item in split_payload.get("splits", [])]
    if len(all_indices) != 20 or len(set(all_indices)) != 20:
        raise ValueError("Expected exactly 20 distinct PAMAP2 GZSL splits")
    selected = selected_indices(args, all_indices)
    spec, splits, verified = preflight(selected)
    if verified != all_indices:
        raise ValueError("GZSL split manifest changed during protocol validation")
    validate_reused_order(splits, selected)
    config = experiment_config(args, selected)
    run_dir = validate_existing_run(config)

    print(
        "PAMAP2 semantic-v3 A2 calibrated GZSL AutoGluon soft-OOF\n"
        "  order: reused ZSL seen-only CatBoost forward order\n"
        f"  chain: {args.num_bag_folds}-fold bagged OOF probabilities + normalized entropy\n"
        "  calibration: 3-fold outer-seen-only pseudo-GZSL (3 rows/activity)\n"
        "  bias correction: temperature softmax + calibrated seen-class penalty\n"
        f"  physical GPU: {args.gpu_id} (logical device 0)\n"
        f"  selected splits: {selected}\n"
        f"  A2 root: {A2_ROOT}\n"
        f"  reused order file: {REUSED_ORDER_FILE}\n"
        f"  order reuse note: {ORDER_REUSE_NOTE}\n"
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
    print(f"Completed calibrated PAMAP2 GZSL soft-OOF run: {output}", flush=True)
    summarize(config, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
