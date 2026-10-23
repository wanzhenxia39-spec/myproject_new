#!/usr/bin/env python3
"""Run calibrated MHEALTH GZSL with semantic-v3 A2 AutoGluon soft-OOF.

The split-specific forward attribute order is learned by lightweight CatBoost
models from outer-seen training rows only.  Fixed GZSL evaluation windows are
excluded from order selection.  Calibration uses only outer-seen data through
class-disjoint pseudo-GZSL folds; final seen/unseen test metrics never select
the temperature or seen-class penalty.
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
    PROJECT / "scripts/run_mhealth_a2_autogluon_soft_oof_catboost_forward.py"
)
ZSL_PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "mhealth_a2_semantic_v3_zsl_protocol", ZSL_PROTOCOL_SCRIPT
)
protocol = importlib.util.module_from_spec(ZSL_PROTOCOL_SPEC)
assert ZSL_PROTOCOL_SPEC.loader is not None
ZSL_PROTOCOL_SPEC.loader.exec_module(protocol)

DATASET = "mhealth"
SPLITS = PROJECT / "split/gzsl_20/mhealth_unseen4_random20_gzsl_splits_seed42.json"
SAMPLES = (
    PROJECT
    / "split/gzsl_20/mhealth_unseen4_random20_gzsl_eval20_seen20_samples_seed42.json"
)
A2_ROOT = PROJECT / "runtime/refinement_semantic_v3/mhealth_gpt_5_6_sol_11_15"
PROTOCOL_ROOT = (
    PROJECT / "runtime/mhealth_a2_semantic_v3_gzsl_catboost_seen_only_order_seed42"
)
ORDER_WORK = PROTOCOL_ROOT / "order_preparation"
ORDER_FILE = ORDER_WORK / "attribute_orders_catboost.json"
RUN_ROOT = PROJECT / "runtime/zshar_runs"
RUN_NAME = (
    "mhealth_gzsl_autogluon_soft_oof_a2_semantic_v3_"
    "catboost_seen_only_forward_calibrated_grid_cal5_bag2_"
    "random20_eval20_seen20_seed42"
)
EVAL_SAMPLES_PER_ACTIVITY = 20
MANAGER_MANIFEST = "attribute_backend_manifest.json"
METRICS = (
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
SELECTION_SCOPE = (
    "outer-seen sensor samples only; fixed GZSL evaluation windows excluded; "
    "independent per-attribute CatBoost models; no outer-unseen samples, labels, "
    "or final-test metrics"
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


def configure_order_protocol() -> None:
    """Point the maintained A2 CatBoost-order helpers at this GZSL protocol."""
    protocol.SPLITS = SPLITS
    protocol.SAMPLES = SAMPLES
    protocol.A2_ROOT = A2_ROOT
    protocol.PROTOCOL_ROOT = PROTOCOL_ROOT
    protocol.ORDER_WORK = ORDER_WORK
    protocol.ORDER_FILE = ORDER_FILE
    protocol.EVAL_SAMPLES_PER_ACTIVITY = EVAL_SAMPLES_PER_ACTIVITY
    protocol.SELECTION_SCOPE = SELECTION_SCOPE


configure_order_protocol()
_BASE_RANK_SPLIT = protocol.rank_split


def _evaluation_ids(split_index: int) -> set[int]:
    payload = read_json(SAMPLES)
    groups = payload.get("samples", {}).get(str(split_index), {})
    return {int(value) for values in groups.values() for value in values}


def rank_split_without_gzsl_test_rows(
    args: argparse.Namespace, *, split: Any, base_table: Any
) -> dict[str, Any]:
    """Reuse the ranking implementation after excluding fixed GZSL test rows."""
    from zshar.data import DatasetTable

    held_out = _evaluation_ids(split.index)
    filtered_frame = base_table.frame[
        ~base_table.frame["window_index"].astype(int).isin(held_out)
    ].copy()
    filtered = DatasetTable(
        frame=filtered_frame,
        feature_columns=list(base_table.feature_columns),
        attribute_columns=list(base_table.attribute_columns),
        activity_vectors=dict(base_table.activity_vectors),
    )
    result = _BASE_RANK_SPLIT(args, split=split, base_table=filtered)
    result.update(
        {
            "selection_scope": SELECTION_SCOPE,
            "excluded_gzsl_evaluation_window_count": len(held_out),
            "gzsl_sample_manifest": str(SAMPLES.resolve()),
            "gzsl_sample_manifest_sha256": protocol.sha256(SAMPLES),
        }
    )
    return result


# prepare_orders resolves this symbol dynamically from the imported module.
protocol.rank_split = rank_split_without_gzsl_test_rows


def selected_indices(args: argparse.Namespace, available: Sequence[int]) -> list[int]:
    return protocol.selected_indices(args, available)


def preflight(selected: Sequence[int]) -> tuple[Any, list[Any], Any, list[int]]:
    from zshar.data import load_dataset_table, replace_activity_vectors
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
        split_payload.get("dataset") != DATASET
        or split_payload.get("evaluation_protocol") != "gzsl"
        or int(split_payload.get("seed", -1)) != 42
        or int(split_payload.get("num_splits", -1)) != 20
    ):
        raise ValueError(f"Invalid MHEALTH seed-42 GZSL split manifest: {SPLITS}")
    if (
        sample_payload.get("dataset") != DATASET
        or sample_payload.get("evaluation_protocol") != "gzsl"
        or int(sample_payload.get("seed", -1)) != 42
        or int(sample_payload.get("seen_samples_per_activity", -1))
        != EVAL_SAMPLES_PER_ACTIVITY
        or int(sample_payload.get("unseen_samples_per_activity", -1))
        != EVAL_SAMPLES_PER_ACTIVITY
    ):
        raise ValueError(f"Invalid MHEALTH GZSL eval20/seen20 manifest: {SAMPLES}")

    spec = load_dataset_spec(DATASET)
    splits = load_split_manifest(SPLITS, spec)
    indices = [int(split.index) for split in splits]
    if len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError("Expected exactly 20 distinct MHEALTH GZSL splits")
    if set(map(int, sample_payload.get("samples", {}))) != set(indices):
        raise ValueError("GZSL sample-manifest split indices differ from the split manifest")
    sample_manifest = load_sample_manifest(SAMPLES)
    base_table = load_dataset_table(spec)

    selected_set = set(selected)
    for split in splits:
        if split.index not in selected_set:
            continue
        protocol.validate_a2_artifacts(split, spec)
        table = replace_activity_vectors(
            base_table,
            A2_ROOT / f"split_{split.index}" / "activity_attribute_vectors_a2.json",
        )
        by_activity = sample_payload["samples"][str(split.index)]
        # The archived gzsl_20 manifest uses legacy underscore-style activity
        # keys while the maintained dataset profile uses canonical display
        # names.  Validate class identity from the explicit window IDs below,
        # rather than comparing those presentation-only names.
        if len(by_activity) != 12:
            raise ValueError(f"split {split.index} must contain 12 evaluation classes")
        if len(split.seen_activities) != 8 or len(split.unseen_activities) != 4:
            raise ValueError(f"split {split.index} must contain 8 seen + 4 unseen classes")
        if any(
            len(ids) != EVAL_SAMPLES_PER_ACTIVITY
            or len(set(map(int, ids))) != EVAL_SAMPLES_PER_ACTIVITY
            for ids in by_activity.values()
        ):
            raise ValueError(
                f"split {split.index} must have 20 fixed samples for every activity"
            )
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="gzsl",
            samples_per_activity=EVAL_SAMPLES_PER_ACTIVITY,
            seed=42,
            manifest_ids=sample_manifest[split.index],
        )
        if len(evaluation) != 240 or not evaluation.activity_name.value_counts().eq(20).all():
            raise ValueError(
                f"split {split.index} does not resolve to 12 x 20 balanced GZSL rows"
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
        f"Validated MHEALTH GZSL A2 protocol for {len(selected_set)} split(s); "
        "each split has 8 seen + 4 unseen classes and 20 fixed test rows per class.",
        flush=True,
    )
    return spec, splits, base_table, indices


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
        attribute_order_config=str(ORDER_FILE),
        attribute_order_variant="forward",
        split_indices=set(selected),
        samples_per_activity=EVAL_SAMPLES_PER_ACTIVITY,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        gzsl_bias_correction="temperature_seen_penalty",
        gamma_grid=gamma_grid(),
        calibration_mode="grid",
        calibration_class_folds=3,
        calibration_pseudo_unseen_count=4,
        # Only 10 rows remain for the smallest seen activity after the fixed
        # 20-row GZSL test holdout.  Five preserves rows for model fitting and
        # is sufficient for the two-fold soft-OOF protocol.
        calibration_samples_per_activity=5,
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
        if {key: value for key, value in previous.items() if key not in ignored} != {
            key: value for key, value in current.items() if key not in ignored
        }:
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
    for metric in METRICS:
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
    pd.DataFrame(rows).to_csv(run_dir / "gzsl_report.csv", index=False)
    write_json(
        run_dir / "gzsl_report.json",
        {
            "dataset": DATASET,
            "scenario": "gzsl",
            "backend": "autogluon_soft_oof",
            "attribute_set": "semantic_v3_refined_A2_per_split_11_15d",
            "attribute_order": "outer-seen-only CatBoost forward ranking",
            "chain_features": "2-fold bagged OOF probabilities plus normalized entropy",
            "calibration": "3-fold outer-seen-only pseudo-GZSL grid calibration",
            "bias_correction": "temperature_seen_penalty",
            "split_count": len(expected),
            "metrics": rows,
        },
    )
    print("\n" + pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"GZSL report saved to: {run_dir}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("check", "prepare", "train-test", "report"),
        default="train-test",
    )
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--split-indices", help="Optional comma-separated subset")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--order-num-cpus", type=int, default=112)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--rebuild-orders", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_id < 0 or min(args.num_cpus, args.order_num_cpus) <= 0:
        raise ValueError("GPU id must be nonnegative and CPU values positive")
    if args.time_limit != 300:
        raise ValueError("This matched protocol requires --time-limit 300")
    if args.iterations <= 0 or args.learning_rate <= 0 or args.depth <= 0:
        raise ValueError("CatBoost order parameters must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    split_payload = read_json(SPLITS)
    all_indices = [int(item["split_index"]) for item in split_payload.get("splits", [])]
    if len(all_indices) != 20 or len(set(all_indices)) != 20:
        raise ValueError("Expected exactly 20 distinct MHEALTH GZSL splits")
    selected = selected_indices(args, all_indices)
    spec, splits, base_table, verified = preflight(selected)
    if verified != all_indices:
        raise ValueError("GZSL split manifest changed during protocol validation")

    print(
        "MHEALTH semantic-v3 A2 calibrated GZSL AutoGluon soft-OOF\n"
        "  order: outer-seen-only CatBoost; fixed GZSL test windows excluded\n"
        "  chain: forward, 2-fold bagged OOF probabilities + normalized entropy\n"
        "  calibration: 3-fold outer-seen-only pseudo-GZSL grid search "
        "(5 validation rows/activity)\n"
        "  bias correction: temperature softmax + calibrated seen-class penalty\n"
        f"  time limit per attribute fit: {args.time_limit}s\n"
        f"  physical GPU: {args.gpu_id} (logical device 0)\n"
        f"  selected splits: {selected}\n"
        f"  A2 root: {A2_ROOT}\n"
        f"  order file: {ORDER_FILE}",
        flush=True,
    )
    if args.mode == "check":
        print("Check only: no order model or classifier was trained.", flush=True)
        return 0

    if args.mode in {"prepare", "train-test"}:
        protocol.prepare_orders(args, splits, base_table, selected)
    if args.mode == "prepare":
        return 0
    protocol.validate_order_file(splits, selected)

    config = experiment_config(args, selected)
    run_dir = validate_existing_run(config)
    if args.mode == "report":
        summarize(config, selected)
        return 0
    preserve_partial_models(run_dir, resume=args.resume)

    from zshar.pipeline.runner import run_experiment

    output = run_experiment(spec, config)
    print(f"Completed calibrated MHEALTH GZSL soft-OOF run: {output}", flush=True)
    summarize(config, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
