#!/usr/bin/env python3
"""Resume the original calibrated MHEALTH GZSL A2 bag5 experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BASE_SCRIPT = PROJECT / "scripts/run_mhealth_a2_gzsl_autogluon_soft_oof_catboost_forward.py"
BASE_SPEC = importlib.util.spec_from_file_location("mhealth_gzsl_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(base)

REUSED_ORDER_FILE = (
    PROJECT
    / "runtime/mhealth_a2_semantic_v3_catboost_seen_only_order_seed42"
    / "order_preparation/attribute_orders_catboost.json"
)
RUN_NAME = (
    "mhealth_gzsl_autogluon_soft_oof_a2_semantic_v3_"
    "reuse_zsl_catboost_seen_only_forward_calibrated_grid_cal3_bag5_"
    "random20_eval20_seen20_seed42"
)
ORDER_REUSE_NOTE = (
    "The forward order is reused unchanged from the completed ZSL experiment. "
    "That ranking used all outer-seen windows under the ZSL protocol, including "
    "windows later designated as seen-test by the gzsl_20 sample manifest."
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_reused_order(splits: Sequence[Any], selected: Sequence[int]) -> None:
    from zshar.data import load_activity_vectors

    if not REUSED_ORDER_FILE.is_file():
        raise FileNotFoundError(f"Missing reusable order file: {REUSED_ORDER_FILE}")
    payload = read_json(REUSED_ORDER_FILE)
    if (
        payload.get("dataset") != base.DATASET
        or payload.get("attribute_set") != "semantic_v3_refined_A2_per_split_11_15d"
        or int(payload.get("seed", -1)) != 42
        or payload.get("ranking_backend") != "catboost"
    ):
        raise ValueError(f"Invalid reusable MHEALTH A2 CatBoost order: {REUSED_ORDER_FILE}")
    records = payload.get("splits", {})
    selected_set = set(map(int, selected))
    for split in splits:
        if split.index not in selected_set:
            continue
        record = records.get(str(split.index))
        if not isinstance(record, dict):
            raise ValueError(f"Reusable order omits split {split.index}")
        vector_path = base.A2_ROOT / f"split_{split.index}" / "activity_attribute_vectors_a2.json"
        _, attributes = load_activity_vectors(vector_path)
        for variant in ("forward", "reverse"):
            order = list(record.get(variant, []))
            if len(order) != len(attributes) or set(order) != set(attributes):
                raise ValueError(f"Stale or incompatible reusable order for split {split.index}")
        if record.get("attribute_vectors_sha256") != base.protocol.sha256(vector_path):
            raise ValueError(f"Stale or incompatible reusable order for split {split.index}")
        if list(record.get("seen_activities", [])) != list(split.seen_activities):
            raise ValueError(f"Stale or incompatible reusable order for split {split.index}")
        if list(record.get("unseen_activities", [])) != list(split.unseen_activities):
            raise ValueError(f"Stale or incompatible reusable order for split {split.index}")


def backend_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "time_limit": args.time_limit,
        "presets": "medium_quality",
        "num_gpus": 1,
        "num_cpus": args.num_cpus,
        "num_bag_folds": 5,
        "num_stack_levels": 0,
        "fit_weighted_ensemble": True,
        "model_suite": "default",
        "consider_labels_correlation": True,
        "uncertainty_metric": "normalized_entropy",
        "lightgbm_device": "cuda",
        "gpu_model_types": ["NN_TORCH", "FASTAI", "CAT", "XGB"],
        "verbosity": 2,
    }


def experiment_config(args: argparse.Namespace, selected: Sequence[int]) -> Any:
    from zshar.pipeline.runner import ExperimentConfig

    return ExperimentConfig(
        dataset=base.DATASET,
        scenario="gzsl",
        backend="autogluon_soft_oof",
        backend_config=backend_config(args),
        seed=42,
        split_manifest=str(base.SPLITS),
        sample_manifest=str(base.SAMPLES),
        attribute_vectors_root=str(base.A2_ROOT),
        attribute_order_config=str(REUSED_ORDER_FILE),
        attribute_order_variant="forward",
        split_indices=set(selected),
        samples_per_activity=base.EVAL_SAMPLES_PER_ACTIVITY,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        gzsl_bias_correction="temperature_seen_penalty",
        gamma_grid=base.gamma_grid(),
        calibration_mode="grid",
        calibration_class_folds=3,
        calibration_pseudo_unseen_count=4,
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


def validate_existing_run(config: Any) -> Path:
    """Allow only the intentional system-disk to data-disk migration."""
    run_dir = Path(config.run_root) / base.DATASET / str(config.run_name)
    saved = run_dir / "config.json"
    if saved.is_file():
        previous = base.normalize(read_json(saved))
        current = base.normalize(asdict(config))
        ignored = {"reuse_models", "evaluate_only", "dry_run", "run_root"}
        old = {key: value for key, value in previous.items() if key not in ignored}
        new = {key: value for key, value in current.items() if key not in ignored}
        # ExperimentConfig stores this field as a set; the saved JSON preserves
        # an arbitrary serialization order, which is not protocol-significant.
        old["split_indices"] = sorted(map(int, old.get("split_indices", [])))
        new["split_indices"] = sorted(map(int, new.get("split_indices", [])))
        if old != new:
            raise ValueError(f"Existing run uses a different protocol: {saved}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("check", "train-test", "report"), default="train-test")
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--split-indices", help="Optional comma-separated subset")
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runtime/zshar_runs")
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_id < 0 or args.num_cpus <= 0:
        raise ValueError("GPU id must be nonnegative and --num-cpus must be positive")
    if args.time_limit != 300:
        raise ValueError("This matched protocol requires --time-limit 300")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    split_payload = read_json(base.SPLITS)
    all_indices = [int(item["split_index"]) for item in split_payload.get("splits", [])]
    if len(all_indices) != 20 or len(set(all_indices)) != 20:
        raise ValueError("Expected exactly 20 distinct MHEALTH GZSL splits")
    selected = base.selected_indices(args, all_indices)
    spec, splits, _, verified = base.preflight(selected)
    if verified != all_indices:
        raise ValueError("GZSL split manifest changed during validation")
    validate_reused_order(splits, selected)
    config = experiment_config(args, selected)
    run_dir = validate_existing_run(config)

    print(
        "MHEALTH calibrated GZSL AutoGluon soft-OOF bag5 resume\n"
        "  order: reused ZSL seen-only CatBoost forward order\n"
        "  chain: 5-fold bagged OOF probabilities + normalized entropy\n"
        "  calibration: 3-fold outer-seen-only pseudo-GZSL (3 rows/activity)\n"
        f"  physical GPU: {args.gpu_id} (logical device 0)\n"
        f"  run: {run_dir}",
        flush=True,
    )
    if args.mode == "check":
        print("Check only: no classifier or calibration model was trained.", flush=True)
        return 0
    if args.mode == "report":
        base.summarize(config, selected)
        return 0

    base.preserve_partial_models(run_dir, resume=args.resume)
    from zshar.pipeline.runner import run_experiment

    output = run_experiment(spec, config)
    print(f"Completed calibrated MHEALTH GZSL bag5 resume: {output}", flush=True)
    base.summarize(config, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
