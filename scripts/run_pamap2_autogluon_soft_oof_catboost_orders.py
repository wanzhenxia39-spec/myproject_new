#!/usr/bin/env python3
"""Run PAMAP2 AutoGluon soft-OOF chains with CatBoost-derived A2 orders.

The CatBoost seen-only order is fixed before this launcher starts.  This
script reuses the completed PAMAP2 soft-OOF protocol: split-specific old A2
vectors, 20 seed-42 ZSL splits, fixed 50-window unseen-class evaluation,
bagged OOF probabilities, ordinal-probability prototype matching,
and validation-accuracy attribute weighting.  It trains forward first and the
strictly reversed CatBoost order second on one selected physical GPU.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

COMMON_SCRIPT = PROJECT / "scripts/run_pamap2_catboost_chain_seen_only_orders.py"
COMMON_SPEC = importlib.util.spec_from_file_location(
    "pamap2_catboost_seen_only_common", COMMON_SCRIPT
)
common = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(common)

CATBOOST_ORDERS = (
    PROJECT / "configs/attribute_orders/pamap2/catboost_seen_only_seed42.json"
)
DATASET = "pamap2"
DATASET_LABEL = "PAMAP2"
SPLIT_MANIFEST = common.SPLITS
SAMPLE_MANIFEST = common.SAMPLES
SAMPLES_PER_ACTIVITY = 50
ORDER_LABEL = "CatBoost seen-only"
ORDER_SUMMARY_METADATA = {
    "order_method": "seen-only independent CatBoost validation ranking",
    "ranking_backend": "catboost",
    "selection_scope": "seen-only independent CatBoost ranking",
}
RUN_ROOT = PROJECT / "runtime/zshar_runs"
RUN_NAME_PREFIX = "pamap2_zsl_autogluon_soft_oof_a2_catboost_seen_only"
ATTRIBUTE_SET = "old_pipeline_A2_refined_per_split"
ATTRIBUTE_VECTORS_ROOT: Path | None = common.A2_ROOT
VARIANTS = ("forward", "reverse")
METRICS = (
    "activity_top1_accuracy",
    "activity_top3_recall",
    "activity_macro_f1",
    "attribute_exact_match",
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


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol() -> tuple[Any, list[Any], list[int]]:
    """Validate CatBoost ordering and every invariant shared with soft-OOF runs."""
    from zshar.data import load_activity_vectors

    spec, splits, _, indices = common.preflight()
    if not CATBOOST_ORDERS.is_file():
        raise FileNotFoundError(
            "Missing canonical CatBoost seen-only order file: "
            f"{CATBOOST_ORDERS}. Run collect_pamap2_attribute_orders.py first."
        )
    payload = read_json(CATBOOST_ORDERS)
    if (
        payload.get("dataset") != "pamap2"
        or payload.get("attribute_set") != "old_pipeline_A2_refined_per_split"
        or int(payload.get("seed", -1)) != 42
        or payload.get("ranking_backend") != "catboost"
        or payload.get("selection_scope")
        != "seen-class training data only; no unseen samples, labels, or test metrics"
        or payload.get("split_manifest_sha256") != file_hash(common.SPLITS)
    ):
        raise ValueError(f"Invalid CatBoost seen-only order metadata: {CATBOOST_ORDERS}")
    entries = payload.get("splits", {})
    if set(map(int, entries)) != set(indices):
        raise ValueError("CatBoost order file split indices differ from the split manifest")
    for split in splits:
        vector_path = common.resolve_vector(split.index)
        _, attributes = load_activity_vectors(vector_path)
        entry = entries[str(split.index)]
        forward = list(map(str, entry.get("forward", [])))
        reverse = list(map(str, entry.get("reverse", [])))
        if (
            entry.get("attribute_vectors_sha256") != file_hash(vector_path)
            or len(forward) != len(set(forward))
            or set(forward) != set(attributes)
            or reverse != forward[::-1]
        ):
            raise ValueError(f"Invalid CatBoost order for split {split.index}")
    return spec, splits, indices


def backend_config(args: argparse.Namespace) -> dict[str, Any]:
    """AutoGluon soft-OOF configuration for the requested experiment."""
    return {
        "time_limit": args.time_limit,
        "presets": args.presets,
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


def run_name_suffix(args: argparse.Namespace) -> str:
    return (
        f"bag{int(args.num_bag_folds)}_random20_"
        f"eval{SAMPLES_PER_ACTIVITY}_seed42"
    )


def experiment_config(
    args: argparse.Namespace, variant: str, selected: Sequence[int]
):
    from zshar.pipeline.runner import ExperimentConfig

    return ExperimentConfig(
        dataset=DATASET,
        scenario="zsl",
        backend="autogluon_soft_oof",
        backend_config=backend_config(args),
        seed=42,
        split_manifest=str(SPLIT_MANIFEST),
        sample_manifest=str(SAMPLE_MANIFEST),
        attribute_vectors_root=(
            str(ATTRIBUTE_VECTORS_ROOT) if ATTRIBUTE_VECTORS_ROOT is not None else None
        ),
        attribute_order_config=str(CATBOOST_ORDERS),
        attribute_order_variant=variant,
        split_indices=set(selected),
        samples_per_activity=SAMPLES_PER_ACTIVITY,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        run_root=str(args.run_root.expanduser().resolve()),
        run_name=f"{args.run_name_prefix}_{variant}_{run_name_suffix(args)}",
        reuse_models=args.resume,
        evaluate_only=False,
        dry_run=args.dry_run,
    )


def _normalize_config_value(value: Any) -> Any:
    """Normalize JSON-equivalent config values before protocol comparison.

    ``ExperimentConfig`` keeps some sequence defaults as tuples, while the same
    values are necessarily restored as lists from ``config.json``.  Treat those
    representations as equivalent; sets are sorted to keep split selection
    comparisons deterministic.
    """
    if isinstance(value, dict):
        return {
            str(key): _normalize_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, set):
        return sorted(_normalize_config_value(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def validate_existing_run(config: Any) -> Path:
    run_dir = Path(config.run_root) / DATASET / str(config.run_name)
    saved_config = run_dir / "config.json"
    if saved_config.is_file():
        previous = _normalize_config_value(read_json(saved_config))
        current = _normalize_config_value(asdict(config))
        for payload in (previous, current):
            if payload.get("split_indices") is not None:
                payload["split_indices"] = sorted(
                    map(int, payload["split_indices"])
                )
        ignored = {"reuse_models", "evaluate_only", "dry_run"}
        if {key: value for key, value in previous.items() if key not in ignored} != {
            key: value for key, value in current.items() if key not in ignored
        }:
            raise ValueError(f"Existing run uses a different protocol: {saved_config}")
    return run_dir


def summarize(
    args: argparse.Namespace, configs: Sequence[Any], selected: Sequence[int]
) -> Path:
    expected = set(selected)
    frames: dict[str, pd.DataFrame] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for config in configs:
        variant = str(config.attribute_order_variant)
        path = Path(config.run_root) / DATASET / str(config.run_name) / "summary.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed summary: {path}")
        frame = pd.read_csv(path).set_index("split_index").sort_index()
        if frame.index.has_duplicates or set(map(int, frame.index)) != expected:
            raise ValueError(f"Expected exactly {len(expected)} completed splits: {path}")
        if set(frame["backend"].astype(str)) != {"autogluon_soft_oof"}:
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

    paired: pd.DataFrame | None = None
    if {"forward", "reverse"}.issubset(frames):
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
        args.run_root.expanduser().resolve()
        / DATASET
        / (
            f"{args.run_name_prefix}_{'_'.join(VARIANTS)}_comparison_"
            f"{run_name_suffix(args)}"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output / "comparison.csv", index=False)
    if paired is not None:
        paired.to_csv(output / "paired_split_metrics.csv")
    write_json(
        output / "comparison.json",
        {
            "dataset": DATASET,
            "scenario": "zsl",
            "attribute_set": ATTRIBUTE_SET,
            "backend": "autogluon_soft_oof",
            "chain_features": "bagged OOF class probabilities plus normalized entropy",
            **ORDER_SUMMARY_METADATA,
            "split_count_per_order": len(expected),
            "order_sequence": list(VARIANTS),
            "attribute_order_config": str(CATBOOST_ORDERS.resolve()),
            "attribute_order_sha256": file_hash(CATBOOST_ORDERS),
            "backend_config": backend_config(args),
            "metrics": aggregate_rows,
        },
    )
    print("\n" + aggregate.to_string(index=False), flush=True)
    print(f"Order comparison saved to: {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "train-test", "report"), default="train-test"
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name-prefix", default=RUN_NAME_PREFIX)
    parser.add_argument("--split-indices", help="Optional comma-separated subset for a smoke test")
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--num-bag-folds", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu_id < 0 or args.time_limit <= 0 or args.num_cpus <= 0:
        raise ValueError("--gpu-id, --time-limit, and --num-cpus must be positive")
    if args.num_bag_folds < 2:
        raise ValueError("Soft OOF requires --num-bag-folds >= 2")
    if args.presets != "medium_quality":
        raise ValueError("This matched protocol requires --presets medium_quality")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    spec, _, all_indices = validate_protocol()
    selected = common.selected_indices(args, all_indices)
    configs = [experiment_config(args, variant, selected) for variant in VARIANTS]
    run_dirs = [validate_existing_run(config) for config in configs]
    print(
        f"{DATASET_LABEL} AutoGluon soft-OOF chain with {ORDER_LABEL} orders\n"
        f"  execution order: {' -> '.join(VARIANTS)}\n"
        f"  physical GPU: {args.gpu_id} (one visible GPU to AutoGluon)\n"
        f"  order file: {CATBOOST_ORDERS}\n"
        f"  selected splits: {selected}\n"
        + "\n".join(
            f"  {variant} run: {run_dir}"
            for variant, run_dir in zip(VARIANTS, run_dirs)
        ),
        flush=True,
    )
    if args.mode == "check":
        return 0
    if args.mode == "report":
        summarize(args, configs, selected)
        return 0
    for run_dir in run_dirs:
        common.preserve_partial_models(
            run_dir, resume=args.resume, dry_run=args.dry_run
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
