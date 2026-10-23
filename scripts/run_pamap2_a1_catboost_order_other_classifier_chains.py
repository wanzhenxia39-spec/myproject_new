#!/usr/bin/env python3
"""Run remaining PAMAP2 A1-v1.5 base learners as hard classifier chains."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

COMMON_SCRIPT = PROJECT / "scripts/run_pamap2_a1_catboost_order_independent_backends.py"
COMMON_SPEC = importlib.util.spec_from_file_location("pamap2_a1_order_common", COMMON_SCRIPT)
common = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(common)

DEFAULT_BASE_BACKENDS = (
    "lightgbm",
    "xgboost",
    "linear_svm",
    "extra_trees",
    "random_forest",
    "neuralnet_fastai",
    "neuralnet_torch",
    "ordinal_logistic",
    "coral_mlp",
    "mlp_independent",
)
RUN_ROOT = PROJECT / "runtime/zshar_runs"
COMPARISON_NAME = (
    "pamap2_zsl_other_generic_classifier_chains_a1_v1_5_"
    "catboost_seen_only_forward_random20_eval50_seed42"
)


def parse_base_backends(value: str) -> list[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested).difference(DEFAULT_BASE_BACKENDS))
    if not requested or unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown or empty base-backend selection: {unknown}; "
            f"choices={list(DEFAULT_BASE_BACKENDS)}"
        )
    if len(requested) != len(set(requested)):
        raise argparse.ArgumentTypeError("Base-backend list contains duplicates")
    return requested


def base_backend_config(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    config: dict[str, Any] = {"random_state": 42}
    if backend == "lightgbm":
        config.update({"n_jobs": args.num_cpus, "device": "cuda"})
    elif backend == "xgboost":
        config.update(
            {"n_jobs": args.num_cpus, "tree_method": "hist", "device": "cuda"}
        )
    elif backend in {"extra_trees", "random_forest"}:
        config["n_jobs"] = args.num_cpus
    elif backend == "linear_svm":
        config["calibrate_proba"] = True
    elif backend in {
        "neuralnet_fastai",
        "neuralnet_torch",
        "ordinal_logistic",
        "coral_mlp",
        "mlp_independent",
    }:
        config["device"] = "cuda:0"
    return config


def chain_config(args: argparse.Namespace, base_backend: str) -> dict[str, Any]:
    return {
        "base_backend": base_backend,
        "consider_labels_correlation": True,
        "chain_feature_encoding": "ordinal",
        **base_backend_config(args, base_backend),
    }


def run_name(base_backend: str) -> str:
    return (
        f"pamap2_zsl_{base_backend}_generic_classifier_chain_a1_v1_5_"
        "catboost_seen_only_forward_random20_eval50_seed42"
    )


def experiment_config(
    args: argparse.Namespace, base_backend: str, selected: Sequence[int]
):
    from zshar.pipeline.runner import ExperimentConfig

    return ExperimentConfig(
        dataset="pamap2",
        scenario="zsl",
        backend="generic_classifier_chain",
        backend_config=chain_config(args, base_backend),
        seed=42,
        split_manifest=str(common.SPLITS),
        sample_manifest=str(common.SAMPLES),
        attribute_order_config=str(common.ORDERS),
        attribute_order_variant="forward",
        split_indices=set(selected),
        samples_per_activity=50,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        run_root=str(args.run_root.expanduser().resolve()),
        run_name=run_name(base_backend),
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
    run_dir = Path(config.run_root) / "pamap2" / str(config.run_name)
    saved = run_dir / "config.json"
    if saved.is_file():
        previous = normalize(common.read_json(saved))
        current = normalize(asdict(config))
        ignored = {"reuse_models", "evaluate_only", "dry_run"}
        if {k: v for k, v in previous.items() if k not in ignored} != {
            k: v for k, v in current.items() if k not in ignored
        }:
            raise ValueError(f"Existing run uses a different protocol: {saved}")
    return run_dir


def summarize(
    args: argparse.Namespace,
    configs: Sequence[Any],
    selected: Sequence[int],
) -> Path:
    expected = set(selected)
    rows: list[dict[str, Any]] = []
    split_frames: list[pd.DataFrame] = []
    for config in configs:
        base_backend = str(config.backend_config["base_backend"])
        path = Path(config.run_root) / "pamap2" / str(config.run_name) / "summary.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path).sort_values("split_index")
        if len(frame) != len(expected) or set(frame.split_index.astype(int)) != expected:
            raise ValueError(f"Incomplete summary: {path}")
        if set(frame.backend.astype(str)) != {"generic_classifier_chain"}:
            raise ValueError(f"Unexpected chain backend: {path}")
        if set(frame.attribute_order_variant.astype(str)) != {"forward"}:
            raise ValueError(f"Unexpected attribute order: {path}")
        split_frames.append(frame.assign(base_backend=base_backend))
        for metric in common.METRICS:
            values = pd.to_numeric(frame[metric], errors="raise")
            rows.append(
                {
                    "base_backend": base_backend,
                    "chain_backend": "generic_classifier_chain",
                    "metric": metric,
                    "split_count": len(frame),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(frame) > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    output = args.run_root.expanduser().resolve() / "pamap2" / COMPARISON_NAME
    output.mkdir(parents=True, exist_ok=True)
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(output / "comparison.csv", index=False)
    pd.concat(split_frames, ignore_index=True).to_csv(
        output / "all_split_metrics.csv", index=False
    )
    common.write_json(
        output / "comparison.json",
        {
            "dataset": "pamap2",
            "scenario": "zsl",
            "attribute_set": "initial_A1_v1_5_9d",
            "base_backends": [config.backend_config["base_backend"] for config in configs],
            "chain_backend": "generic_classifier_chain",
            "chain_inference": "autoregressive predicted hard labels",
            "chain_feature_encoding": "ordinal",
            "attribute_order_variant": "catboost_seen_only_forward",
            "attribute_order_config": str(common.ORDERS.resolve()),
            "attribute_order_sha256": common.sha256(common.ORDERS),
            "split_count": len(expected),
            "samples_per_unseen_activity": 50,
            "prototype_metric": "ordinal_probability",
            "attribute_weighting": "validation_accuracy",
            "metrics": rows,
        },
    )
    print("\n" + aggregate.to_string(index=False), flush=True)
    print(f"Comparison saved to: {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "train-test", "report"),
        default="train-test",
    )
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument(
        "--base-backends",
        type=parse_base_backends,
        default=list(DEFAULT_BASE_BACKENDS),
    )
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--split-indices")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu_id < 0 or args.num_cpus <= 0:
        raise ValueError("GPU id must be nonnegative and resource limits positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    spec, _, available = common.preflight()
    selected = common.selected_indices(args, available)
    configs = [
        experiment_config(args, backend, selected) for backend in args.base_backends
    ]
    run_dirs = [validate_existing_run(config) for config in configs]
    print(
        "PAMAP2 A1-v1.5 remaining base learners with CatBoost-order hard chains\n"
        f"  physical GPU: {args.gpu_id}\n"
        f"  base-backend sequence: {' -> '.join(args.base_backends)}\n"
        f"  selected splits: {selected}\n"
        f"  A1 vectors: {common.A1}\n"
        f"  order file: {common.ORDERS}\n"
        + "\n".join(
            f"  {backend} run: {run_dir}"
            for backend, run_dir in zip(args.base_backends, run_dirs)
        ),
        flush=True,
    )
    if args.mode == "check":
        print("Check only: no classifier was trained or evaluated.", flush=True)
        return 0
    if args.mode == "report":
        summarize(args, configs, selected)
        return 0
    for run_dir in run_dirs:
        common.preserve_partial_models(run_dir, args.resume)
    from zshar.pipeline.runner import run_experiment

    for position, (backend, config) in enumerate(zip(args.base_backends, configs), 1):
        print(
            f"\nStage {position}/{len(configs)}: base_backend={backend}",
            flush=True,
        )
        run_experiment(spec, config)
    summarize(args, configs, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
