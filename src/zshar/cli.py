from __future__ import annotations

import argparse
import json
import os

from zshar.datasets import available_datasets, load_dataset_spec
from zshar.models import CLASSIFIER_BACKENDS
from zshar.pipeline.runner import ExperimentConfig, run_experiment
from zshar.pipeline.training import parse_backend_config
from zshar.semantics.gap_analysis import run_gap_analysis


def _csv_ints(value: str | None) -> set[int] | None:
    return {int(item.strip()) for item in value.split(",") if item.strip()} if value else None


def _csv_floats(value: str | None) -> list[float] | None:
    return [float(item.strip()) for item in value.split(",") if item.strip()] if value else None


def _run(args: argparse.Namespace) -> None:
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    backend_config = parse_backend_config(args.backend_config)
    if args.backend in {
        "autogluon",
        "autogluon_multilabel",
        "autogluon_soft_oof",
    }:
        time_limit = args.time_limit
        if time_limit is None:
            time_limit = 300 if args.backend == "autogluon_soft_oof" else 3600
        num_bag_folds = args.num_bag_folds
        if num_bag_folds is None:
            num_bag_folds = 2 if args.backend == "autogluon_soft_oof" else 0
        backend_config = {
            "time_limit": time_limit,
            "presets": args.presets,
            "num_gpus": args.num_gpus,
            "num_cpus": args.num_cpus,
            "num_bag_folds": num_bag_folds,
            "num_stack_levels": args.num_stack_levels,
            "fit_weighted_ensemble": not args.no_weighted_ensemble,
            "model_suite": args.autogluon_model_suite,
            **backend_config,
        }
    else:
        backend_config.setdefault("random_state", args.seed)
    config = ExperimentConfig(
        dataset=args.dataset,
        scenario="cross_subject" if args.scenario == "subject" else args.scenario,
        backend=args.backend,
        backend_config=backend_config,
        seed=args.seed,
        split_manifest=args.split_manifest,
        sample_manifest=args.sample_manifest,
        attribute_vectors_root=args.attribute_vectors_root,
        attribute_order_config=args.attribute_order_config,
        attribute_order_variant=args.attribute_order_variant,
        attribute_feature_bindings=args.attribute_feature_bindings,
        attribute_feature_include_auxiliary=not args.attribute_feature_primary_only,
        subject_manifest=args.subject_manifest,
        split_indices=_csv_ints(args.split_indices),
        max_splits=args.max_splits,
        samples_per_activity=args.samples_per_activity,
        prototype_metric=args.prototype_metric,
        attribute_weighting=args.attribute_weighting,
        gamma=args.gamma,
        temperature=args.temperature,
        gzsl_bias_correction=args.gzsl_bias_correction,
        gamma_grid=_csv_floats(args.gamma_grid),
        calibration_mode=args.calibration_mode,
        calibration_class_folds=args.calibration_class_folds,
        calibration_pseudo_unseen_count=args.calibration_pseudo_unseen_count,
        calibration_samples_per_activity=args.calibration_samples_per_activity,
        calibration_temperature_grid=_csv_floats(args.calibration_temperature_grid)
        or (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        calibration_tpe_trials=args.calibration_tpe_trials,
        calibration_tau_min=args.calibration_tau_min,
        calibration_tau_max=args.calibration_tau_max,
        calibration_gamma_min=args.calibration_gamma_min,
        calibration_gamma_max=args.calibration_gamma_max,
        calibration_gamma_coarse_step=args.calibration_gamma_coarse_step,
        calibration_gamma_fine_step=args.calibration_gamma_fine_step,
        calibration_min_seen_retention=args.calibration_min_seen_retention,
        calibration_min_validation_ausuc=args.calibration_min_validation_ausuc,
        calibration_one_standard_error_rule=not args.no_calibration_one_standard_error_rule,
        run_root=args.run_root,
        run_name=args.run_name,
        reuse_models=args.reuse_models,
        evaluate_only=args.evaluate_only,
        dry_run=args.dry_run,
    )
    output = run_experiment(load_dataset_spec(args.dataset, args.dataset_config), config)
    print(f"Run directory: {output}")


def _gap(args: argparse.Namespace) -> None:
    run_gap_analysis(
        dataset=args.dataset, dataset_config=args.dataset_config, a1_json=args.a1_json,
        split_manifest=args.split_manifest, seed=args.seed, max_splits=args.max_splits,
        top_k=args.top_k, distance_metric=args.distance_metric, output=args.output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zshar", description="Unified ZSL/GZSL/cross-subject HAR experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("datasets", help="list dataset profiles")
    sub.add_parser("backends", help="list shared classifier backends")

    run = sub.add_parser("run", help="run a ZSL, GZSL or cross-subject experiment")
    run.add_argument("--dataset", required=True, choices=available_datasets())
    run.add_argument("--dataset-config")
    run.add_argument(
        "--scenario",
        choices=("zsl", "gzsl", "cross_subject", "subject"),
        default="zsl",
    )
    run.add_argument("--backend", choices=CLASSIFIER_BACKENDS, default="autogluon")
    run.add_argument("--backend-config", help="JSON object or JSON file; values override named flags")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--split-manifest")
    run.add_argument("--sample-manifest")
    run.add_argument(
        "--attribute-vectors-root",
        help=(
            "root containing split_<index>/activity_attribute_vectors_a2.json; "
            "requires one valid vector file for every selected activity split"
        ),
    )
    run.add_argument(
        "--attribute-order-config",
        help=(
            "JSON containing split-specific forward/reverse/llm/llm_pseudo_zsl "
            "chain orders; supported by autogluon_multilabel, autogluon_soft_oof, "
            "catboost_classifier_chain, catboost_soft_oof, and "
            "logistic_regression_chain with either dataset-profile vectors or "
            "split-specific A2 vectors"
        ),
    )
    run.add_argument(
        "--attribute-order-variant",
        choices=("forward", "reverse", "llm", "llm_pseudo_zsl"),
        default="forward",
    )
    run.add_argument(
        "--attribute-feature-bindings",
        help=(
            "bound ontology JSON containing primary_groups/auxiliary_groups; each "
            "attribute model receives only columns from its selected groups"
        ),
    )
    run.add_argument(
        "--attribute-feature-primary-only",
        action="store_true",
        help="with --attribute-feature-bindings, exclude auxiliary groups",
    )
    run.add_argument(
        "--subject-manifest",
        help="index-explicit subject_80_20 manifest; defaults by dataset and seed",
    )
    run.add_argument("--split-indices", help="comma-separated manifest split indices")
    run.add_argument("--max-splits", type=int)
    run.add_argument("--samples-per-activity", type=int)
    run.add_argument("--prototype-metric", choices=("cosine", "euclidean", "manhattan", "ordinal_probability"), default="cosine")
    run.add_argument("--attribute-weighting", choices=("uniform", "validation_accuracy"), default="uniform")
    run.add_argument("--gamma", type=float, default=0.0)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument(
        "--gzsl-bias-correction",
        choices=("temperature_seen_penalty", "seen_penalty_only"),
        default="temperature_seen_penalty",
        help=(
            "GZSL score correction: preserve the existing temperature-softmax plus seen "
            "penalty path, or subtract the seen penalty directly from raw scores"
        ),
    )
    run.add_argument("--gamma-grid", help="comma-separated values used to compute AUSUC")
    run.add_argument(
        "--calibration-mode",
        choices=("fixed", "grid", "tpe"),
        default="fixed",
        help="GZSL calibration: fixed values or outer-seen-only automatic selection",
    )
    run.add_argument("--calibration-class-folds", type=int, default=3)
    run.add_argument("--calibration-pseudo-unseen-count", type=int)
    run.add_argument("--calibration-samples-per-activity", type=int, default=20)
    run.add_argument(
        "--calibration-temperature-grid",
        default="0.01,0.03,0.1,0.3,1,3,10",
        help="positive comma-separated temperatures for calibration-mode=grid",
    )
    run.add_argument("--calibration-tpe-trials", type=int, default=50)
    run.add_argument("--calibration-tau-min", type=float, default=0.01)
    run.add_argument("--calibration-tau-max", type=float, default=10.0)
    run.add_argument("--calibration-gamma-min", type=float, default=-1.0)
    run.add_argument("--calibration-gamma-max", type=float, default=1.0)
    run.add_argument("--calibration-gamma-coarse-step", type=float, default=0.05)
    run.add_argument("--calibration-gamma-fine-step", type=float, default=0.001)
    run.add_argument("--calibration-min-seen-retention", type=float, default=0.5)
    run.add_argument("--calibration-min-validation-ausuc", type=float, default=0.0)
    run.add_argument("--no-calibration-one-standard-error-rule", action="store_true")
    run.add_argument("--run-root")
    run.add_argument("--run-name")
    run.add_argument("--reuse-models", action="store_true")
    run.add_argument(
        "--evaluate-only",
        action="store_true",
        help="require and reuse every saved split model; never train missing models",
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--gpu-id", help="sets CUDA_VISIBLE_DEVICES; omitted means preserve the environment")
    run.add_argument("--num-gpus", type=int, default=1)
    run.add_argument("--num-cpus", type=int)
    run.add_argument(
        "--time-limit",
        type=int,
        help="seconds per attribute predictor (default: soft OOF 300, other AutoGluon 3600)",
    )
    run.add_argument("--presets", default="medium_quality")
    run.add_argument(
        "--num-bag-folds",
        type=int,
        help="bagging folds (default: soft OOF 2, other AutoGluon 0)",
    )
    run.add_argument("--num-stack-levels", type=int, default=0)
    run.add_argument(
        "--autogluon-model-suite",
        choices=("default", "zshar_five"),
        default="default",
        help=(
            "AutoGluon base-model library: default, or zshar_five "
            "(CatBoost+LightGBM+ExtraTrees+MLP Independent+Logistic Regression)"
        ),
    )
    run.add_argument("--no-weighted-ensemble", action="store_true")
    run.set_defaults(handler=_run)

    gap = sub.add_parser("gap", help="compute dataset-independent semantic GAP")
    gap.add_argument("--dataset", required=True, choices=available_datasets())
    gap.add_argument("--dataset-config")
    gap.add_argument("--a1-json", "--a1_json", dest="a1_json",
                     help="Current A1 activity vectors; defaults to the dataset profile attributes path")
    gap.add_argument("--seed", type=int, default=42)
    gap.add_argument("--split-manifest")
    gap.add_argument("--max-splits", type=int)
    gap.add_argument("--top-k", type=int, default=2)
    gap.add_argument("--distance-metric", choices=("normalized_ordinal", "mad", "euclidean"), default="normalized_ordinal")
    gap.add_argument("--output", required=True)
    gap.set_defaults(handler=_gap)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "datasets":
        print("\n".join(available_datasets()))
    elif args.command == "backends":
        print("\n".join(CLASSIFIER_BACKENDS))
    else:
        args.handler(args)


if __name__ == "__main__":
    main()
