from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from zshar.data import load_activity_vectors, load_dataset_table, replace_activity_vectors
from zshar.datasets import DatasetSpec
from zshar.evaluation.cross_subject import evaluate_cross_subject
from zshar.evaluation.calibration import GZSL_BIAS_CORRECTIONS
from zshar.evaluation.gzsl import evaluate_gzsl
from zshar.evaluation.zsl import evaluate_zsl
from zshar.features.groups import resolve_attribute_features
from zshar.paths import DEFAULT_RUNTIME_ROOT
from zshar.pipeline.calibration import run_gzsl_calibration
from zshar.pipeline.training import load_or_train_attribute_models
from zshar.splits import (
    default_subject_manifest_path,
    default_manifest_paths,
    generate_activity_splits,
    load_sample_manifest,
    load_split_manifest,
    load_subject_manifest,
    sample_evaluation_rows,
    select_splits,
)


@dataclass
class ExperimentConfig:
    dataset: str
    scenario: str = "zsl"
    backend: str = "autogluon"
    backend_config: dict[str, Any] | None = None
    seed: int = 42
    split_manifest: str | None = None
    sample_manifest: str | None = None
    attribute_vectors_root: str | None = None
    attribute_order_config: str | None = None
    attribute_order_variant: str = "forward"
    attribute_feature_bindings: str | None = None
    attribute_feature_include_auxiliary: bool = True
    subject_manifest: str | None = None
    split_indices: set[int] | None = None
    max_splits: int | None = None
    samples_per_activity: int | None = None
    prototype_metric: str = "cosine"
    attribute_weighting: str = "uniform"
    gamma: float = 0.0
    temperature: float = 1.0
    gzsl_bias_correction: str = "temperature_seen_penalty"
    gamma_grid: Sequence[float] | None = None
    calibration_mode: str = "fixed"
    calibration_class_folds: int = 3
    calibration_pseudo_unseen_count: int | None = None
    calibration_samples_per_activity: int = 20
    calibration_temperature_grid: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    calibration_tpe_trials: int = 50
    calibration_tau_min: float = 0.01
    calibration_tau_max: float = 10.0
    calibration_gamma_min: float = -1.0
    calibration_gamma_max: float = 1.0
    calibration_gamma_coarse_step: float = 0.05
    calibration_gamma_fine_step: float = 0.001
    calibration_min_seen_retention: float = 0.5
    calibration_min_validation_ausuc: float = 0.0
    calibration_one_standard_error_rule: bool = True
    run_root: str | None = None
    run_name: str | None = None
    reuse_models: bool = False
    evaluate_only: bool = False
    dry_run: bool = False


def _json_default(value: Any):
    if isinstance(value, (set, tuple, Path)):
        return list(value) if not isinstance(value, Path) else str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _resolve_run_dir(config: ExperimentConfig) -> Path:
    root = Path(config.run_root) if config.run_root else DEFAULT_RUNTIME_ROOT / "zshar_runs"
    name = config.run_name or (
        f"{config.dataset}_{config.scenario}_{config.backend}_seed{config.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    return root / config.dataset / name


def _resolve_manifests(config: ExperimentConfig, spec: DatasetSpec):
    default_split, default_samples = default_manifest_paths(
        spec, scenario=config.scenario, seed=config.seed
    )
    split_path = Path(config.split_manifest) if config.split_manifest else default_split
    sample_path = Path(config.sample_manifest) if config.sample_manifest else default_samples
    splits = (
        load_split_manifest(split_path, spec)
        if split_path.exists()
        else generate_activity_splits(spec, seed=config.seed, max_splits=config.max_splits)
    )
    splits = select_splits(splits, config.split_indices)
    if config.max_splits is not None:
        splits = splits[: int(config.max_splits)]
    samples = load_sample_manifest(sample_path) if sample_path.exists() else {}
    return splits, samples, split_path, sample_path


def _resolve_split_attribute_vectors(root: Path, split_index: int) -> Path:
    candidates = (
        root / f"split_{split_index}" / "activity_attribute_vectors_a2.json",
        root / f"split_{split_index:03d}" / "activity_attribute_vectors_a2.json",
    )
    existing = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if not existing:
        expected = " or ".join(str(path) for path in dict.fromkeys(candidates))
        raise FileNotFoundError(
            f"Missing split-specific A2 attribute vectors for split {split_index}: {expected}"
        )
    if len(existing) > 1:
        digests = {
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in existing
        }
        if len(digests) != 1:
            raise ValueError(
                f"Conflicting padded and unpadded A2 vectors for split {split_index}: "
                f"{[str(path) for path in existing]}"
            )
    return existing[0]


def _attribute_vector_provenance(path: Path) -> dict[str, Any]:
    vectors, attributes = load_activity_vectors(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split_specific": True,
        "activity_count": len(vectors),
        "activities": list(vectors),
        "attribute_count": len(attributes),
        "attributes": attributes,
    }


def _resolve_attribute_feature_binding(
    *,
    path: str | Path,
    dataset: str,
    feature_columns: Sequence[str],
    attribute_columns: Sequence[str],
    include_auxiliary: bool,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Resolve a bound ontology into exact per-attribute input columns."""
    binding_path = Path(path).expanduser().resolve()
    if not binding_path.is_file():
        raise FileNotFoundError(f"Attribute-feature binding file does not exist: {binding_path}")
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("attributes"), list):
        raise ValueError("Attribute-feature bindings must contain an 'attributes' list")
    resolved = resolve_attribute_features(
        feature_columns,
        ontology=payload,
        include_auxiliary=bool(include_auxiliary),
        dataset=dataset,
    )
    mapping = {
        str(attribute): list(entry["feature_names"])
        for attribute, entry in resolved["attributes"].items()
    }
    expected = list(map(str, attribute_columns))
    missing = sorted(set(expected).difference(mapping))
    extra = sorted(set(mapping).difference(expected))
    if missing or extra or len(mapping) != len(expected):
        raise ValueError(
            "Attribute-feature binding schema does not match the activity vectors: "
            f"missing={missing}, extra={extra}"
        )
    provenance = {
        "path": str(binding_path),
        "sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        "include_auxiliary": bool(include_auxiliary),
        "group_version": resolved["group_version"],
        "attributes": {
            attribute: {
                "primary_groups": list(resolved["attributes"][attribute]["primary_groups"]),
                "auxiliary_groups": list(resolved["attributes"][attribute]["auxiliary_groups"]),
                "selected_groups": list(resolved["attributes"][attribute]["selected_groups"]),
                "feature_count": len(columns),
                "feature_names": columns,
            }
            for attribute, columns in mapping.items()
        },
    }
    return mapping, provenance


def _load_split_attribute_orders(
    path: str | Path,
    *,
    dataset: str,
    seed: int,
    variant: str,
    split_attributes: dict[int, Sequence[str]],
) -> tuple[dict[int, list[str]], dict[str, Any]]:
    """Load and validate one explicit chain order for every selected split."""
    order_path = Path(path).expanduser().resolve()
    if not order_path.is_file():
        raise FileNotFoundError(f"Attribute-order configuration does not exist: {order_path}")
    if variant not in {"forward", "reverse", "llm", "llm_pseudo_zsl"}:
        raise ValueError(
            "attribute_order_variant must be 'forward', 'reverse', 'llm', or "
            "'llm_pseudo_zsl'"
        )
    payload = json.loads(order_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("splits"), dict):
        raise ValueError("Attribute-order configuration must contain a 'splits' object")
    configured_dataset = payload.get("dataset")
    if configured_dataset is not None and str(configured_dataset) != dataset:
        raise ValueError(
            f"Attribute-order dataset mismatch: expected {dataset!r}, "
            f"found {configured_dataset!r}"
        )
    configured_seed = payload.get("seed")
    if configured_seed is not None and int(configured_seed) != int(seed):
        raise ValueError(
            f"Attribute-order seed mismatch: expected {seed}, found {configured_seed}"
        )

    orders: dict[int, list[str]] = {}
    for split_index, expected_attributes in split_attributes.items():
        split_payload = payload["splits"].get(str(split_index))
        if not isinstance(split_payload, dict) or variant not in split_payload:
            raise ValueError(
                f"Attribute-order configuration has no {variant!r} order for "
                f"split {split_index}"
            )
        raw_order = split_payload[variant]
        if not isinstance(raw_order, list) or not raw_order:
            raise ValueError(
                f"Attribute order for split {split_index} must be a non-empty list"
            )
        order = [str(attribute) for attribute in raw_order]
        if len(order) != len(set(order)):
            raise ValueError(f"Attribute order for split {split_index} contains duplicates")
        expected = [str(attribute) for attribute in expected_attributes]
        missing = sorted(set(expected).difference(order))
        extra = sorted(set(order).difference(expected))
        if missing or extra or len(order) != len(expected):
            raise ValueError(
                f"Attribute order does not match split {split_index} A2 schema: "
                f"missing={missing}, extra={extra}"
            )
        orders[split_index] = order
    provenance = {
        "path": str(order_path),
        "sha256": hashlib.sha256(order_path.read_bytes()).hexdigest(),
        "variant": variant,
        "dataset": payload.get("dataset"),
        "seed": payload.get("seed"),
    }
    return orders, provenance


def _write_run_metadata(
    run_dir: Path, spec: DatasetSpec, config: ExperimentConfig, plan: dict[str, Any]
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.evaluate_only:
        # Keep the training provenance immutable when metrics are recomputed from
        # saved models.  The reevaluation invocation is recorded separately.
        _write_json(run_dir / "reevaluation_config.json", asdict(config))
        _write_json(run_dir / "reevaluation_plan.json", plan)
        return
    _write_json(run_dir / "config.json", asdict(config))
    _write_json(run_dir / "dataset_profile.json", spec.raw)
    _write_json(run_dir / "plan.json", plan)


def _write_aggregate_metrics(
    run_dir: Path,
    summary_rows: Sequence[dict[str, Any]],
    metric_names: Sequence[str],
) -> None:
    """Persist split-level mean/SD statistics next to ``summary.csv``."""
    frame = pd.DataFrame(summary_rows)
    payload: dict[str, Any] = {
        "split_count": int(len(frame)),
        "standard_deviation": "sample (ddof=1; 0.0 when only one split is present)",
        "metrics": {},
    }
    csv_row: dict[str, Any] = {"split_count": int(len(frame))}
    for metric in metric_names:
        if metric not in frame:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        stats = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        payload["metrics"][metric] = stats
        for statistic in ("mean", "std", "min", "max"):
            csv_row[f"{metric}_{statistic}"] = stats[statistic]
    _write_json(run_dir / "aggregate_metrics.json", payload)
    pd.DataFrame([csv_row]).to_csv(run_dir / "aggregate_metrics.csv", index=False)


def _run_cross_subject(
    spec: DatasetSpec, config: ExperimentConfig, run_dir: Path
) -> Path:
    if config.split_indices or config.max_splits is not None or config.sample_manifest:
        raise ValueError(
            "cross_subject uses one index-explicit --subject-manifest; "
            "--split-indices, --max-splits and --sample-manifest do not apply."
        )
    subject_path = (
        Path(config.subject_manifest)
        if config.subject_manifest
        else default_subject_manifest_path(spec, seed=config.seed)
    )
    if not subject_path.exists():
        raise FileNotFoundError(subject_path)
    subject_split = load_subject_manifest(subject_path, spec)
    plan = {
        "dataset": spec.key,
        "scenario": "cross_subject",
        "backend": config.backend,
        "subject_manifest": str(subject_path),
        "train_subjects": list(subject_split.train_subjects),
        "test_subjects": list(subject_split.test_subjects),
        "train_samples": len(subject_split.train_indices),
        "eval_samples": len(subject_split.test_indices),
        "run_dir": str(run_dir),
    }
    if config.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "plan.json", plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return run_dir

    _write_run_metadata(run_dir, spec, config, plan)
    table = load_dataset_table(spec)
    train_ids, test_ids = set(subject_split.train_indices), set(subject_split.test_indices)
    available = set(table.frame["window_index"].astype(int))
    missing = sorted((train_ids | test_ids) - available)
    if missing:
        raise ValueError(
            f"Subject manifest references {len(missing)} windows absent after dataset filtering; "
            f"first values: {missing[:10]}"
        )
    training = table.frame[table.frame["window_index"].isin(train_ids)].copy()
    evaluation = table.frame[table.frame["window_index"].isin(test_ids)].copy()
    observed_train_subjects = set(training["user_id"].astype(int))
    observed_test_subjects = set(evaluation["user_id"].astype(int))
    if observed_train_subjects != set(subject_split.train_subjects):
        raise ValueError("Training window user IDs do not match subject manifest train_subjects.")
    if observed_test_subjects != set(subject_split.test_subjects):
        raise ValueError("Evaluation window user IDs do not match subject manifest test_subjects.")
    if observed_train_subjects & observed_test_subjects:
        raise AssertionError("Cross-subject train/test subject leakage detected")

    split_dir = run_dir / "split_001"
    split_backend_config = dict(config.backend_config or {})
    feature_binding = None
    if config.attribute_feature_bindings:
        feature_mapping, feature_binding = _resolve_attribute_feature_binding(
            path=config.attribute_feature_bindings,
            dataset=spec.key,
            feature_columns=table.feature_columns,
            attribute_columns=table.attribute_columns,
            include_auxiliary=config.attribute_feature_include_auxiliary,
        )
        split_backend_config["attribute_feature_columns"] = feature_mapping
        _write_json(split_dir / "attribute_feature_binding.json", feature_binding)
    manager, validation_report = load_or_train_attribute_models(
        reuse_models=config.reuse_models,
        evaluate_only=config.evaluate_only,
        model_path=split_dir / "models",
        frame=training,
        feature_columns=table.feature_columns,
        attribute_columns=table.attribute_columns,
        backend=config.backend,
        backend_config=split_backend_config,
        seed=config.seed,
    )
    metrics, details = evaluate_cross_subject(
        manager=manager,
        frame=evaluation,
        feature_columns=table.feature_columns,
        attribute_columns=table.attribute_columns,
        activity_vectors=table.activity_vectors,
        candidate_activities=spec.activity_names,
        metric=config.prototype_metric,
        attribute_weighting=config.attribute_weighting,
        validation_report=validation_report,
    )
    flat = {
        "dataset": spec.key,
        "scenario": "cross_subject",
        "backend": config.backend,
        "train_subjects": "|".join(map(str, subject_split.train_subjects)),
        "test_subjects": "|".join(map(str, subject_split.test_subjects)),
        "train_samples": len(training),
        "eval_samples": len(evaluation),
        "activity_accuracy": metrics["activity"]["accuracy"],
        "activity_top1_accuracy": metrics["activity"]["top1_accuracy"],
        "activity_top3_recall": metrics["activity"]["top3_recall"],
        "activity_macro_f1": metrics["activity"]["macro_f1"],
        "attribute_exact_match": metrics["attribute_exact_match"],
    }
    split_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(split_dir / "details.csv", index=False)
    _write_json(split_dir / "metrics.json", metrics)
    _write_json(split_dir / "split.json", asdict(subject_split))
    pd.DataFrame([flat]).to_csv(run_dir / "summary.csv", index=False)
    _write_aggregate_metrics(
        run_dir,
        [flat],
        ("activity_top1_accuracy", "activity_top3_recall", "activity_macro_f1"),
    )
    print(f"Completed cross-subject evaluation: {flat}")
    return run_dir


def run_experiment(spec: DatasetSpec, config: ExperimentConfig) -> Path:
    if config.scenario not in {"zsl", "gzsl", "cross_subject"}:
        raise ValueError("scenario must be zsl, gzsl or cross_subject")
    if config.calibration_mode not in {"fixed", "grid", "tpe"}:
        raise ValueError("calibration_mode must be fixed, grid or tpe")
    if config.gzsl_bias_correction not in GZSL_BIAS_CORRECTIONS:
        choices = ", ".join(GZSL_BIAS_CORRECTIONS)
        raise ValueError(f"gzsl_bias_correction must be one of: {choices}")
    if config.scenario != "gzsl" and config.calibration_mode != "fixed":
        raise ValueError("Automatic calibration is only valid for scenario=gzsl")
    if config.scenario != "gzsl" and config.gzsl_bias_correction != "temperature_seen_penalty":
        raise ValueError("GZSL bias correction options are only valid for scenario=gzsl")
    if config.attribute_vectors_root and config.scenario == "cross_subject":
        raise ValueError(
            "--attribute-vectors-root is activity-split-specific and cannot be used "
            "with scenario=cross_subject"
        )
    ordered_backends = {
        "autogluon_multilabel",
        "autogluon_soft_oof",
        "catboost",
        "catboost_classifier_chain",
        "catboost_soft_oof",
        "generic_classifier_chain",
        "logistic_regression",
        "logistic_regression_chain",
    }
    if config.attribute_order_config and config.backend not in ordered_backends:
        raise ValueError(
            "--attribute-order-config is supported only by ordered attribute "
            f"backends: {sorted(ordered_backends)}"
        )
    feature_subset_backends = {
        "autogluon",
        "lightgbm",
        "xgboost",
        "catboost",
        "logistic_regression",
        "linear_svm",
        "extra_trees",
        "random_forest",
        "neuralnet_fastai",
        "neuralnet_torch",
        "ordinal_logistic",
        "coral_mlp",
        "mlp_independent",
        "mlp_multitask",
    }
    if config.attribute_feature_bindings and config.backend not in feature_subset_backends:
        raise ValueError(
            "Attribute-specific feature subsets are supported by independent attribute "
            f"backends and mlp_multitask; unsupported backend={config.backend!r}"
        )
    if (
        config.gzsl_bias_correction in {"seen_penalty_only", "none"}
        and not np.isclose(float(config.temperature), 1.0)
    ):
        raise ValueError(
            "--temperature must be 1.0 when GZSL temperature scaling is disabled"
        )
    run_dir = _resolve_run_dir(config)
    if config.evaluate_only and not run_dir.exists():
        raise FileNotFoundError(
            f"Evaluation-only mode requires an existing run directory: {run_dir}"
        )
    if config.evaluate_only and config.scenario == "gzsl" and config.calibration_mode != "fixed":
        raise ValueError(
            "Evaluation-only mode currently requires --calibration-mode fixed for GZSL; "
            "automatic calibration trains additional models."
        )
    if config.scenario == "cross_subject":
        return _run_cross_subject(spec, config, run_dir)

    splits, sample_manifest, split_path, sample_path = _resolve_manifests(config, spec)
    if not splits:
        raise ValueError("No activity splits selected")
    vector_paths: dict[int, Path] = {}
    vector_provenance: dict[int, dict[str, Any]] = {}
    if config.attribute_vectors_root:
        vector_root = Path(config.attribute_vectors_root).expanduser().resolve()
        if not vector_root.is_dir():
            raise FileNotFoundError(f"A2 attribute-vector root does not exist: {vector_root}")
        expected_activities = set(spec.activity_names)
        for split in splits:
            vector_path = _resolve_split_attribute_vectors(vector_root, split.index)
            provenance = _attribute_vector_provenance(vector_path)
            observed_activities = set(provenance["activities"])
            if observed_activities != expected_activities:
                raise ValueError(
                    f"A2 activities do not match {spec.key} for split {split.index}: "
                    f"missing={sorted(expected_activities - observed_activities)}, "
                    f"extra={sorted(observed_activities - expected_activities)}"
                )
            vector_paths[split.index] = vector_path
            vector_provenance[split.index] = provenance
    attribute_orders: dict[int, list[str]] = {}
    attribute_order_provenance: dict[str, Any] | None = None
    if config.attribute_order_config:
        if vector_provenance:
            split_attributes = {
                split.index: vector_provenance[split.index]["attributes"]
                for split in splits
            }
        else:
            _, profile_attributes = load_activity_vectors(spec.attributes_path)
            split_attributes = {
                split.index: profile_attributes for split in splits
            }
        attribute_orders, attribute_order_provenance = _load_split_attribute_orders(
            config.attribute_order_config,
            dataset=spec.key,
            seed=config.seed,
            variant=config.attribute_order_variant,
            split_attributes=split_attributes,
        )
    base_table = load_dataset_table(spec) if config.attribute_feature_bindings else None
    feature_mapping: dict[str, list[str]] = {}
    feature_binding_provenance: dict[str, Any] | None = None
    if config.attribute_feature_bindings:
        schemas = {
            tuple(vector_provenance[split.index]["attributes"])
            if vector_provenance
            else tuple(base_table.attribute_columns)
            for split in splits
        }
        if len(schemas) != 1:
            raise ValueError(
                "One attribute-feature binding file cannot cover multiple split schemas"
            )
        attribute_schema = list(next(iter(schemas)))
        feature_mapping, feature_binding_provenance = _resolve_attribute_feature_binding(
            path=config.attribute_feature_bindings,
            dataset=spec.key,
            feature_columns=base_table.feature_columns,
            attribute_columns=attribute_schema,
            include_auxiliary=config.attribute_feature_include_auxiliary,
        )
    plan = {
        "dataset": spec.key,
        "scenario": config.scenario,
        "backend": config.backend,
        "split_count": len(splits),
        "split_indices": [item.index for item in splits],
        "split_manifest": str(split_path) if split_path.exists() else None,
        "sample_manifest": str(sample_path) if sample_path.exists() else None,
        "attribute_vectors_root": (
            str(Path(config.attribute_vectors_root).expanduser().resolve())
            if config.attribute_vectors_root
            else None
        ),
        "attribute_vectors_mode": (
            "split_specific_a2" if config.attribute_vectors_root else "dataset_profile"
        ),
        "attribute_vector_files": {
            str(index): payload for index, payload in vector_provenance.items()
        },
        "attribute_order": attribute_order_provenance,
        "attribute_orders": {
            str(index): order for index, order in attribute_orders.items()
        },
        "attribute_feature_binding": feature_binding_provenance,
        "calibration_mode": config.calibration_mode,
        "gzsl_bias_correction": config.gzsl_bias_correction,
        "run_dir": str(run_dir),
    }
    if config.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "plan.json", plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return run_dir

    _write_run_metadata(run_dir, spec, config, plan)
    if base_table is None:
        base_table = load_dataset_table(spec)
    summary_rows: list[dict[str, Any]] = []
    for split in splits:
        split_dir = run_dir / f"split_{split.index:03d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        if config.attribute_vectors_root:
            table = replace_activity_vectors(base_table, vector_paths[split.index])
            provenance = vector_provenance[split.index]
            _write_json(split_dir / "attribute_vectors.json", provenance)
        else:
            table = base_table
            provenance = {
                "path": str(spec.attributes_path.resolve()),
                "sha256": hashlib.sha256(spec.attributes_path.read_bytes()).hexdigest(),
                "split_specific": False,
                "activity_count": len(table.activity_vectors),
                "activities": list(table.activity_vectors),
                "attribute_count": len(table.attribute_columns),
                "attributes": list(table.attribute_columns),
            }
        split_backend_config = dict(config.backend_config or {})
        if feature_mapping:
            split_backend_config["attribute_feature_columns"] = feature_mapping
            _write_json(
                split_dir / "attribute_feature_binding.json",
                dict(feature_binding_provenance or {}),
            )
        split_attribute_order: list[str] | None = None
        if attribute_orders:
            split_attribute_order = attribute_orders[split.index]
            configured_order = split_backend_config.get("label_order")
            if configured_order is not None and list(configured_order) != split_attribute_order:
                raise ValueError(
                    f"backend-config label_order conflicts with the order file for "
                    f"split {split.index}"
                )
            split_backend_config["label_order"] = split_attribute_order
            _write_json(
                split_dir / "attribute_order.json",
                {
                    **dict(attribute_order_provenance or {}),
                    "split_index": split.index,
                    "order": split_attribute_order,
                },
            )
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario=config.scenario,
            samples_per_activity=config.samples_per_activity or spec.eval_samples_per_activity,
            seed=config.seed,
            manifest_ids=sample_manifest.get(split.index),
        )
        if evaluation.empty:
            raise ValueError(f"Split {split.index} has no evaluation rows")
        evaluation_ids = set(evaluation["window_index"].astype(int))
        training = table.frame[
            table.frame["activity_name"].isin(split.seen_activities)
            & ~table.frame["window_index"].isin(evaluation_ids)
        ].copy()
        if training.empty:
            raise ValueError(f"Split {split.index} has no seen training rows")

        selected_temperature, selected_gamma = float(config.temperature), float(config.gamma)
        calibration_summary: dict[str, Any] = {
            "mode": "fixed",
            "bias_correction": config.gzsl_bias_correction,
            "temperature_applied": config.gzsl_bias_correction
            in {"temperature_seen_penalty", "temperature_only"},
            "score_transform": (
                "identity"
                if config.gzsl_bias_correction in {"seen_penalty_only", "none"}
                else "temperature_softmax"
            ),
            "best_temperature": selected_temperature,
            "best_gamma": selected_gamma,
        }
        if config.scenario == "gzsl" and config.calibration_mode in {"grid", "tpe"}:
            pseudo_count = config.calibration_pseudo_unseen_count or min(
                spec.unseen_count, len(split.seen_activities) - 1
            )
            selected_temperature, selected_gamma, calibration_summary = run_gzsl_calibration(
                frame=training,
                seen_activities=split.seen_activities,
                feature_columns=table.feature_columns,
                attribute_columns=table.attribute_columns,
                activity_vectors=table.activity_vectors,
                backend=config.backend,
                backend_config={
                    **dict(config.backend_config or {}),
                    **(
                        {"attribute_feature_columns": feature_mapping}
                        if feature_mapping
                        else {}
                    ),
                },
                root_path=split_dir / "calibration",
                reuse_models=config.reuse_models,
                seed=config.seed + split.index,
                prototype_metric=config.prototype_metric,
                attribute_weighting=config.attribute_weighting,
                mode=config.calibration_mode,
                fold_count=config.calibration_class_folds,
                pseudo_unseen_count=int(pseudo_count),
                samples_per_activity=config.calibration_samples_per_activity,
                temperature_grid=config.calibration_temperature_grid,
                tpe_trials=config.calibration_tpe_trials,
                tau_min=config.calibration_tau_min,
                tau_max=config.calibration_tau_max,
                gamma_min=config.calibration_gamma_min,
                gamma_max=config.calibration_gamma_max,
                gamma_coarse_step=config.calibration_gamma_coarse_step,
                gamma_fine_step=config.calibration_gamma_fine_step,
                min_seen_retention=config.calibration_min_seen_retention,
                min_validation_ausuc=config.calibration_min_validation_ausuc,
                one_standard_error_rule=config.calibration_one_standard_error_rule,
                bias_correction=config.gzsl_bias_correction,
            )

        manager, validation_report = load_or_train_attribute_models(
            reuse_models=config.reuse_models,
            evaluate_only=config.evaluate_only,
            model_path=split_dir / "models",
            frame=training,
            feature_columns=table.feature_columns,
            attribute_columns=table.attribute_columns,
            backend=config.backend,
            backend_config=split_backend_config,
            seed=config.seed,
        )
        if split_attribute_order is not None:
            loaded_order = list(getattr(manager, "label_order", []))
            if loaded_order != split_attribute_order:
                raise ValueError(
                    f"Saved model order does not match the configured order for split "
                    f"{split.index}: saved={loaded_order}, configured={split_attribute_order}"
                )
        if config.scenario == "zsl":
            metrics, details = evaluate_zsl(
                manager=manager,
                frame=evaluation,
                feature_columns=table.feature_columns,
                attribute_columns=table.attribute_columns,
                activity_vectors=table.activity_vectors,
                unseen_activities=split.unseen_activities,
                metric=config.prototype_metric,
                attribute_weighting=config.attribute_weighting,
                validation_report=validation_report,
            )
            flat = {
                "activity_accuracy": metrics["activity"]["accuracy"],
                "activity_top1_accuracy": metrics["activity"]["top1_accuracy"],
                "activity_top3_recall": metrics["activity"]["top3_recall"],
                "activity_macro_f1": metrics["activity"]["macro_f1"],
                "attribute_exact_match": metrics["attribute_exact_match"],
            }
        else:
            metrics, details = evaluate_gzsl(
                manager=manager,
                frame=evaluation,
                feature_columns=table.feature_columns,
                attribute_columns=table.attribute_columns,
                activity_vectors=table.activity_vectors,
                seen_activities=split.seen_activities,
                unseen_activities=split.unseen_activities,
                metric=config.prototype_metric,
                attribute_weighting=config.attribute_weighting,
                validation_report=validation_report,
                gamma=selected_gamma,
                temperature=selected_temperature,
                bias_correction=config.gzsl_bias_correction,
                gamma_grid=config.gamma_grid,
            )
            metrics["calibration"] = calibration_summary
            flat = {key: metrics[key] for key in (
                "seen_accuracy", "unseen_accuracy", "harmonic_mean", "overall_accuracy"
            )}
            flat.update({
                "activity_top1_accuracy": metrics["activity"]["top1_accuracy"],
                "activity_top3_recall": metrics["activity"]["top3_recall"],
                "activity_macro_f1": metrics["activity"]["macro_f1"],
                "temperature": selected_temperature,
                "gamma": selected_gamma,
                "calibration_mode": config.calibration_mode,
                "gzsl_bias_correction": config.gzsl_bias_correction,
                "calibration_validation_ausuc": calibration_summary.get("best_ausuc"),
            })
            if "ausuc" in metrics:
                flat["ausuc"] = metrics["ausuc"]
        metrics["attribute_vectors"] = provenance
        if feature_binding_provenance is not None:
            metrics["attribute_feature_binding"] = feature_binding_provenance
        if split_attribute_order is not None:
            metrics["attribute_order"] = {
                **dict(attribute_order_provenance or {}),
                "split_index": split.index,
                "order": split_attribute_order,
            }
        row = {
            "dataset": spec.key,
            "scenario": config.scenario,
            "backend": config.backend,
            "split_index": split.index,
            "seen_activities": "|".join(split.seen_activities),
            "unseen_activities": "|".join(split.unseen_activities),
            "train_samples": len(training),
            "eval_samples": len(evaluation),
            "attribute_vectors_path": provenance["path"],
            "attribute_vectors_sha256": provenance["sha256"],
            "attribute_count": provenance["attribute_count"],
            "attribute_vectors_split_specific": provenance["split_specific"],
            "attribute_order_path": (
                attribute_order_provenance["path"]
                if attribute_order_provenance is not None
                else None
            ),
            "attribute_order_sha256": (
                attribute_order_provenance["sha256"]
                if attribute_order_provenance is not None
                else None
            ),
            "attribute_order_variant": (
                attribute_order_provenance["variant"]
                if attribute_order_provenance is not None
                else None
            ),
            **flat,
        }
        summary_rows.append(row)
        details.to_csv(split_dir / "details.csv", index=False)
        _write_json(split_dir / "metrics.json", metrics)
        _write_json(split_dir / "split.json", asdict(split))
        pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False)
        common_metrics = (
            "activity_top1_accuracy",
            "activity_top3_recall",
            "activity_macro_f1",
            "attribute_exact_match",
        )
        gzsl_metrics = (
            "seen_accuracy",
            "unseen_accuracy",
            "harmonic_mean",
            "overall_accuracy",
            "ausuc",
        )
        _write_aggregate_metrics(
            run_dir,
            summary_rows,
            common_metrics + (gzsl_metrics if config.scenario == "gzsl" else ()),
        )
        print(f"Completed split {split.index}: {flat}")
    return run_dir
