from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from zshar.evaluation.calibration import build_gamma_grid, tune_calibration
from zshar.evaluation.scoring import score_activity_candidates
from zshar.pipeline.training import load_or_train_attribute_models


def select_pseudo_unseen_partitions(
    activities: Sequence[str],
    *,
    pseudo_unseen_count: int,
    fold_count: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    """Choose reproducible, diverse class-wise calibration partitions."""
    names = list(dict.fromkeys(str(item) for item in activities))
    count = int(pseudo_unseen_count)
    if len(names) < 2 or count <= 0 or count >= len(names):
        raise ValueError("pseudo_unseen_count must be between 1 and num_seen_activities - 1")
    combinations = list(itertools.combinations(names, count))
    random.Random(seed).shuffle(combinations)
    wanted = min(max(1, int(fold_count)), len(combinations))
    selected: list[tuple[str, ...]] = []
    coverage = {name: 0 for name in names}
    while len(selected) < wanted:
        remaining = [candidate for candidate in combinations if candidate not in selected]
        candidate = max(
            remaining,
            key=lambda values: (
                sum(1.0 / (1.0 + coverage[name]) for name in values),
                min(
                    (len(set(values) ^ set(previous)) for previous in selected),
                    default=len(values),
                ),
            ),
        )
        selected.append(candidate)
        for name in candidate:
            coverage[name] += 1
    result = []
    for pseudo_unseen in selected:
        unseen_set = set(pseudo_unseen)
        result.append(
            ([name for name in names if name not in unseen_set], list(pseudo_unseen))
        )
    return result


def build_calibration_fold_frames(
    frame: pd.DataFrame,
    *,
    pseudo_seen: Sequence[str],
    pseudo_unseen: Sequence[str],
    samples_per_activity: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build class-disjoint pseudo-GZSL training and balanced validation rows."""
    if samples_per_activity <= 0:
        raise ValueError("calibration_samples_per_activity must be positive")
    candidates = list(pseudo_seen) + list(pseudo_unseen)
    validation_parts = []
    for offset, activity in enumerate(candidates):
        group = frame[frame["activity_name"] == activity]
        if group.empty:
            raise ValueError(f"No calibration rows for activity {activity!r}")
        if len(group) < int(samples_per_activity):
            raise ValueError(
                f"Calibration activity {activity!r} has {len(group)} rows, fewer than "
                f"the required {samples_per_activity} validation samples"
            )
        validation_parts.append(
            group.sample(n=int(samples_per_activity), random_state=seed + offset * 1009)
        )
    validation = pd.concat(validation_parts).sort_values("window_index").copy()
    validation_ids = set(validation["window_index"].astype(int))
    training = frame[
        frame["activity_name"].isin(pseudo_seen)
        & ~frame["window_index"].isin(validation_ids)
    ].copy()
    if training.empty:
        raise ValueError("Pseudo-GZSL calibration training frame is empty")
    if set(training["activity_name"]) != set(pseudo_seen):
        missing = sorted(set(pseudo_seen) - set(training["activity_name"]))
        raise ValueError(f"Pseudo-seen activities have no remaining training rows: {missing}")
    if set(training["window_index"]) & validation_ids:
        raise AssertionError("Calibration training/validation window leakage detected")
    return training, validation


def _json_safe_fold_manifest(
    *,
    index: int,
    pseudo_seen: Sequence[str],
    pseudo_unseen: Sequence[str],
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "fold_index": int(index),
        "pseudo_seen": list(pseudo_seen),
        "pseudo_unseen": list(pseudo_unseen),
        "train_samples": int(len(training)),
        "validation_samples": int(len(validation)),
        "train_window_indices": training["window_index"].astype(int).tolist(),
        "validation_window_indices": validation["window_index"].astype(int).tolist(),
    }


def run_gzsl_calibration(
    *,
    frame: pd.DataFrame,
    seen_activities: Sequence[str],
    feature_columns: Sequence[str],
    attribute_columns: Sequence[str],
    activity_vectors: Mapping[str, Mapping[str, str]],
    backend: str,
    backend_config: Mapping[str, Any],
    root_path: str | Path,
    reuse_models: bool,
    seed: int,
    prototype_metric: str,
    attribute_weighting: str,
    mode: str,
    fold_count: int,
    pseudo_unseen_count: int,
    samples_per_activity: int,
    temperature_grid: Sequence[float],
    tpe_trials: int,
    tau_min: float,
    tau_max: float,
    gamma_min: float,
    gamma_max: float,
    gamma_coarse_step: float,
    gamma_fine_step: float,
    min_seen_retention: float,
    min_validation_ausuc: float,
    one_standard_error_rule: bool,
    bias_correction: str = "temperature_seen_penalty",
) -> tuple[float, float, dict[str, Any]]:
    """Train pseudo-unseen folds and select calibration without final-test access."""
    root = Path(root_path)
    root.mkdir(parents=True, exist_ok=True)
    partitions = select_pseudo_unseen_partitions(
        seen_activities,
        pseudo_unseen_count=pseudo_unseen_count,
        fold_count=fold_count,
        seed=seed,
    )
    tuning_folds: list[dict[str, Any]] = []
    fold_manifests: list[dict[str, Any]] = []
    for fold_index, (pseudo_seen, pseudo_unseen) in enumerate(partitions, start=1):
        training, validation = build_calibration_fold_frames(
            frame,
            pseudo_seen=pseudo_seen,
            pseudo_unseen=pseudo_unseen,
            samples_per_activity=samples_per_activity,
            seed=seed + fold_index * 7919,
        )
        fold_root = root / f"fold_{fold_index:02d}"
        manager, validation_report = load_or_train_attribute_models(
            reuse_models=reuse_models,
            model_path=fold_root / "models",
            frame=training,
            feature_columns=list(feature_columns),
            attribute_columns=list(attribute_columns),
            backend=backend,
            backend_config=dict(backend_config),
            seed=seed + fold_index,
        )
        candidates = list(pseudo_seen) + list(pseudo_unseen)
        scores, _ = score_activity_candidates(
            manager=manager,
            frame=validation,
            feature_columns=feature_columns,
            attribute_columns=attribute_columns,
            activity_vectors=activity_vectors,
            candidate_activities=candidates,
            metric=prototype_metric,
            attribute_weighting=attribute_weighting,
            validation_report=validation_report,
        )
        tuning_folds.append(
            {
                "scores": scores,
                "truth": validation["activity_name"].astype(str).tolist(),
                "pseudo_seen": list(pseudo_seen),
                "pseudo_unseen": list(pseudo_unseen),
            }
        )
        manifest = _json_safe_fold_manifest(
            index=fold_index,
            pseudo_seen=pseudo_seen,
            pseudo_unseen=pseudo_unseen,
            training=training,
            validation=validation,
        )
        fold_manifests.append(manifest)
        (fold_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    gamma_values = (
        [0.0]
        if bias_correction == "temperature_only"
        else build_gamma_grid(
            minimum=gamma_min,
            maximum=gamma_max,
            coarse_step=gamma_coarse_step,
            fine_step=gamma_fine_step,
        )
    )
    summary = tune_calibration(
        tuning_folds,
        mode=mode,
        gamma_values=gamma_values,
        temperature_grid=temperature_grid,
        tpe_trials=tpe_trials,
        tau_min=tau_min,
        tau_max=tau_max,
        seed=seed,
        min_seen_retention=min_seen_retention,
        min_validation_ausuc=min_validation_ausuc,
        one_standard_error_rule=one_standard_error_rule,
        bias_correction=bias_correction,
    )
    summary["protocol"] = "outer_seen_only_multifold_pseudo_gzsl"
    summary["folds"] = fold_manifests
    summary["gamma_grid"] = (
        {"fixed": True, "value": 0.0, "count": 1}
        if bias_correction == "temperature_only"
        else {
            "minimum": float(gamma_min),
            "maximum": float(gamma_max),
            "coarse_step_below_zero": float(gamma_coarse_step),
            "fine_step_from_zero": float(gamma_fine_step),
            "count": len(gamma_values),
        }
    )
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return float(summary["best_temperature"]), float(summary["best_gamma"]), summary
