from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from zshar.evaluation.calibration import calibrated_scores, calibration_curve, gzsl_accuracy
from zshar.evaluation.metrics import classification_summary
from zshar.evaluation.scoring import score_activity_candidates, top_k_predictions, top_k_recall


def evaluate_gzsl(
    *,
    manager: Any,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    attribute_columns: Sequence[str],
    activity_vectors: Mapping[str, Mapping[str, str]],
    seen_activities: Sequence[str],
    unseen_activities: Sequence[str],
    metric: str,
    attribute_weighting: str,
    validation_report: Mapping[str, Any] | None = None,
    gamma: float = 0.0,
    temperature: float = 1.0,
    bias_correction: str = "temperature_seen_penalty",
    gamma_grid: Sequence[float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = list(seen_activities) + list(unseen_activities)
    scores, _ = score_activity_candidates(
        manager=manager,
        frame=frame,
        feature_columns=feature_columns,
        attribute_columns=attribute_columns,
        activity_vectors=activity_vectors,
        candidate_activities=candidates,
        metric=metric,
        attribute_weighting=attribute_weighting,
        validation_report=validation_report,
    )
    adjusted = calibrated_scores(
        scores,
        seen_activities,
        gamma=gamma,
        temperature=temperature,
        bias_correction=bias_correction,
    )
    predictions = adjusted.idxmax(axis=1)
    top3 = top_k_predictions(adjusted, k=3)
    summary = gzsl_accuracy(frame["activity_name"], predictions, unseen_activities)
    activity_metrics = classification_summary(frame["activity_name"], predictions)
    activity_metrics["top1_accuracy"] = activity_metrics["accuracy"]
    activity_metrics["top3_recall"] = top_k_recall(frame["activity_name"], top3)
    if gamma_grid:
        curve, ausuc = calibration_curve(
            scores,
            frame["activity_name"],
            seen_activities,
            unseen_activities,
            gamma_grid,
            temperature=temperature,
            bias_correction=bias_correction,
        )
        summary.update({"ausuc": ausuc, "calibration_curve": curve})
    details = frame[["window_index", "user_id", "activity_id", "activity_name"]].copy()
    details["predicted_activity"] = predictions
    details["is_unseen"] = details["activity_name"].isin(unseen_activities)
    details["correct"] = details["activity_name"] == details["predicted_activity"]
    details["top3_activities"] = top3.astype(str).agg("|".join, axis=1)
    details["top3_correct"] = top3.astype(str).eq(
        details["activity_name"].astype(str), axis=0
    ).any(axis=1)
    return {
        **summary,
        "activity": activity_metrics,
        "temperature": float(temperature),
        "gamma": float(gamma),
        "bias_correction": bias_correction,
        "sample_count": int(len(frame)),
    }, details
