from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from zshar.evaluation.metrics import classification_summary
from zshar.evaluation.scoring import (
    score_activity_candidates,
    top_k_predictions,
    top_k_recall,
    top_predictions,
)
from zshar.models.base import classification_metrics


def evaluate_cross_subject(
    *,
    manager: Any,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    attribute_columns: Sequence[str],
    activity_vectors: Mapping[str, Mapping[str, str]],
    candidate_activities: Sequence[str],
    metric: str,
    attribute_weighting: str,
    validation_report: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate activity recognition on subjects absent from model training."""
    scores, predicted_attributes = score_activity_candidates(
        manager=manager,
        frame=frame,
        feature_columns=feature_columns,
        attribute_columns=attribute_columns,
        activity_vectors=activity_vectors,
        candidate_activities=candidate_activities,
        metric=metric,
        attribute_weighting=attribute_weighting,
        validation_report=validation_report,
    )
    predictions, margins = top_predictions(scores)
    top3 = top_k_predictions(scores, k=3)
    per_subject = {
        str(subject): classification_summary(
            group["activity_name"], predictions.loc[group.index]
        )
        for subject, group in frame.groupby("user_id", sort=True)
    }
    per_attribute = {
        attribute: classification_metrics(
            frame[attribute].astype(str), predicted_attributes[attribute]
        )
        for attribute in attribute_columns
    }
    exact = float(
        (
            frame[list(attribute_columns)].astype(str).to_numpy()
            == predicted_attributes.to_numpy()
        ).all(axis=1).mean()
    )
    details = frame[["window_index", "user_id", "activity_id", "activity_name"]].copy()
    details["predicted_activity"] = predictions
    details["correct"] = details["activity_name"] == details["predicted_activity"]
    details["prototype_margin"] = margins
    details["top3_activities"] = top3.astype(str).agg("|".join, axis=1)
    details["top3_correct"] = top3.astype(str).eq(
        details["activity_name"].astype(str), axis=0
    ).any(axis=1)
    activity_metrics = classification_summary(frame["activity_name"], predictions)
    activity_metrics["top1_accuracy"] = activity_metrics["accuracy"]
    activity_metrics["top3_recall"] = top_k_recall(frame["activity_name"], top3)
    return {
        "activity": activity_metrics,
        "per_subject": per_subject,
        "attribute_exact_match": exact,
        "per_attribute": per_attribute,
        "sample_count": int(len(frame)),
    }, details
