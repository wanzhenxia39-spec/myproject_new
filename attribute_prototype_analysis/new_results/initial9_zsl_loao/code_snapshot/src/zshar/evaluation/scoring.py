from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from zshar.semantics.prototypes import (
    probability_row,
    rank_activity_prototypes,
    resolve_attribute_weights,
)


def validation_attribute_weights(
    attribute_columns: Sequence[str],
    report: Mapping[str, Any] | None,
) -> dict[str, float]:
    per_attribute = dict((report or {}).get("per_attribute", {}))
    accuracies = {
        attribute: float(per_attribute.get(attribute, {}).get("accuracy", 1.0))
        for attribute in attribute_columns
    }
    return resolve_attribute_weights(attribute_columns, "validation_accuracy", accuracies)


def score_activity_candidates(
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = frame[list(feature_columns)]
    predicted = manager.predict_all(features).astype(str)
    probability_tables = manager.predict_proba_all(features) if metric == "ordinal_probability" else {}
    if attribute_weighting == "validation_accuracy":
        weights = validation_attribute_weights(attribute_columns, validation_report)
    else:
        weights = resolve_attribute_weights(attribute_columns, "uniform")
    prototypes = {name: activity_vectors[name] for name in candidate_activities}
    score_rows = []
    for index, predicted_row in predicted.iterrows():
        distributions = {
            attr: probability_row(probability_tables, attr, index)
            for attr in attribute_columns
        }
        ranking = rank_activity_prototypes(
            predicted_values=predicted_row.to_dict(),
            activity_vectors=prototypes,
            attribute_cols=attribute_columns,
            metric=metric,
            attribute_weights=weights,
            probability_distributions=distributions,
        )
        score_rows.append({name: match.score for name, match in ranking})
    scores = pd.DataFrame(score_rows, index=frame.index, columns=list(candidate_activities))
    return scores, predicted


def top_predictions(scores: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    values = scores.to_numpy(dtype=float)
    order = np.argsort(-values, axis=1)
    columns = np.asarray(scores.columns, dtype=object)
    first = columns[order[:, 0]]
    if scores.shape[1] > 1:
        margin = values[np.arange(len(values)), order[:, 0]] - values[np.arange(len(values)), order[:, 1]]
    else:
        margin = np.full(len(values), np.nan)
    return pd.Series(first, index=scores.index), pd.Series(margin, index=scores.index)


def top_k_predictions(scores: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Return ranked candidate names for every row, capped by candidate count."""
    if int(k) <= 0:
        raise ValueError("k must be positive")
    if scores.shape[1] == 0:
        raise ValueError("At least one activity candidate is required")
    width = min(int(k), scores.shape[1])
    values = scores.to_numpy(dtype=float)
    order = np.argsort(-values, axis=1)[:, :width]
    candidates = np.asarray(scores.columns, dtype=object)
    return pd.DataFrame(
        candidates[order],
        index=scores.index,
        columns=[f"rank_{rank}" for rank in range(1, width + 1)],
    )


def top_k_recall(truth: pd.Series, ranked: pd.DataFrame) -> float:
    """Fraction of samples whose one true class occurs in the top-k ranking."""
    if len(truth) != len(ranked):
        raise ValueError("truth and ranked predictions must have equal length")
    if not len(truth):
        return 0.0
    expected = truth.astype(str).to_numpy()[:, None]
    observed = ranked.astype(str).to_numpy()
    return float(np.any(expected == observed, axis=1).mean())
