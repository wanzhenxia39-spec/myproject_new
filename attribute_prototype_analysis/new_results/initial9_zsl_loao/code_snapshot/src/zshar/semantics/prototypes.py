"""Configurable matching between predicted attributes and activity prototypes.

All public scores use the convention "larger is better".  Distance-based
methods return ``score = 1 - normalized_distance``; this preserves the same
ranking as minimizing the distance while keeping score margins interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from zshar.ordinal import resolve_ordinal_values

# Kept for callers that import the historical three-level constant directly.
ORDINAL_VALUE_MAP = {"Low": 0.0, "Medium": 0.5, "High": 1.0}

PROTOTYPE_METRICS = (
    "cosine",
    "euclidean",
    "manhattan",
    "ordinal_probability",
)
ATTRIBUTE_WEIGHTING_METHODS = ("uniform", "validation_accuracy")
EPS = 1e-10


@dataclass(frozen=True)
class PrototypeMatch:
    """One candidate's matching result."""

    score: float
    distance: float


def predict_proba_for_model(
    model: Any,
    X: pd.DataFrame,
    *,
    constant_value: Optional[str] = None,
) -> pd.DataFrame:
    """Return a label-column probability table for one attribute model."""
    if constant_value is not None:
        return pd.DataFrame(
            {str(constant_value): np.ones(len(X), dtype=float)},
            index=X.index,
        )

    probabilities = model.predict_proba(X)
    if isinstance(probabilities, pd.Series):
        probabilities = probabilities.to_frame()
    if not isinstance(probabilities, pd.DataFrame):
        values = np.asarray(probabilities)
        if values.ndim == 1:
            values = values[:, None]
        class_labels = getattr(model, "class_labels", None)
        if class_labels is None:
            class_labels = getattr(model, "classes_", None)
        if class_labels is None or len(class_labels) != values.shape[1]:
            raise ValueError(
                "predict_proba did not return a DataFrame and class labels could not be resolved."
            )
        probabilities = pd.DataFrame(values, columns=list(class_labels), index=X.index)

    result = probabilities.copy()
    result.index = X.index
    result.columns = [str(column) for column in result.columns]
    return result.astype(float)


def probability_row(
    probability_tables: Mapping[str, pd.DataFrame],
    attr: str,
    sample_index: Any,
) -> Optional[dict[str, float]]:
    """Extract one normalized class-probability distribution."""
    table = probability_tables.get(attr)
    if table is None or sample_index not in table.index:
        return None
    row = table.loc[sample_index]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    distribution = {
        str(level): max(0.0, float(probability))
        for level, probability in row.items()
        if np.isfinite(float(probability))
    }
    total = sum(distribution.values())
    if not np.isfinite(total) or total <= EPS:
        return None
    return {level: probability / total for level, probability in distribution.items()}


def resolve_attribute_weights(
    attribute_cols: Sequence[str],
    weighting: str,
    validation_weights: Optional[Mapping[str, float]] = None,
) -> dict[str, float]:
    """Resolve positive weights for a selected weighting policy."""
    if weighting not in ATTRIBUTE_WEIGHTING_METHODS:
        raise ValueError(
            f"Unknown attribute weighting {weighting!r}; "
            f"choose from {ATTRIBUTE_WEIGHTING_METHODS}."
        )
    if weighting == "uniform":
        return {attr: 1.0 for attr in attribute_cols}
    source = validation_weights or {}
    return {
        attr: max(float(source.get(attr, 1.0)), EPS)
        for attr in attribute_cols
    }


def _valid_hard_values(
    predicted_values: Mapping[str, str],
    prototype_values: Mapping[str, str],
    attribute_cols: Sequence[str],
    attribute_weights: Mapping[str, float],
    ordinal_values: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(
        [ordinal_values.get(str(predicted_values.get(attr)), np.nan) for attr in attribute_cols],
        dtype=float,
    )
    prototype = np.asarray(
        [ordinal_values.get(str(prototype_values.get(attr)), np.nan) for attr in attribute_cols],
        dtype=float,
    )
    weights = np.asarray(
        [float(attribute_weights.get(attr, 1.0)) for attr in attribute_cols],
        dtype=float,
    )
    valid = ~(np.isnan(predicted) | np.isnan(prototype) | np.isnan(weights)) & (weights > 0.0)
    return predicted[valid], prototype[valid], weights[valid]


def match_prototype(
    *,
    predicted_values: Mapping[str, str],
    prototype_values: Mapping[str, str],
    attribute_cols: Sequence[str],
    metric: str,
    attribute_weights: Mapping[str, float],
    probability_distributions: Optional[Mapping[str, Mapping[str, float]]] = None,
    ordinal_values: Optional[Mapping[str, float]] = None,
) -> PrototypeMatch:
    """Match one sample to one candidate activity prototype."""
    if metric not in PROTOTYPE_METRICS:
        raise ValueError(f"Unknown prototype metric {metric!r}; choose from {PROTOTYPE_METRICS}.")

    if metric == "ordinal_probability":
        weighted_distance = 0.0
        weight_sum = 0.0
        distributions = probability_distributions or {}
        for attr in attribute_cols:
            distribution = distributions.get(attr)
            labels = set(distribution or {})
            labels.add(str(prototype_values.get(attr)))
            values = dict(ordinal_values or resolve_ordinal_values(labels))
            prototype = values.get(str(prototype_values.get(attr)))
            weight = float(attribute_weights.get(attr, 1.0))
            if prototype is None or not distribution or not np.isfinite(weight) or weight <= 0.0:
                continue
            expected_distance = sum(
                float(distribution.get(level, 0.0)) * abs(value - prototype)
                for level, value in values.items()
            )
            weighted_distance += weight * expected_distance
            weight_sum += weight
        if weight_sum <= EPS:
            return PrototypeMatch(score=-1.0, distance=float("inf"))
        distance = weighted_distance / weight_sum
        return PrototypeMatch(score=float(1.0 - distance), distance=float(distance))

    labels = [
        str(values.get(attr))
        for values in (predicted_values, prototype_values)
        for attr in attribute_cols
    ]
    values = dict(ordinal_values or resolve_ordinal_values(labels))
    predicted, prototype, weights = _valid_hard_values(
        predicted_values,
        prototype_values,
        attribute_cols,
        attribute_weights,
        values,
    )
    if not len(weights):
        return PrototypeMatch(score=-1.0, distance=float("inf"))

    if metric == "cosine":
        sqrt_weights = np.sqrt(weights)
        predicted_weighted = predicted * sqrt_weights
        prototype_weighted = prototype * sqrt_weights
        score = float(
            np.dot(predicted_weighted, prototype_weighted)
            / (
                np.linalg.norm(predicted_weighted)
                * np.linalg.norm(prototype_weighted)
                + EPS
            )
        )
        return PrototypeMatch(score=score, distance=float(1.0 - score))

    absolute_error = np.abs(predicted - prototype)
    if metric == "euclidean":
        distance = float(np.sqrt(np.sum(weights * absolute_error**2) / np.sum(weights)))
    else:
        distance = float(np.sum(weights * absolute_error) / np.sum(weights))
    return PrototypeMatch(score=float(1.0 - distance), distance=distance)


def rank_activity_prototypes(
    *,
    predicted_values: Mapping[str, str],
    activity_vectors: Mapping[str, Mapping[str, str]],
    attribute_cols: Sequence[str],
    metric: str,
    attribute_weights: Mapping[str, float],
    probability_distributions: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> list[tuple[str, PrototypeMatch]]:
    """Return activity prototypes sorted from best to worst."""
    ordinal_labels = {
        str(prototype.get(attr))
        for prototype in activity_vectors.values()
        for attr in attribute_cols
    }
    for distribution in (probability_distributions or {}).values():
        ordinal_labels.update(map(str, distribution))
    ordinal_values = resolve_ordinal_values(ordinal_labels)
    matches = [
        (
            activity,
            match_prototype(
                predicted_values=predicted_values,
                prototype_values=prototype,
                attribute_cols=attribute_cols,
                metric=metric,
                attribute_weights=attribute_weights,
                probability_distributions=probability_distributions,
                ordinal_values=ordinal_values,
            ),
        )
        for activity, prototype in activity_vectors.items()
    ]
    return sorted(matches, key=lambda item: item[1].score, reverse=True)
