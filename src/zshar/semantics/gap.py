from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from zshar.models.base import ORDINAL_VALUE_MAP
from zshar.splits import ActivitySplit

GAP_DISTANCE_METRICS = ("normalized_ordinal", "mad", "euclidean")
DEFAULT_GAP_DISTANCE = "normalized_ordinal"


def normalized_ordinal_prototype_distance(
    left: Sequence[float], right: Sequence[float],
) -> float:
    """Equal-weight (1/K) sum_k |a_ik-a_jk|/2, with L/M/H encoded as 0/1/2.

    The fixed ordinal span is 2; it is never inferred from observed classes.
    Averaging over the current K makes A1 and A2 with different sizes comparable.
    """
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.ndim != 1 or not a.size or a.shape != b.shape:
        raise ValueError("Prototypes must be nonempty one-dimensional vectors of equal length")
    if not np.isin(a, [0, 1, 2]).all() or not np.isin(b, [0, 1, 2]).all():
        raise ValueError("Normalized ordinal distance requires ordinal levels 0, 1, 2")
    return float(np.mean(np.abs(a - b) / 2.0))


def vectorize_prototypes(
    activity_vectors: Mapping[str, Mapping[str, str]],
) -> tuple[list[str], dict[str, np.ndarray]]:
    if not activity_vectors:
        raise ValueError("Activity prototypes must not be empty")
    attributes = list(next(iter(activity_vectors.values())))
    if not attributes or any(set(values) != set(attributes) for values in activity_vectors.values()):
        raise ValueError("Activity prototypes must share the same nonempty attribute set")
    vectors = {
        activity: np.asarray([ORDINAL_VALUE_MAP[str(values[attr])] for attr in attributes], dtype=float)
        for activity, values in activity_vectors.items()
    }
    return attributes, vectors


def attribute_distance(
    left: Sequence[float],
    right: Sequence[float],
    weights: Sequence[float],
    *,
    metric: str = DEFAULT_GAP_DISTANCE,
) -> float:
    if metric == "normalized_ordinal":
        # vectorize_prototypes uses 0/.5/1, equivalent to raw 0/1/2 divided by 2.
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        w = np.asarray(weights, dtype=float)
        if w.shape != a.shape or not w.size or not np.isfinite(w).all() or (w <= 0).any():
            raise ValueError("Normalized ordinal distance requires positive uniform weights")
        if not np.all(w == w[0]):
            raise ValueError("Normalized ordinal distance uses equal weights, not classifier reliability")
        return normalized_ordinal_prototype_distance(a * 2, b * 2)
    differences = np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
    normalized_weights = np.asarray(weights, dtype=float)
    normalized_weights = normalized_weights / normalized_weights.sum()
    if metric == "mad":
        return float(np.sum(normalized_weights * differences))
    if metric == "euclidean":
        return float(math.sqrt(np.sum(normalized_weights * differences**2)))
    raise ValueError(f"metric must be one of {GAP_DISTANCE_METRICS}")


def d_knn(
    vectors: Mapping[str, Sequence[float]],
    seen: Sequence[str],
    unseen: Sequence[str],
    weights: Sequence[float],
    *,
    k: int = 2,
    metric: str = DEFAULT_GAP_DISTANCE,
) -> tuple[float, dict[str, float]]:
    if k <= 0 or not seen:
        raise ValueError("k and the seen activity set must be non-empty")
    per_unseen = {}
    for unseen_activity in unseen:
        distances = sorted(
            attribute_distance(vectors[unseen_activity], vectors[item], weights, metric=metric)
            for item in seen
        )
        per_unseen[unseen_activity] = float(np.mean(distances[: min(k, len(distances))]))
    return float(np.mean(list(per_unseen.values()))), per_unseen


def unseen_confusion(
    vectors: Mapping[str, Sequence[float]],
    unseen: Sequence[str],
    weights: Sequence[float],
    *,
    metric: str = DEFAULT_GAP_DISTANCE,
) -> tuple[float, float, dict[str, float]]:
    if len(unseen) < 2:
        return float("nan"), float("nan"), {}
    nearest = {}
    for activity in unseen:
        nearest[activity] = min(
            attribute_distance(vectors[activity], vectors[other], weights, metric=metric)
            for other in unseen
            if other != activity
        )
    u_nn = float(np.mean(list(nearest.values())))
    return u_nn, 1.0 - u_nn, nearest


def compute_gap(
    split: ActivitySplit,
    activity_vectors: Mapping[str, Mapping[str, str]],
    *,
    k: int = 2,
    metric: str = DEFAULT_GAP_DISTANCE,
    attribute_weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    attributes, vectors = vectorize_prototypes(activity_vectors)
    weights = [float((attribute_weights or {}).get(attr, 1.0)) for attr in attributes]
    coverage, coverage_per_activity = d_knn(
        vectors, split.seen_activities, split.unseen_activities, weights, k=k, metric=metric
    )
    u_nn, confusion, confusion_per_activity = unseen_confusion(
        vectors, split.unseen_activities, weights, metric=metric
    )
    joint = (coverage + confusion) / 2.0
    return {
        "split_index": split.index,
        "seen_activities": list(split.seen_activities),
        "unseen_activities": list(split.unseen_activities),
        "D_kNN": coverage,
        "U_NN": u_nn,
        "U_Conf": confusion,
        "JointGap_DkNN_UConf": joint,
        "D_kNN_per_unseen": coverage_per_activity,
        "U_NN_per_unseen": confusion_per_activity,
    }


def assign_difficulty(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if len(frame) < 3:
        frame["difficulty"] = "Unranked"
        return frame
    frame["difficulty"] = pd.qcut(
        frame["JointGap_DkNN_UConf"].rank(method="first"),
        q=3,
        labels=["Easy", "Medium", "Hard"],
    )
    return frame
