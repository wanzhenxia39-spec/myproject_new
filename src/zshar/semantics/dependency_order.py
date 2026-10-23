"""Directional dependency ordering for categorical semantic attributes."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


def categorical_entropy(values: Sequence[Any]) -> float:
    """Return the empirical base-2 entropy of a nonempty categorical vector."""
    labels = [str(value) for value in values]
    if not labels:
        raise ValueError("Entropy requires at least one observation")
    total = float(len(labels))
    return float(
        -sum((count / total) * math.log2(count / total) for count in Counter(labels).values())
    )


def conditional_entropy(target: Sequence[Any], given: Sequence[Any]) -> float:
    """Return empirical H(target | given), with one equally weighted row per item."""
    target_labels = [str(value) for value in target]
    given_labels = [str(value) for value in given]
    if not target_labels or len(target_labels) != len(given_labels):
        raise ValueError("target and given must be nonempty vectors of equal length")
    grouped: dict[str, list[str]] = {}
    for condition, value in zip(given_labels, target_labels):
        grouped.setdefault(condition, []).append(value)
    total = float(len(target_labels))
    return float(
        sum((len(values) / total) * categorical_entropy(values) for values in grouped.values())
    )


def directional_dependency_matrix(
    prototypes: pd.DataFrame,
    *,
    zero_entropy_value: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute D[i,j] = 1 - H(A_j | A_i) / H(A_j).

    Rows must be seen-activity semantic prototypes and columns must be distinct
    categorical attributes. A constant target has no uncertainty to reduce, so
    its incoming dependency scores are defined as ``zero_entropy_value``.
    Self-dependencies are always zero because they are excluded from an order.
    """
    if not isinstance(prototypes, pd.DataFrame) or prototypes.empty:
        raise ValueError("prototypes must be a nonempty pandas DataFrame")
    attributes = [str(column) for column in prototypes.columns]
    if not attributes or len(attributes) != len(set(attributes)):
        raise ValueError("prototype attribute names must be nonempty and unique")
    if prototypes.isna().any().any():
        raise ValueError("semantic prototypes cannot contain missing values")
    if not 0.0 <= float(zero_entropy_value) <= 1.0:
        raise ValueError("zero_entropy_value must be in [0, 1]")

    values = prototypes.astype(str)
    entropies = {
        attribute: categorical_entropy(values[attribute].tolist())
        for attribute in attributes
    }
    matrix = np.zeros((len(attributes), len(attributes)), dtype=float)
    for source_index, source in enumerate(attributes):
        for target_index, target in enumerate(attributes):
            if source_index == target_index:
                continue
            target_entropy = entropies[target]
            if target_entropy <= 1e-15:
                score = float(zero_entropy_value)
            else:
                remaining = conditional_entropy(
                    values[target].tolist(), values[source].tolist()
                )
                score = 1.0 - remaining / target_entropy
            matrix[source_index, target_index] = float(np.clip(score, 0.0, 1.0))
    return matrix, entropies


def order_objective(order: Sequence[int], dependencies: np.ndarray) -> float:
    """Sum all directional dependencies that point forward in ``order``."""
    indices = [int(index) for index in order]
    size = int(dependencies.shape[0])
    if dependencies.shape != (size, size):
        raise ValueError("dependencies must be a square matrix")
    if len(indices) != size or set(indices) != set(range(size)):
        raise ValueError("order must contain each dependency-matrix index exactly once")
    return float(
        sum(
            dependencies[indices[left], indices[right]]
            for left in range(size)
            for right in range(left + 1, size)
        )
    )


def optimal_dependency_order(
    dependencies: np.ndarray,
    *,
    max_attributes: int = 20,
    tie_tolerance: float = 1e-12,
) -> tuple[list[int], float]:
    """Maximize the forward-edge objective with subset dynamic programming.

    ``dp[S]`` stores the best order of subset ``S``. Adding attribute ``j`` as
    the last item contributes ``sum(D[i,j] for i in S - {j})``. Equal-score
    solutions are resolved lexicographically by the original schema indices.
    """
    matrix = np.asarray(dependencies, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.shape[0]:
        raise ValueError("dependencies must be a nonempty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("dependencies must contain only finite values")
    size = int(matrix.shape[0])
    if size > int(max_attributes):
        raise ValueError(
            f"Subset DP is limited to {max_attributes} attributes; received {size}"
        )

    state_count = 1 << size
    scores = np.full(state_count, -np.inf, dtype=float)
    orders: list[tuple[int, ...] | None] = [None] * state_count
    scores[0] = 0.0
    orders[0] = ()
    for mask in range(1, state_count):
        best_score = -np.inf
        best_order: tuple[int, ...] | None = None
        for last in range(size):
            last_bit = 1 << last
            if not mask & last_bit:
                continue
            previous = mask ^ last_bit
            prefix = orders[previous]
            if prefix is None:
                continue
            gain = sum(
                matrix[source, last]
                for source in range(size)
                if previous & (1 << source)
            )
            candidate_score = float(scores[previous] + gain)
            candidate_order = prefix + (last,)
            if (
                candidate_score > best_score + tie_tolerance
                or (
                    abs(candidate_score - best_score) <= tie_tolerance
                    and (best_order is None or candidate_order < best_order)
                )
            ):
                best_score = candidate_score
                best_order = candidate_order
        scores[mask] = best_score
        orders[mask] = best_order

    optimum = orders[-1]
    if optimum is None:
        raise RuntimeError("Dependency-order dynamic programming produced no solution")
    objective = order_objective(optimum, matrix)
    if not np.isclose(objective, scores[-1], atol=tie_tolerance, rtol=0.0):
        raise AssertionError("Dynamic-programming objective reconstruction mismatch")
    return list(optimum), objective


def semantic_dependency_order(prototypes: pd.DataFrame) -> dict[str, Any]:
    """Return matrix diagnostics and optimal/reverse attribute name orders."""
    attributes = [str(column) for column in prototypes.columns]
    matrix, entropies = directional_dependency_matrix(prototypes)
    indices, objective = optimal_dependency_order(matrix)
    forward = [attributes[index] for index in indices]
    reverse = forward[::-1]
    reverse_indices = list(reversed(indices))
    return {
        "attributes": attributes,
        "seen_prototype_count": int(len(prototypes)),
        "entropies": entropies,
        "dependency_matrix": {
            source: {
                target: float(matrix[source_index, target_index])
                for target_index, target in enumerate(attributes)
            }
            for source_index, source in enumerate(attributes)
        },
        "forward": forward,
        "reverse": reverse,
        "forward_objective": objective,
        "reverse_objective": order_objective(reverse_indices, matrix),
    }
