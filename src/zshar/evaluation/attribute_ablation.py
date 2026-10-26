"""Frozen-output, leave-one-attribute-out ordinal prototype matching."""
from __future__ import annotations

import numpy as np


def ordinal_distance_components(probabilities, prototypes, native_class_orders=None):
    """Return (N,C,A) distances using the fixed Low/Medium/High scale.

    Raw probabilities have shape (N,A,3). Normalize in each predictor's native
    class-column order, as probability_row() does in the original evaluator.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prototypes = np.asarray(prototypes, dtype=np.int64)
    if probabilities.ndim != 3 or probabilities.shape[2] != 3:
        raise ValueError("Expected probabilities (N,A,3)")
    if prototypes.ndim != 2 or prototypes.shape[1] != probabilities.shape[1]:
        raise ValueError("Prototype attribute dimensions differ")
    if not np.isfinite(probabilities).all() or (probabilities < -1e-6).any() or (probabilities > 1+1e-6).any():
        raise ValueError("Invalid probabilities")
    if ((prototypes < 0) | (prototypes > 2)).any():
        raise ValueError("Expected prototype ordinals 0,1,2")
    values = np.maximum(probabilities, 0.)
    totals = np.zeros(values.shape[:2], dtype=float)
    orders = native_class_orders or [list(range(3)) for _ in range(values.shape[1])]
    if len(orders) != values.shape[1]:
        raise ValueError("One native class order is required per attribute")
    for attribute, order in enumerate(orders):
        if not order or len(set(order)) != len(order) or set(order)-{0,1,2}:
            raise ValueError("Invalid native class order")
        omitted = list({0,1,2}-set(order))
        if omitted and np.any(values[:, attribute, omitted] != 0):
            raise ValueError("Nonzero probabilities outside native class schema")
        for level in order:
            totals[:, attribute] += values[:, attribute, level]
    if (totals <= 0).any() or not np.allclose(totals, 1., atol=1e-5, rtol=0):
        raise ValueError("Probability rows must sum to one")
    normalized = values / totals[:, :, None]
    distances = np.zeros((len(values), len(prototypes), values.shape[1]))
    for level in range(3):
        distances += normalized[:, None, :, level] * np.abs(level-prototypes[None, :, :]) / 2.
    return distances


def ordinal_scores(components, weights, removed=None):
    """Remove one coordinate from BOTH distance terms and weight denominator."""
    components = np.asarray(components, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if components.ndim != 3 or components.shape[2] != len(weights):
        raise ValueError("Mismatched distance and weight dimensions")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Weights must be finite and positive")
    if removed is not None and not 0 <= int(removed) < len(weights):
        raise ValueError("Invalid removed attribute")
    numerator = np.zeros(components.shape[:2])
    denominator = 0.
    for attribute, weight in enumerate(weights):
        if attribute == removed:
            continue
        numerator += weight * components[:, :, attribute]
        denominator += float(weight)
    if denominator <= 0:
        raise ValueError("At least one attribute must remain")
    return 1. - numerator / denominator


def candidate_predictions(scores, candidates):
    """Preserve the original top_predictions() candidate order and tie policy."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or not scores.shape[1] or scores.shape[1] != len(candidates):
        raise ValueError("Missing or mismatched candidates")
    return np.asarray(candidates, dtype=str)[np.argsort(-scores, axis=1)[:, 0]]
