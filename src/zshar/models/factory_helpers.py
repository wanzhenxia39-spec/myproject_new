"""Shared helpers used by model-manager implementations."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .base import ORDINAL_VALUE_MAP


def sanitize_model_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return safe or "attribute"


def choose_problem_type(y: pd.Series, label_count_threshold: int = 1):
    counts = y.dropna().astype(str).value_counts().to_dict()
    if not counts:
        raise ValueError("Label column has no valid classes after dropping NaN.")
    if len(counts) == 1:
        return "constant", counts
    below = {
        label: count for label, count in counts.items() if count < label_count_threshold
    }
    if below:
        raise ValueError(
            f"Some classes have fewer than label_count_threshold={label_count_threshold}: {below}"
        )
    return ("binary" if len(counts) == 2 else "multiclass"), counts


def overall_attribute_metrics(
    truth: pd.DataFrame, prediction: pd.DataFrame
) -> tuple[float, float]:
    exact = float((truth.astype(str) == prediction.astype(str)).all(axis=1).mean())
    true_numeric = (
        truth.astype(str)
        .apply(lambda column: column.map(ORDINAL_VALUE_MAP))
        .astype(float)
        .to_numpy()
    )
    pred_numeric = (
        prediction.astype(str)
        .apply(lambda column: column.map(ORDINAL_VALUE_MAP))
        .astype(float)
        .to_numpy()
    )
    cosine = np.sum(true_numeric * pred_numeric, axis=1) / (
        np.linalg.norm(true_numeric, axis=1)
        * np.linalg.norm(pred_numeric, axis=1)
        + 1e-10
    )
    return exact, float(np.nanmean(cosine))
