"""Attribute quality on held-out IMU windows, before activity matching."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LEVELS = ("Low", "Medium", "High")


def align_probabilities(probabilities: pd.DataFrame, levels=LEVELS) -> pd.DataFrame:
    """Align by label name; unlearned states get zero mass, not new ordinal ranks."""
    table = probabilities.copy()
    table.columns = table.columns.astype(str)
    if table.columns.duplicated().any() or set(table.columns) - set(levels):
        raise ValueError("Invalid probability class labels")
    table = table.reindex(columns=list(levels), fill_value=0.0).astype(float)
    values = table.to_numpy()
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Probabilities must be nonempty and finite")
    # Float32 weighted ensembles can overshoot 1 by one ULP (1.19e-7).
    # Reject material violations; repair only the affected roundoff rows.
    if (values < -1e-6).any() or (values > 1 + 1e-6).any():
        raise ValueError("Probabilities outside [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Probability rows must sum to one")
    affected = ((values < 0) | (values > 1)).any(axis=1)
    if affected.any():
        repaired = np.clip(values[affected], 0, 1)
        repaired /= repaired.sum(axis=1, keepdims=True)
        table.iloc[affected] = repaired
    return table


def score_attribute(truth, prediction, probabilities: pd.DataFrame, levels=LEVELS):
    """Use sample-mean EOD with the full ontology denominator L-1.

    macro_f1 follows the existing project: average over true/predicted states.
    macro_f1_all_levels also reports the fixed three-state alternative.
    """
    truth = np.asarray(truth, dtype=str)
    prediction = np.asarray(prediction, dtype=str)
    if len(truth) != len(prediction) or len(truth) != len(probabilities):
        raise ValueError("Truth, predictions and probability rows do not align")
    if len(levels) < 2 or not set(truth).issubset(levels) or not set(prediction).issubset(levels):
        raise ValueError("Unknown ordinal label or invalid ontology")
    table = align_probabilities(probabilities, levels)
    ordinal = {label: rank for rank, label in enumerate(levels)}
    target = np.array([ordinal[label] for label in truth])
    distances = np.abs(np.arange(len(levels))[None, :] - target[:, None]) / (len(levels) - 1)
    per_sample_eod = np.sum(table.to_numpy() * distances, axis=1)
    per_state = f1_score(truth, prediction, labels=list(levels), average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "macro_f1_all_levels": float(per_state.mean()),
        "eod": float(per_sample_eod.mean()),
        "accuracy": float(np.mean(truth == prediction)),
        "validation_state_count": int(len(set(truth))),
        "validation_state_support": {level: int(np.sum(truth == level)) for level in levels},
        "per_state_f1": dict(zip(levels, map(float, per_state))),
    }, per_sample_eod


def audit_partition(fit_indices, tuning_indices, train_indices, validation_indices):
    """AutoGluon inner fit/tuning rows must partition only the outer train set."""
    fit, tuning, train, validation = map(set, (fit_indices, tuning_indices, train_indices, validation_indices))
    if fit & tuning or train & validation or (fit | tuning) & validation:
        raise ValueError("Held-out validation overlaps fitting or tuning rows")
    if fit | tuning != train:
        raise ValueError("Saved AutoGluon rows do not match the reconstructed Seen training split")
    return {"fit_samples": len(fit), "tuning_samples": len(tuning),
            "outer_train_samples": len(train), "outer_validation_samples": len(validation),
            "outer_overlap": 0, "outer_validation_used_for_autogluon_selection": False}


def aggregate_matrices(rows: pd.DataFrame, attributes, datasets, metric):
    """Equal weight per split; ddof=1 is descriptive variation across class splits."""
    if rows.duplicated(["dataset", "split_index", "attribute"]).any():
        raise ValueError("Duplicate split/attribute metrics")
    grouped = rows.groupby(["attribute", "dataset"])[metric]
    def matrix(series):
        return series.unstack("dataset").reindex(index=attributes, columns=datasets)
    return matrix(grouped.mean()), matrix(grouped.std(ddof=1)), matrix(grouped.count())
