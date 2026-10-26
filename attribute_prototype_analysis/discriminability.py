"""Coverage and combined prototype diagnostics; distance/APCR live in standalone files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from exact_apcr import compute_exact_apcr
from attribute_discrimination_coverage import compute_attribute_coverage
from prototype_distance_heatmap import (
    DEFAULT_LEVEL_MAP, PrototypeTable, load_prototypes, encode_prototypes,
    pairwise_ordinal_distances, plot_blue_heatmap,
)

def analyze_prototypes(table: PrototypeTable) -> dict:
    """Compute d(i,j), R_k, single-attribute equality, and full-vector exact APCR.

    Exact APCR uses equality of ALL coordinates and is not the average of
    per-attribute equality rates. Equality comparisons use no tolerance.
    """
    values = table.values
    if (values.shape != (len(table.activities), len(table.attributes)) or
            values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1 or
            not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()):
        raise ValueError("Expected a finite normalized C-by-K matrix, C>=2, K>=1")
    left, right = np.triu_indices(len(table.activities), k=1)
    absolute_difference = np.abs(values[left] - values[right])
    different = values[left] != values[right]
    pair_count = len(left)
    attribute_stats = pd.DataFrame(compute_attribute_coverage({
        activity: dict(zip(table.attributes, values[i])) for i, activity in enumerate(table.activities)
    }))
    coverage = attribute_stats["coverage"].to_numpy()
    # Exact APCR has one implementation, in the standalone exact_apcr.py.
    apcr = compute_exact_apcr({activity: dict(zip(table.attributes, values[i]))
                               for i, activity in enumerate(table.activities)})
    colliding_pairs = {(pair["activity_i"], pair["activity_j"]) for pair in apcr["collision_pairs"]}
    collision = np.asarray([(table.activities[i], table.activities[j]) in colliding_pairs
                            for i, j in zip(left, right)])
    distances = pairwise_ordinal_distances(table).to_numpy()
    pair_stats = pd.DataFrame({
        "activity_i": [table.activities[i] for i in left],
        "activity_j": [table.activities[j] for j in right],
        "normalized_ordinal_l1": absolute_difference.mean(axis=1),
        "different_attributes": different.sum(axis=1),
        "exact_collision": collision.astype(int),
    })
    return {
        "distance_matrix": pd.DataFrame(distances, index=table.activities, columns=table.activities),
        "attribute_metrics": attribute_stats,
        "pair_metrics": pair_stats,
        "summary": {
            "class_count": len(table.activities), "attribute_count": len(table.attributes),
            "total_pairs": pair_count, "exact_collision_pairs": apcr["exact_collision_pairs"],
            "exact_apcr": apcr["exact_apcr"],
            "mean_attribute_coverage": float(coverage.mean()),
            "mean_single_attribute_collision_rate": float((~different).mean()),
            "mean_pairwise_distance": float(absolute_difference.mean()),
        },
    }

