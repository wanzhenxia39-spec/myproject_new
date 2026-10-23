from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from zshar.datasets import DatasetSpec


@dataclass
class DatasetTable:
    frame: pd.DataFrame
    feature_columns: list[str]
    attribute_columns: list[str]
    activity_vectors: dict[str, dict[str, str]]


def load_activity_vectors(path: str | Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Attribute vector file is empty or invalid: {path}")
    vectors = {str(activity): dict(values) for activity, values in payload.items()}
    columns = list(next(iter(vectors.values())))
    if any(list(values) != columns for values in vectors.values()):
        raise ValueError(f"Activities do not share one attribute schema: {path}")
    return vectors, columns


def load_dataset_table(
    spec: DatasetSpec,
    activity_ids: Iterable[int] | None = None,
    *,
    mmap: bool = True,
) -> DatasetTable:
    for path in (spec.features_path, spec.feature_names_path, spec.labels_path, spec.attributes_path):
        if not path.exists():
            raise FileNotFoundError(path)
    mmap_mode = "r" if mmap else None
    features = np.load(spec.features_path, mmap_mode=mmap_mode)
    feature_names = np.load(spec.feature_names_path, allow_pickle=True)
    labels = np.load(spec.labels_path, mmap_mode=mmap_mode)
    if features.ndim != 2:
        raise ValueError(f"Expected features (N,D), got {features.shape}")
    if labels.ndim != 3 or labels.shape[2] < 2:
        raise ValueError(f"Expected labels (N,T,>=2), got {labels.shape}")
    if len(features) != len(labels) or features.shape[1] != len(feature_names):
        raise ValueError("Feature, feature-name and label dimensions do not match")

    vectors, attribute_columns = load_activity_vectors(spec.attributes_path)
    wanted = set(spec.activity_ids if activity_ids is None else (int(item) for item in activity_ids))
    activity = np.asarray(labels[:, 0, 0], dtype=int)
    user = np.asarray(labels[:, 0, 1], dtype=int)
    mask = np.isin(activity, list(wanted))
    indices = np.flatnonzero(mask)
    names = np.asarray([spec.labels.get(int(item), "") for item in activity[indices]], dtype=object)
    valid = np.asarray([name in vectors for name in names], dtype=bool)
    indices, names = indices[valid], names[valid]

    feature_columns = [str(item) for item in feature_names]
    frame = pd.DataFrame(np.asarray(features[indices]), columns=feature_columns)
    frame.insert(0, "user_id", user[indices])
    frame.insert(0, "activity_name", names)
    frame.insert(0, "activity_id", activity[indices])
    frame.insert(0, "window_index", indices)
    for attribute in attribute_columns:
        frame[attribute] = [vectors[name][attribute] for name in names]
    return DatasetTable(frame, feature_columns, attribute_columns, vectors)


def replace_activity_vectors(
    table: DatasetTable,
    path: str | Path,
) -> DatasetTable:
    """Return a dataset table labeled with one split-specific vector schema."""
    vectors, attribute_columns = load_activity_vectors(path)
    activities = set(table.frame["activity_name"].astype(str))
    missing = sorted(activities.difference(vectors))
    if missing:
        raise ValueError(
            f"Split-specific attribute vectors omit dataset activities {missing}: {path}"
        )
    frame = table.frame.drop(columns=table.attribute_columns, errors="ignore").copy()
    activity_names = frame["activity_name"].astype(str)
    for attribute in attribute_columns:
        frame[attribute] = [vectors[name][attribute] for name in activity_names]
    return DatasetTable(
        frame=frame,
        feature_columns=list(table.feature_columns),
        attribute_columns=attribute_columns,
        activity_vectors=vectors,
    )
