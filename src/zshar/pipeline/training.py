from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from zshar.models import create_model_manager, load_model_manager


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def split_train_validation(frame: pd.DataFrame, *, seed: int, validation_fraction: float = 0.3):
    counts = frame["activity_name"].value_counts()
    stratify = frame["activity_name"] if (counts >= 2).all() else None
    return train_test_split(
        frame,
        test_size=float(validation_fraction),
        random_state=seed,
        stratify=stratify,
    )


def train_attribute_models(
    frame: pd.DataFrame,
    feature_columns: list[str],
    attribute_columns: list[str],
    model_path: str | Path,
    *,
    backend: str,
    backend_config: dict[str, Any],
    seed: int,
    validation_fraction: float = 0.3,
    label_count_threshold: int = 1,
):
    set_seed(seed)
    train, validation = split_train_validation(
        frame, seed=seed, validation_fraction=validation_fraction
    )
    manager = create_model_manager(backend, attribute_columns, backend_config)
    report = manager.fit_all(
        train[feature_columns],
        train[attribute_columns].astype(str),
        validation[feature_columns],
        validation[attribute_columns].astype(str),
        root_path=str(model_path),
        label_count_threshold=label_count_threshold,
    )
    return manager, report


def load_or_train_attribute_models(
    *,
    reuse_models: bool,
    model_path: str | Path,
    frame: pd.DataFrame,
    feature_columns: list[str],
    attribute_columns: list[str],
    backend: str,
    backend_config: dict[str, Any],
    seed: int,
    evaluate_only: bool = False,
):
    path = Path(model_path)
    if evaluate_only and not path.exists():
        raise FileNotFoundError(
            f"Evaluation-only mode requires an existing model directory: {path}"
        )
    if (reuse_models or evaluate_only) and path.exists():
        manager = load_model_manager(str(path), expected_backend=backend, attribute_cols=attribute_columns)
        requested_features = dict(backend_config.get("attribute_feature_columns") or {})
        saved_features = dict(getattr(manager, "attribute_feature_columns", {}) or {})
        if requested_features != saved_features:
            raise ValueError(
                "Saved model feature bindings differ from the requested feature bindings. "
                "Use the original bindings or a new run name to train new models."
            )
        report = dict(getattr(manager, "validation_report", {}))
        return manager, report
    if path.exists():
        raise FileExistsError(
            f"Model directory already exists: {path}. Use --reuse-models or choose a new run name."
        )
    return train_attribute_models(
        frame,
        feature_columns,
        attribute_columns,
        path,
        backend=backend,
        backend_config=backend_config,
        seed=seed,
    )


def parse_backend_config(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    stripped = raw.strip()
    if stripped.startswith("{"):
        payload = json.loads(stripped)
    else:
        path = Path(raw)
        payload = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else json.loads(raw)
        )
    if not isinstance(payload, dict):
        raise ValueError("Backend configuration must be a JSON object")
    return payload
