from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from zshar.datasets import DatasetSpec
from zshar.paths import PROJECT_ROOT


@dataclass(frozen=True)
class ActivitySplit:
    index: int
    seen_ids: tuple[int, ...]
    unseen_ids: tuple[int, ...]
    seen_activities: tuple[str, ...]
    unseen_activities: tuple[str, ...]


@dataclass(frozen=True)
class SubjectSplit:
    """One strict train-subject/test-subject split resolved to window identities."""

    train_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    policy: str = "seen_all_unseen_balanced_split"


def _make_split(index: int, unseen_ids: Iterable[int], spec: DatasetSpec) -> ActivitySplit:
    unseen = tuple(sorted(int(item) for item in unseen_ids))
    seen = tuple(item for item in spec.activity_ids if item not in set(unseen))
    return ActivitySplit(
        index=int(index),
        seen_ids=seen,
        unseen_ids=unseen,
        seen_activities=tuple(spec.labels[item] for item in seen),
        unseen_activities=tuple(spec.labels[item] for item in unseen),
    )


def generate_activity_splits(
    spec: DatasetSpec,
    unseen_count: int | None = None,
    *,
    seed: int = 42,
    max_splits: int | None = None,
) -> list[ActivitySplit]:
    count = int(unseen_count or spec.unseen_count)
    combinations = list(enumerate(itertools.combinations(spec.activity_ids, count), start=1))
    if max_splits is not None and max_splits < len(combinations):
        combinations = sorted(random.Random(seed).sample(combinations, int(max_splits)))
    return [_make_split(index, unseen, spec) for index, unseen in combinations]


def load_split_manifest(path: str | Path, spec: DatasetSpec) -> list[ActivitySplit]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("splits", payload) if isinstance(payload, dict) else payload
    result = []
    for ordinal, row in enumerate(rows):
        index = int(row.get("split_index", ordinal))
        unseen_ids = row.get("unseen_ids")
        if unseen_ids is None:
            unseen_names = set(row["unseen_activities"])
            unseen_ids = [key for key, value in spec.labels.items() if value in unseen_names]
        result.append(_make_split(index, unseen_ids, spec))
    return result


def default_manifest_paths(
    spec: DatasetSpec,
    *,
    scenario: str,
    seed: int,
) -> tuple[Path, Path]:
    unseen = spec.unseen_count
    split_root = PROJECT_ROOT / "split"
    if scenario == "zsl":
        directory = split_root / "zero_shot_split"
        stem = f"{spec.key}_unseen{unseen}_random20"
        return (
            directory / f"{stem}_splits_seed{seed}.json",
            directory / f"{stem}_eval{spec.eval_samples_per_activity}_samples_seed{seed}.json",
        )
    directory = split_root / "gzsl"
    stem = f"{spec.key}_unseen{unseen}_random20_gzsl"
    split_path = directory / f"{stem}_splits_seed{seed}.json"
    candidates = sorted(directory.glob(f"{stem}_*samples_seed{seed}.json"))
    return split_path, (candidates[0] if candidates else directory / f"{stem}_samples_seed{seed}.json")


def default_subject_manifest_path(spec: DatasetSpec, *, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "split"
        / "subject_80_20"
        / f"{spec.key}_subject80_20_balanced_seed{seed}.json"
    )


def load_subject_manifest(
    path: str | Path,
    spec: DatasetSpec,
    *,
    policy: str = "seen_all_unseen_balanced_split",
) -> SubjectSplit:
    """Load and validate an index-explicit cross-subject manifest.

    The balanced policy is required because ``all_subject_split_80_20`` only
    records subject IDs and does not contain deterministic train/test windows.
    """
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Subject manifest must be a JSON object: {source}")
    manifest_dataset = str(payload.get("dataset", spec.key)).lower()
    if manifest_dataset != spec.key.lower():
        raise ValueError(
            f"Subject manifest dataset mismatch: expected {spec.key!r}, "
            f"found {manifest_dataset!r}."
        )
    row = payload.get(policy)
    if not isinstance(row, dict):
        raise ValueError(f"Subject manifest has no {policy!r} object: {source}")

    train_subjects = tuple(int(item) for item in row.get("train_subjects", []))
    test_subjects = tuple(int(item) for item in row.get("test_subjects", []))
    train_indices = tuple(int(item) for item in row.get("train_indices", []))
    test_indices = tuple(int(item) for item in row.get("test_indices", []))
    if not train_subjects or not test_subjects:
        raise ValueError("Cross-subject manifest needs non-empty train_subjects and test_subjects.")
    if not train_indices or not test_indices:
        raise ValueError(
            f"Cross-subject policy {policy!r} needs explicit train_indices and test_indices."
        )
    if set(train_subjects) & set(test_subjects):
        raise ValueError("Cross-subject train_subjects and test_subjects overlap.")
    if set(train_indices) & set(test_indices):
        raise ValueError("Cross-subject train_indices and test_indices overlap.")
    if len(set(train_indices)) != len(train_indices) or len(set(test_indices)) != len(test_indices):
        raise ValueError("Cross-subject manifest contains duplicate window indices.")
    return SubjectSplit(
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        train_indices=train_indices,
        test_indices=test_indices,
        policy=policy,
    )


def load_sample_manifest(path: str | Path) -> dict[int, set[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("samples", payload)
    result: dict[int, set[int]] = {}
    for split_key, activities in raw.items():
        ids: set[int] = set()
        for values in activities.values():
            ids.update(int(item) for item in values)
        result[int(split_key)] = ids
    return result


def sample_evaluation_rows(
    frame: pd.DataFrame,
    split: ActivitySplit,
    *,
    scenario: str,
    samples_per_activity: int,
    seed: int,
    manifest_ids: set[int] | None = None,
) -> pd.DataFrame:
    candidate_names = split.unseen_activities if scenario == "zsl" else (
        split.seen_activities + split.unseen_activities
    )
    candidates = frame[frame["activity_name"].isin(candidate_names)]
    if manifest_ids is not None:
        return candidates[candidates["window_index"].isin(manifest_ids)].copy()
    sampled = []
    for offset, (_, group) in enumerate(candidates.groupby("activity_name", sort=False)):
        count = min(int(samples_per_activity), len(group))
        sampled.append(group.sample(n=count, random_state=seed + split.index * 1009 + offset))
    return pd.concat(sampled, ignore_index=True) if sampled else candidates.iloc[0:0].copy()


def select_splits(splits: list[ActivitySplit], indices: set[int] | None) -> list[ActivitySplit]:
    return splits if not indices else [item for item in splits if item.index in indices]
