from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zshar.paths import PROJECT_ROOT, resolve_project_path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    labels: dict[int, str]
    features_path: Path
    feature_names_path: Path
    labels_path: Path
    attributes_path: Path
    metadata_path: Path
    unseen_count: int
    eval_samples_per_activity: int
    raw: dict[str, Any]

    @property
    def activity_ids(self) -> list[int]:
        return list(self.labels)

    @property
    def activity_names(self) -> list[str]:
        return list(self.labels.values())


def _profile_dir() -> Path:
    return PROJECT_ROOT / "configs" / "datasets"


def available_datasets() -> list[str]:
    return sorted(path.stem for path in _profile_dir().glob("*.json"))


def load_dataset_spec(name: str, config_path: str | Path | None = None) -> DatasetSpec:
    path = resolve_project_path(config_path) if config_path else _profile_dir() / f"{name.lower()}.json"
    if not path.exists():
        raise ValueError(
            f"Unknown dataset {name!r}. Available datasets: {', '.join(available_datasets())}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload["files"]
    labels = {int(key): str(value) for key, value in payload["labels"].items()}
    return DatasetSpec(
        key=str(payload["key"]),
        name=str(payload.get("name", payload["key"])),
        labels=labels,
        features_path=resolve_project_path(files["features"]),
        feature_names_path=resolve_project_path(files["feature_names"]),
        labels_path=resolve_project_path(files["labels"]),
        attributes_path=resolve_project_path(payload["attributes"]),
        metadata_path=resolve_project_path(payload["metadata"]),
        unseen_count=int(payload["defaults"]["unseen_count"]),
        eval_samples_per_activity=int(payload["defaults"]["eval_samples_per_activity"]),
        raw=payload,
    )

