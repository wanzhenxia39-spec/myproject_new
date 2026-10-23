"""Re-extract the complete MHEALTH feature matrix from every saved window.

This command deliberately reuses ``data_<version>.npy``.  It neither reads raw
MHEALTH logs nor changes window boundaries or labels.  Results are built in a
partial NPY file, validated, and atomically installed only after all windows
have succeeded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from time import monotonic

import numpy as np
from numpy.lib.format import open_memmap

from zshar.datasets import load_dataset_spec
from zshar.features.all_features import extract_features
from zshar.features.groups import expand_feature_groups, resolve_feature_groups
from zshar.preprocessing.preprocess_mhealth import window_to_sensor_dict


EXPECTED_CHANNELS = 15
SENSOR_PREFIXES = ("hand", "chest", "ankle")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(data: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _extract_one(window: np.ndarray, fs: float) -> dict[str, float]:
    return extract_features(
        window_to_sensor_dict(window), fs=fs, channel_names=SENSOR_PREFIXES
    )


def _validate_inputs(windows: np.ndarray, labels_path: Path | None) -> None:
    if windows.ndim != 3 or windows.shape[1] < 13 or windows.shape[2] != EXPECTED_CHANNELS:
        raise ValueError(
            "MHEALTH windows must have shape (N, T, 15), with T > 12; "
            f"got {windows.shape}"
        )
    if windows.shape[0] == 0:
        raise ValueError("MHEALTH window array is empty")
    if not np.isfinite(windows).all():
        raise ValueError("MHEALTH windows contain NaN or infinity")
    if labels_path is not None:
        if not labels_path.is_file():
            raise FileNotFoundError(f"Label file does not exist: {labels_path}")
        labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        if labels.shape[0] != windows.shape[0]:
            raise ValueError(
                f"Window/label count mismatch: {windows.shape[0]} != {labels.shape[0]}"
            )


def _expected_schema(first_window: np.ndarray, fs: float) -> tuple[list[str], str]:
    first = _extract_one(first_window, fs)
    names = sorted(first)
    configured = expand_feature_groups("mhealth")["feature_names"]
    if set(names) != set(configured) or len(names) != len(configured):
        missing = sorted(set(configured) - set(names))
        extra = sorted(set(names) - set(configured))
        raise ValueError(
            "Extractor output does not match the configured MHEALTH G1-G14 schema: "
            f"extractor={len(names)}, configured={len(configured)}, "
            f"missing={missing}, extra={extra}"
        )
    resolve_feature_groups(names, dataset="mhealth")
    schema_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return names, schema_hash


def _backup_incomplete(path: Path, feature_count: int) -> Path | None:
    if not path.exists():
        return None
    existing = np.load(path, mmap_mode="r", allow_pickle=False)
    if existing.ndim == 2 and existing.shape[1] == feature_count:
        return None
    columns = existing.shape[1] if existing.ndim == 2 else "invalid"
    backup = path.with_name(f"{path.stem}.incomplete_{columns}_backup{path.suffix}")
    if backup.exists():
        if _sha256(backup) != _sha256(path):
            raise FileExistsError(f"Refusing to replace a different backup: {backup}")
    else:
        shutil.copy2(path, backup)
    return backup


def reextract_all_windows(
    *,
    windows_path: str | Path,
    features_path: str | Path,
    feature_names_path: str | Path,
    labels_path: str | Path | None = None,
    fs: float = 20.0,
    checkpoint_every: int = 50,
    resume: bool = False,
) -> dict:
    """Extract and install a 434-column vector for every MHEALTH window."""
    if fs <= 0 or checkpoint_every <= 0:
        raise ValueError("fs and checkpoint_every must be positive")
    windows_path = Path(windows_path).expanduser().resolve()
    features_path = Path(features_path).expanduser().resolve()
    feature_names_path = Path(feature_names_path).expanduser().resolve()
    labels = Path(labels_path).expanduser().resolve() if labels_path else None
    if not windows_path.is_file():
        raise FileNotFoundError(f"Window file does not exist: {windows_path}")
    if len({windows_path, features_path, feature_names_path}) != 3:
        raise ValueError("Window, feature, and feature-name paths must be distinct")
    features_path.parent.mkdir(parents=True, exist_ok=True)
    feature_names_path.parent.mkdir(parents=True, exist_ok=True)

    windows = np.load(windows_path, mmap_mode="r", allow_pickle=False)
    _validate_inputs(windows, labels)
    names, schema_hash = _expected_schema(windows[0], fs)
    feature_count = len(names)
    expected_shape = (windows.shape[0], feature_count)

    partial = features_path.with_name(features_path.name + ".partial.npy")
    progress_path = features_path.with_name(features_path.name + ".progress.json")
    run_identity = {
        "windows_path": str(windows_path),
        "windows_size": windows_path.stat().st_size,
        "windows_mtime_ns": windows_path.stat().st_mtime_ns,
        "window_shape": list(windows.shape),
        "fs": float(fs),
        "schema_sha256": schema_hash,
        "feature_count": feature_count,
    }

    start = 0
    if resume:
        if not partial.is_file() or not progress_path.is_file():
            raise FileNotFoundError("Resume requested but partial matrix/progress file is missing")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("run_identity") != run_identity:
            raise ValueError("Partial extraction belongs to different windows, fs, or schema")
        matrix = open_memmap(partial, mode="r+")
        if matrix.shape != expected_shape or matrix.dtype != np.dtype(np.float32):
            raise ValueError("Partial feature matrix has an unexpected shape or dtype")
        start = int(progress.get("completed_windows", 0))
        if not 0 <= start <= windows.shape[0]:
            raise ValueError("Invalid completed_windows in progress file")
    else:
        if partial.exists() or progress_path.exists():
            raise FileExistsError(
                f"Partial extraction already exists. Use --resume or remove after inspection: {partial}"
            )
        matrix = open_memmap(partial, mode="w+", dtype=np.float32, shape=expected_shape)
        _write_json_atomic(
            {"run_identity": run_identity, "completed_windows": 0}, progress_path
        )

    started = monotonic()
    for index in range(start, windows.shape[0]):
        features = _extract_one(windows[index], fs)
        if set(features) != set(names) or len(features) != feature_count:
            raise ValueError(f"Feature schema changed at window {index}")
        vector = np.asarray([features[name] for name in names], dtype=np.float32)
        if not np.isfinite(vector).all():
            bad = [names[i] for i in np.flatnonzero(~np.isfinite(vector))[:10]]
            raise ValueError(f"Non-finite features at window {index}: {bad}")
        matrix[index] = vector
        completed = index + 1
        if completed % checkpoint_every == 0 or completed == windows.shape[0]:
            matrix.flush()
            _write_json_atomic(
                {"run_identity": run_identity, "completed_windows": completed}, progress_path
            )
            elapsed = monotonic() - started
            print(
                f"MHEALTH features: {completed}/{windows.shape[0]} windows "
                f"({elapsed:.1f}s since start)",
                flush=True,
            )

    matrix.flush()
    del matrix
    checked = np.load(partial, mmap_mode="r", allow_pickle=False)
    if checked.shape != expected_shape or not np.isfinite(checked).all():
        raise RuntimeError("Completed partial matrix failed shape/finite-value validation")
    del checked

    feature_backup = _backup_incomplete(features_path, feature_count)
    names_backup = None
    if feature_names_path.exists():
        old_names = np.load(feature_names_path, allow_pickle=True)
        if old_names.shape != (feature_count,) or list(map(str, old_names)) != names:
            old_count = old_names.shape[0] if old_names.ndim == 1 else "invalid"
            names_backup = feature_names_path.with_name(
                f"{feature_names_path.stem}.incomplete_{old_count}_backup{feature_names_path.suffix}"
            )
            if names_backup.exists():
                if _sha256(names_backup) != _sha256(feature_names_path):
                    raise FileExistsError(f"Refusing to replace a different backup: {names_backup}")
            else:
                shutil.copy2(feature_names_path, names_backup)

    names_temporary = feature_names_path.with_name(feature_names_path.name + ".complete.npy")
    with names_temporary.open("wb") as handle:
        np.save(handle, np.asarray(names, dtype=object), allow_pickle=True)
    os.replace(partial, features_path)
    os.replace(names_temporary, feature_names_path)
    progress_path.unlink()

    report = {
        **run_identity,
        "completed_windows": windows.shape[0],
        "features_path": str(features_path),
        "feature_names_path": str(feature_names_path),
        "features_shape": list(expected_shape),
        "features_dtype": "float32",
        "features_sha256": _sha256(features_path),
        "feature_names_sha256": _sha256(feature_names_path),
        "incomplete_feature_backup": str(feature_backup) if feature_backup else None,
        "incomplete_name_backup": str(names_backup) if names_backup else None,
    }
    report_path = features_path.with_name(features_path.stem + ".reextract_report.json")
    _write_json_atomic(report, report_path)
    report["report_path"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> None:
    spec = load_dataset_spec("mhealth")
    default_dir = spec.feature_names_path.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=default_dir / "data_20_120.npy")
    parser.add_argument("--labels", type=Path, default=spec.labels_path)
    parser.add_argument("--features-output", type=Path, default=spec.features_path)
    parser.add_argument("--names-output", type=Path, default=spec.feature_names_path)
    parser.add_argument("--fs", type=float, default=20.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    report = reextract_all_windows(
        windows_path=args.windows,
        labels_path=args.labels,
        features_path=args.features_output,
        feature_names_path=args.names_output,
        fs=args.fs,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
