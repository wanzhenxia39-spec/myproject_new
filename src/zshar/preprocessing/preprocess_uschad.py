#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocess USC-HAD and extract a complete feature vector per window.

USC-HAD raw format:
- Directory structure: Subject1/ ... Subject14/
- Each trial is stored as a .mat file named a<m>t<n>.mat.
- sensor_readings has shape (T, 6):
  acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z.
- Original sampling rate is 100 Hz.

This script follows version1/preprocess_pamap2.py style:
- Downsample 100 Hz -> target_sr, default 20 Hz.
- Slice each trial into non-overlapping windows of length seq_len, default 120.
- Map USC-HAD activity ids 1..12 to 0..11 in label[:,:,0].
- Remap subjects to 0-based ids in label[:,:,1].

Saved outputs:
- data_{version}.npy               : windowed sensor data, shape (N, seq_len, 6)
- label_{version}.npy              : windowed labels, shape (N, seq_len, 2)
                                     labels[:,:,0] = mapped activity id (0..11)
                                     labels[:,:,1] = remapped subject id (0..S-1)
- all_features_{version}.npy       : feature matrix, shape (N, num_features)
- all_features_name_{version}.npy  : ordered feature names, shape (num_features,)
- activity_map_{version}.json      : raw activity id -> mapped id/name metadata
- subject_map_{version}.json       : raw subject id -> remapped subject id
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

from zshar.features.all_features import extract_features


USCHAD_ACTIVITY_NAMES = {
    1: "Walking Forward",
    2: "Walking Left",
    3: "Walking Right",
    4: "Walking Upstairs",
    5: "Walking Downstairs",
    6: "Running Forward",
    7: "Jumping Up",
    8: "Sitting",
    9: "Standing",
    10: "Sleeping",
    11: "Elevator Up",
    12: "Elevator Down",
}

VALID_ACTIVITY_IDS = set(USCHAD_ACTIVITY_NAMES)
CHANNEL_NAMES = ("hip",)


def down_sample(data: np.ndarray, window_sample: float) -> np.ndarray:
    n = data.shape[0]
    if float(window_sample).is_integer():
        window = int(window_sample)
        m = n // window
        return np.mean(data[: m * window].reshape(m, window, -1), axis=1)

    window = int(window_sample)
    remainder = 0.0
    downsampled = []
    i = 0
    while i + window < n:
        remainder += window_sample - window
        if remainder >= 1:
            remainder -= 1
            slice_data = data[i : i + window + 1]
            downsampled.append(np.mean(slice_data, axis=0))
            i += window + 1
        else:
            slice_data = data[i : i + window]
            downsampled.append(np.mean(slice_data, axis=0))
            i += window
    return np.asarray(downsampled)


def fill_nan_values(data: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(data)
    df.interpolate(method="linear", limit_direction="both", inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df.values


def extract_number(text: str) -> int:
    match = re.search(r"\d+", text)
    return int(match.group()) if match else -1


def parse_trial_filename(filename: str) -> Tuple[int, int]:
    match = re.fullmatch(r"a(\d+)t(\d+)\.mat", filename)
    if not match:
        raise ValueError(f"Unexpected USC-HAD trial filename: {filename}")
    return int(match.group(1)), int(match.group(2))


def list_subject_dirs(data_path: str) -> List[str]:
    return sorted(
        [
            name
            for name in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, name)) and name.lower().startswith("subject")
        ],
        key=extract_number,
    )


def list_trial_files(subject_path: str) -> List[str]:
    return sorted(
        [
            name
            for name in os.listdir(subject_path)
            if os.path.isfile(os.path.join(subject_path, name))
            and re.fullmatch(r"a\d+t\d+\.mat", name)
        ],
        key=lambda x: parse_trial_filename(x),
    )


def scalar_to_str(value: object) -> str:
    if isinstance(value, np.ndarray):
        value = np.squeeze(value)
        if value.shape == ():
            return str(value.item())
    return str(value)


def scalar_to_int(value: object, default: int = -1) -> int:
    try:
        return int(float(scalar_to_str(value)))
    except (TypeError, ValueError):
        return default


def read_one_uschad_trial(file_path: str) -> Tuple[np.ndarray, Dict[str, object]]:
    mat = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    if "sensor_readings" not in mat:
        raise KeyError(f"Missing sensor_readings in {file_path}")

    data = np.asarray(mat["sensor_readings"], dtype=float)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(
            f"Unexpected sensor_readings shape in {file_path}: {data.shape}, expected (T, 6)"
        )

    data = fill_nan_values(data)
    metadata = {
        "subject": scalar_to_int(mat.get("subject", -1)),
        "activity_number": scalar_to_int(mat.get("activity_number", -1)),
        "activity": scalar_to_str(mat.get("activity", "")),
        "trial": scalar_to_int(mat.get("trial", -1)),
        "sensor_location": scalar_to_str(mat.get("sensor_location", "")),
        "sensor_orientation": scalar_to_str(mat.get("sensor_orientation", "")),
    }
    return data, metadata


def window_to_sensor_dict(window: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert one window from shape (seq_len, 6) to all_features.extract_features input.

    USC-HAD has one MotionNode sensor at the front-right hip:
      hip_acc(3), hip_gyro(3)
    Each returned array has shape (3, seq_len).
    """
    if window.ndim != 2 or window.shape[1] != 6:
        raise ValueError(f"window must have shape (seq_len, 6), got {window.shape}")
    return {
        "hip_acc": window[:, 0:3].T,
        "hip_gyro": window[:, 3:6].T,
    }


def extract_all_features(
    data: np.ndarray,
    fs: float,
    channel_names: Iterable[str] = CHANNEL_NAMES,
) -> Tuple[np.ndarray, np.ndarray]:
    if data.ndim != 3 or data.shape[2] != 6:
        raise ValueError(f"Expected data shape (N, seq_len, 6), got {data.shape}")

    feature_list: List[np.ndarray] = []
    feature_names = None

    for i in range(data.shape[0]):
        sensors = window_to_sensor_dict(data[i])
        feats_dict = extract_features(sensors, fs=fs, channel_names=channel_names)

        if feature_names is None:
            feature_names = np.asarray(sorted(feats_dict.keys()), dtype=object)

        feat_vec = np.asarray([feats_dict[k] for k in feature_names], dtype=np.float32)
        feature_list.append(feat_vec)

        if i % 500 == 0:
            print(f"Extracting features: {i}/{data.shape[0]}")

    if not feature_list or feature_names is None:
        raise ValueError("No windows available for feature extraction.")

    features = np.vstack(feature_list)
    return features, feature_names


def append_trial_windows(
    all_data: List[np.ndarray],
    all_labels: List[np.ndarray],
    trial_X: np.ndarray,
    activity_idx: int,
    subject_idx: int,
    window_sample: float,
    seq_len: int,
) -> None:
    if len(trial_X) < window_sample:
        return

    trial_down = down_sample(trial_X, window_sample)
    num_windows = len(trial_down) // seq_len
    if num_windows <= 0:
        return

    trial_down = trial_down[: num_windows * seq_len]
    trial_down = trial_down.reshape(num_windows, seq_len, -1)

    label_block = np.zeros((num_windows, seq_len, 2), dtype=np.int32)
    label_block[:, :, 0] = activity_idx
    label_block[:, :, 1] = subject_idx

    all_data.append(trial_down)
    all_labels.append(label_block)


def preprocess_uschad(
    data_path: str,
    save_path: str,
    version: str = "20_120",
    target_sr: int = 20,
    seq_len: int = 120,
):
    original_sr = 100
    window_sample = original_sr / target_sr

    all_data: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    subject_map: Dict[str, Dict[str, int]] = {}
    activity_map: Dict[str, Dict[str, object]] = {
        str(activity_id): {
            "raw_id": activity_id,
            "mapped_idx": activity_id - 1,
            "activity_name": activity_name,
        }
        for activity_id, activity_name in USCHAD_ACTIVITY_NAMES.items()
    }

    subject_dirs = list_subject_dirs(data_path)
    if not subject_dirs:
        raise FileNotFoundError(f"No Subject* folders found under: {data_path}")

    for subject_dir in subject_dirs:
        subject_raw = extract_number(subject_dir)
        subject_idx = len(subject_map)
        subject_map[subject_dir] = {
            "raw_id": subject_raw,
            "mapped_idx": subject_idx,
        }

        subject_path = os.path.join(data_path, subject_dir)
        trial_files = list_trial_files(subject_path)
        if not trial_files:
            print(f"[Warning] {subject_dir} has no aXtY.mat trial files.")
            continue

        print(f"Processing {subject_dir} with {len(trial_files)} trials")

        for trial_file in trial_files:
            file_path = os.path.join(subject_path, trial_file)
            file_activity_id, file_trial_id = parse_trial_filename(trial_file)
            if file_activity_id not in VALID_ACTIVITY_IDS:
                print(f"[Warning] Skip unsupported activity id {file_activity_id}: {file_path}")
                continue
            activity_idx = file_activity_id - 1

            trial_X, metadata = read_one_uschad_trial(file_path)
            mat_activity_id = int(metadata["activity_number"])
            mat_trial_id = int(metadata["trial"])
            mat_subject_id = int(metadata["subject"])

            # A few USC-HAD files contain inconsistent or missing metadata fields.
            # The dataset readme defines the folder/file naming convention, so labels
            # are taken from Subject*/aXtY.mat and metadata mismatches are warnings.
            if mat_activity_id != -1 and mat_activity_id != file_activity_id:
                print(
                    f"[Warning] Activity id mismatch in {file_path}: "
                    f"filename={file_activity_id}, mat={mat_activity_id}; use filename label."
                )
            if mat_activity_id == -1:
                print(
                    f"[Warning] Missing activity_number in {file_path}; "
                    f"use filename label={file_activity_id}."
                )
            if mat_trial_id != -1 and mat_trial_id != file_trial_id:
                print(
                    f"[Warning] Trial id mismatch in {file_path}: "
                    f"filename={file_trial_id}, mat={mat_trial_id}; continue."
                )
            if mat_subject_id != -1 and mat_subject_id != subject_raw:
                print(
                    f"[Warning] Subject id mismatch in {file_path}: "
                    f"folder={subject_raw}, mat={mat_subject_id}; use folder label."
                )

            append_trial_windows(
                all_data=all_data,
                all_labels=all_labels,
                trial_X=trial_X,
                activity_idx=activity_idx,
                subject_idx=subject_idx,
                window_sample=window_sample,
                seq_len=seq_len,
            )

    if not all_data:
        raise RuntimeError("No valid USC-HAD windows were generated. Please check data_path.")

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print("Start extracting complete features...")
    features, feature_names = extract_all_features(data, fs=target_sr, channel_names=CHANNEL_NAMES)

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f"data_{version}.npy"), data)
    np.save(os.path.join(save_path, f"label_{version}.npy"), labels)
    np.save(os.path.join(save_path, f"all_features_{version}.npy"), features)
    np.save(os.path.join(save_path, f"all_features_name_{version}.npy"), feature_names)

    with open(os.path.join(save_path, f"activity_map_{version}.json"), "w", encoding="utf-8") as f:
        json.dump(activity_map, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_path, f"subject_map_{version}.json"), "w", encoding="utf-8") as f:
        json.dump(subject_map, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {save_path}")
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Feature names shape: {feature_names.shape}")
    print(f"Unique activity ids: {np.unique(labels[:, 0, 0]).tolist()}")
    print(f"Unique subject ids: {np.unique(labels[:, 0, 1]).tolist()}")

    return data, labels, features, feature_names


if __name__ == "__main__":
    from zshar.preprocessing.cli import main
    main(default_dataset='uschad')
