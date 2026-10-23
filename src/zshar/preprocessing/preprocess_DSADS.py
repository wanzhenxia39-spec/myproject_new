#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DSADS preprocessing aligned with pamap2_preprocess.py style,
while adapting to all_features.py::extract_features().

Key points:
1. DSADS raw structure:
   a01~a19 / p1~p8 / s01~s60.txt
2. Each sXX.txt is a 125 x 45 CSV segment at 25 Hz.
3. Raw column order per file:
   [T(9), RA(9), LA(9), RL(9), LL(9)]
   each unit = acc(3) + gyro(3) + mag(3)
4. Magnetometer is dropped, so each segment becomes 125 x 30.
5. For each (activity, subject), concatenate s01~s60 into one continuous block,
   downsample 25 Hz -> 20 Hz, then reshape into non-overlapping windows of length 120.
6. Feature extraction uses all_features.extract_features() with 5 sensor prefixes:
   torso, right_arm, left_arm, right_leg, left_leg

Saved outputs:
- data_{version}.npy              shape: (N, seq_len, 30)
- label_{version}.npy             shape: (N, seq_len, 2)
- all_features_{version}.npy      shape: (N, F)
- all_features_name_{version}.npy shape: (F,)
"""

import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from zshar.features.all_features import extract_features


DSADS_ACTIVITY_NAMES = [
    "sitting",
    "standing",
    "lying on back",
    "lying on right side",
    "ascending stairs",
    "descending stairs",
    "standing in an elevator still",
    "moving around in an elevator",
    "walking in a parking lot",
    "walking on a treadmill with a speed of 4 km/h (in flat)",
    "walking on a treadmill with a speed of 4 km/h (in 15 deg inclined positions)",
    "running on a treadmill with a speed of 8 km/h",
    "exercising on a stepper",
    "exercising on a cross trainer",
    "cycling on an exercise bike in horizontal positions",
    "cycling on an exercise bike in vertical positions",
    "rowing",
    "jumping",
    "playing basketball",
]

# Must match DSADS file column-group order exactly:
# T, RA, LA, RL, LL
CHANNEL_NAMES = ["torso", "right_arm", "left_arm", "right_leg", "left_leg"]


def down_sample(data: np.ndarray, window_sample: float) -> np.ndarray:
    """
    Same downsampling logic style as pamap2_preprocess.py.
    Supports non-integer window_sample, e.g. 25/20 = 1.25.
    """
    n = data.shape[0]
    if float(window_sample).is_integer():
        window = int(window_sample)
        m = n // window
        return np.mean(data[: m * window].reshape(m, window, -1), axis=1)
    else:
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


def list_sorted_dirs(path: str, prefix: str) -> List[str]:
    dirs = [
        d
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d.startswith(prefix)
    ]
    return sorted(dirs, key=extract_number)


def list_sorted_files(path: str, suffix: str = ".txt", prefix: str = "s") -> List[str]:
    files = [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
        and f.startswith(prefix)
        and f.endswith(suffix)
    ]
    return sorted(files, key=extract_number)


def read_one_dsads_segment(file_path: str) -> np.ndarray:
    """
    Read one DSADS segment.

    Raw shape: (125, 45)
    Raw layout by 9-column unit:
      acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z

    Keep only acc+gyro from each of the 5 body units.
    Output shape: (125, 30)
    Output order:
      torso_acc(3), torso_gyro(3),
      right_arm_acc(3), right_arm_gyro(3),
      left_arm_acc(3), left_arm_gyro(3),
      right_leg_acc(3), right_leg_gyro(3),
      left_leg_acc(3), left_leg_gyro(3)
    """
    data = pd.read_csv(file_path, header=None, sep=",").values.astype(float)

    if data.shape != (125, 45):
        raise ValueError(
            f"Unexpected DSADS segment shape in {file_path}: {data.shape}, expected (125, 45)"
        )

    data = fill_nan_values(data)

    kept_blocks = []
    for start in range(0, 45, 9):
        unit_block = data[:, start : start + 9]
        acc_gyro_only = unit_block[:, :6]  # drop mag(3)
        kept_blocks.append(acc_gyro_only)

    out = np.concatenate(kept_blocks, axis=1)
    if out.shape != (125, 30):
        raise ValueError(
            f"Unexpected processed segment shape in {file_path}: {out.shape}, expected (125, 30)"
        )
    return out


def window_to_sensor_dict(window: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert one window from shape (seq_len, 30) to the input format expected by
    all_features.extract_features().

    Returned keys:
      torso_acc, torso_gyro,
      right_arm_acc, right_arm_gyro,
      left_arm_acc, left_arm_gyro,
      right_leg_acc, right_leg_gyro,
      left_leg_acc, left_leg_gyro

    Each value shape: (3, seq_len)
    """
    if window.ndim != 2 or window.shape[1] != 30:
        raise ValueError(f"window must have shape (seq_len, 30), got {window.shape}")

    sensors: Dict[str, np.ndarray] = {}
    sensor_dim = 6  # acc(3) + gyro(3)

    for i, prefix in enumerate(CHANNEL_NAMES):
        start = i * sensor_dim
        sensors[f"{prefix}_acc"] = window[:, start : start + 3].T
        sensors[f"{prefix}_gyro"] = window[:, start + 3 : start + 6].T

    return sensors


def extract_all_features(data: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    data: (N, seq_len, 30)
    returns:
      features: (N, F)
      feature_names: (F,)
    """
    feature_list = []
    feature_names = None

    for i in range(data.shape[0]):
        sensors = window_to_sensor_dict(data[i])
        feats_dict = extract_features(sensors, fs=fs, channel_names=CHANNEL_NAMES)

        if feature_names is None:
            feature_names = np.array(sorted(feats_dict.keys()), dtype=object)

        feat_vec = np.array([feats_dict[k] for k in feature_names], dtype=float)
        feature_list.append(feat_vec)

        if i % 500 == 0:
            print(f"Extracting features: {i}/{data.shape[0]}")

    features = np.vstack(feature_list)
    return features, feature_names


def preprocess_dsads(
    data_path: str,
    save_path: str,
    version: str = "20_120",
    target_sr: int = 20,
    seq_len: int = 120,
):
    """
    DSADS preprocessing in PAMAP2 style.

    - Concatenate 60 segments for each (activity, subject)
    - Downsample 25 Hz -> 20 Hz
    - Slice into non-overlapping windows of length 120
    - Build labels with shape (num_windows, seq_len, 2)
      labels[..., 0] = activity_idx (0~18)
      labels[..., 1] = subject_idx  (0~7)
    - Extract all_features.py features
    """
    original_sr = 25
    window_sample = original_sr / target_sr

    all_data = []
    all_labels = []
    activity_map = {}
    subject_map = {}

    activity_dirs = list_sorted_dirs(data_path, "a")
    if not activity_dirs:
        raise FileNotFoundError(f"No activity folders like a01/a02 found under: {data_path}")
    if len(activity_dirs) != 19:
        print(f"[Warning] Found {len(activity_dirs)} activity folders under {data_path}, expected 19.")

    for activity_dir in activity_dirs:
        activity_raw = extract_number(activity_dir)
        activity_idx = activity_raw - 1
        activity_name = (
            DSADS_ACTIVITY_NAMES[activity_idx]
            if 1 <= activity_raw <= len(DSADS_ACTIVITY_NAMES)
            else f"unknown_{activity_raw}"
        )
        activity_map[activity_dir] = {
            "raw_id": activity_raw,
            "mapped_idx": activity_idx,
            "activity_name": activity_name,
        }

        activity_path = os.path.join(data_path, activity_dir)
        subject_dirs = list_sorted_dirs(activity_path, "p")
        if len(subject_dirs) != 8:
            print(f"[Warning] {activity_dir} has {len(subject_dirs)} subject folders, expected 8.")

        for subject_dir in subject_dirs:
            subject_raw = extract_number(subject_dir)
            subject_idx = subject_raw - 1
            subject_map[subject_dir] = {
                "raw_id": subject_raw,
                "mapped_idx": subject_idx,
            }

            subject_path = os.path.join(activity_path, subject_dir)
            segment_files = list_sorted_files(subject_path, suffix=".txt", prefix="s")
            if len(segment_files) != 60:
                print(
                    f"[Warning] {activity_dir}/{subject_dir} has {len(segment_files)} segments, expected 60."
                )
            if not segment_files:
                continue

            print(f"Processing {activity_dir}/{subject_dir} with {len(segment_files)} segments")

            # Concatenate all 60 segments into one long activity block for this subject
            block_data = [read_one_dsads_segment(os.path.join(subject_path, seg)) for seg in segment_files]
            block_X = np.concatenate(block_data, axis=0)  # (60*125, 30)

            # PAMAP2-style resampling + fixed-length windowing
            block_down = down_sample(block_X, window_sample)  # -> approx (6000, 30)
            num_windows = len(block_down) // seq_len
            if num_windows <= 0:
                continue

            block_down = block_down[: num_windows * seq_len]
            block_down = block_down.reshape(num_windows, seq_len, -1)

            label_block = np.zeros((num_windows, seq_len, 2), dtype=int)
            label_block[:, :, 0] = activity_idx
            label_block[:, :, 1] = subject_idx

            all_data.append(block_down)
            all_labels.append(label_block)

    if not all_data:
        raise RuntimeError("No valid windows were generated. Please check data_path and file structure.")

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print("Start extracting features...")
    features, feature_names = extract_all_features(data, fs=target_sr)

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
    print(f"data shape: {data.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"features shape: {features.shape}")
    print(f"feature_names shape: {feature_names.shape}")

    return data, labels, features, feature_names


if __name__ == "__main__":
    from zshar.preprocessing.cli import main
    main(default_dataset='dsads')
