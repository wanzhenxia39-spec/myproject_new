#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocess PAMAP2 and extract a complete feature vector per window.

This version keeps the ORIGINAL PAMAP2 activity ids in label_{version}.npy,
so downstream code can directly use a fixed ID2LABEL dict without requiring
an extra activity-id mapping JSON.

It saves:
- data_{version}.npy               : windowed sensor data, shape (N, seq_len, 18)
- label_{version}.npy              : windowed labels, shape (N, seq_len, 2)
                                     labels[:,:,0] = original PAMAP2 activity id
                                     labels[:,:,1] = remapped user id (0..U-1)
- all_features_{version}.npy       : feature matrix, shape (N, num_features)
- all_features_name_{version}.npy  : ordered feature names, shape (num_features,)
"""

import os
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from zshar.features.all_features import extract_features


# Original PAMAP2 labels used in this project
VALID_ACTIVITY_IDS = {1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24}


def down_sample(data: np.ndarray, window_sample: float) -> np.ndarray:
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


def extract_user_id(filename: str) -> int:
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else -1


def window_to_sensor_dict(window: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert a window of shape (seq_len, 18) into the dict format expected by
    all_features.extract_features.

    18 channels = 3 IMUs × (acc(3) + gyro(3))
    Order:
      hand_acc(3), hand_gyro(3), chest_acc(3), chest_gyro(3), ankle_acc(3), ankle_gyro(3)
    """
    return {
        "hand_acc": window[:, 0:3].T,
        "hand_gyro": window[:, 3:6].T,
        "chest_acc": window[:, 6:9].T,
        "chest_gyro": window[:, 9:12].T,
        "ankle_acc": window[:, 12:15].T,
        "ankle_gyro": window[:, 15:18].T,
    }


def extract_all_features(
    data: np.ndarray,
    fs: float,
    channel_names: Iterable[str] = ("hand", "chest", "ankle"),
) -> Tuple[np.ndarray, List[str]]:
    if data.ndim != 3 or data.shape[2] != 18:
        raise ValueError(f"Expected data shape (N, seq_len, 18), got {data.shape}")

    feature_list: List[np.ndarray] = []
    feature_names: List[str] = []

    for i in range(data.shape[0]):
        window = data[i]
        sensors = window_to_sensor_dict(window)

        feats_dict = extract_features(sensors, fs=fs, channel_names=channel_names)

        if i == 0:
            feature_names = sorted(feats_dict.keys())

        feat_vec = np.asarray([feats_dict[k] for k in feature_names], dtype=np.float32)
        feature_list.append(feat_vec)

        if i % 500 == 0:
            print(f"Extracting features: {i}/{data.shape[0]}")

    if not feature_list:
        raise ValueError("No windows available for feature extraction.")

    features = np.vstack(feature_list)
    return features, feature_names


def _append_block(
    all_data: List[np.ndarray],
    all_labels: List[np.ndarray],
    block_X: np.ndarray,
    activity_id: int,
    user_idx: int,
    window_sample: float,
    seq_len: int,
) -> None:
    """
    Append one contiguous activity block after downsampling and windowing.

    IMPORTANT:
    We store the ORIGINAL PAMAP2 activity id directly in label_block[:,:,0].
    """
    if len(block_X) < window_sample:
        return

    block_down = down_sample(block_X, window_sample)
    num_windows = len(block_down) // seq_len
    if num_windows <= 0:
        return

    block_down = block_down[: num_windows * seq_len]
    block_down = block_down.reshape(num_windows, seq_len, -1)

    label_block = np.zeros((num_windows, seq_len, 2), dtype=np.int32)
    label_block[:, :, 0] = activity_id  # original PAMAP2 label id
    label_block[:, :, 1] = user_idx

    all_data.append(block_down)
    all_labels.append(label_block)


def preprocess_pamap2(
    data_path: str,
    save_path: str,
    version: str,
    target_sr: int = 20,
    seq_len: int = 120,
):
    original_sr = 100
    window_sample = original_sr / target_sr

    hand_acc = [4, 5, 6]
    hand_gyro = [10, 11, 12]
    chest_acc = [21, 22, 23]
    chest_gyro = [27, 28, 29]
    ankle_acc = [38, 39, 40]
    ankle_gyro = [44, 45, 46]
    feature_cols = hand_acc + hand_gyro + chest_acc + chest_gyro + ankle_acc + ankle_gyro

    all_data: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    user_to_idx: Dict[int, int] = {}
    next_user_idx = 0

    for root, _dirs, files in os.walk(data_path):
        for file in files:
            if not file.endswith(".dat"):
                continue

            filepath = os.path.join(root, file)
            print(f"Processing {filepath}")

            df = pd.read_csv(filepath, sep=r"\s+", header=None)

            user_raw = extract_user_id(file)
            if user_raw not in user_to_idx:
                user_to_idx[user_raw] = next_user_idx
                next_user_idx += 1
            user_idx = user_to_idx[user_raw]

            activity_ids = df.iloc[:, 1].values
            X = df.iloc[:, feature_cols].values.astype(float)
            X = fill_nan_values(X)

            current_act = None
            start_idx = 0

            for i in range(len(activity_ids)):
                act = int(activity_ids[i])

                # Ignore null label and unsupported labels.
                if act == 0 or act not in VALID_ACTIVITY_IDS:
                    if current_act is not None:
                        block_X = X[start_idx:i]
                        block_act = current_act
                        _append_block(all_data, all_labels, block_X, block_act, user_idx, window_sample, seq_len)
                        current_act = None
                else:
                    if current_act is None:
                        current_act = act
                        start_idx = i
                    elif current_act != act:
                        block_X = X[start_idx:i]
                        block_act = current_act
                        _append_block(all_data, all_labels, block_X, block_act, user_idx, window_sample, seq_len)
                        current_act = act
                        start_idx = i

            # flush tail
            if current_act is not None:
                block_X = X[start_idx:]
                block_act = current_act
                _append_block(all_data, all_labels, block_X, block_act, user_idx, window_sample, seq_len)

    if not all_data:
        raise ValueError(f"No valid PAMAP2 windows were generated from {data_path}")

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print("Start extracting complete features...")
    features, feature_names = extract_all_features(data, fs=target_sr, channel_names=("hand", "chest", "ankle"))

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f"data_{version}.npy"), data)
    np.save(os.path.join(save_path, f"label_{version}.npy"), labels)
    np.save(os.path.join(save_path, f"all_features_{version}.npy"), features)
    np.save(os.path.join(save_path, f"all_features_name_{version}.npy"), np.asarray(feature_names, dtype=object))

    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Num feature names: {len(feature_names)}")
    print("Saved data/label/features/feature_names successfully")

    # sanity check: labels should be original PAMAP2 ids
    # unique_act_ids = np.unique(labels[:, 0, 0])
    # print(f"Unique activity ids in saved labels: {unique_act_ids.tolist()}")

    return data, labels, features, feature_names


if __name__ == "__main__":
    from zshar.preprocessing.cli import main
    main(default_dataset='pamap2')
