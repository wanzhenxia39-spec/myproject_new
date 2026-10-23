#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MHEALTH dataset preprocessing.

目标：
1. 仿照 pamap2_preprocess.py 的逻辑，对连续活动段进行切分、降采样、定长窗口化。
2. 调用 all_features.py 中的 extract_features() 提取每个窗口的统计特征。
3. 保存以下 4 个文件：
   - data_{version}.npy
   - label_{version}.npy
   - all_features_{version}.npy
   - all_features_name_{version}.npy

说明：
- mHealth 原始采样率为 50 Hz。
- 本脚本明确忽略 ECG 和 magnetometer。
- 仅使用以下 IMU 通道：
    hand/right-lower-arm: acc(3) + gyro(3)
    chest:               acc(3)
    ankle/left-ankle:    acc(3) + gyro(3)
  因此窗口数据维度固定为 15。
- MHEALTH 的 chest 不包含 gyro；特征提取时仅传入存在的键，不伪造 chest_gyro。
"""

import os
import re
import numpy as np
import pandas as pd

from zshar.features.all_features import extract_features


# =========================
# Basic utilities
# =========================
def down_sample(data, window_sample):
    """Downsample by averaging.

    Args:
        data: (T, C)
        window_sample: original_sr / target_sr
    Returns:
        (T', C)
    """
    n = data.shape[0]
    if float(window_sample).is_integer():
        window = int(window_sample)
        m = n // window
        return np.mean(data[:m * window].reshape(m, window, -1), axis=1)
    else:
        window = int(window_sample)
        remainder = 0.0
        downsampled = []
        i = 0
        while i + window < n:
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice_data = data[i:i + window + 1]
                downsampled.append(np.mean(slice_data, axis=0))
                i += window + 1
            else:
                slice_data = data[i:i + window]
                downsampled.append(np.mean(slice_data, axis=0))
                i += window
        return np.array(downsampled)



def fill_nan_values(data):
    """Fill NaN values by interpolation + ffill/bfill."""
    df = pd.DataFrame(data)
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df.values



def extract_user_id(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1


# =========================
# MHEALTH column definition
# README uses 1-based indexing; pandas iloc uses 0-based indexing.
# Raw file format:
#  1-3   chest acc
#  4-5   chest ecg         <- ignored
#  6-8   ankle acc
#  9-11  ankle gyro
# 12-14  ankle mag         <- ignored
# 15-17  arm acc
# 18-20  arm gyro
# 21-23  arm mag           <- ignored
# 24     label
# =========================
CHEST_ACC = [0, 1, 2]
ANKLE_ACC = [5, 6, 7]
ANKLE_GYRO = [8, 9, 10]
HAND_ACC = [14, 15, 16]       # right lower arm / wrist-like position
HAND_GYRO = [17, 18, 19]
LABEL_COL = 23


# =========================
# Feature extraction helpers
# =========================
def window_to_sensor_dict(window):
    """Convert one window into the dict expected by all_features.extract_features().

    window shape: (seq_len, 15)
    ordered as:
        [hand_acc(3), hand_gyro(3), chest_acc(3), ankle_acc(3), ankle_gyro(3)]

    Returns:
        dict[str, np.ndarray], each value shaped as (channels, seq_len)
    """
    sensors = {
        "hand_acc": window[:, 0:3].T,
        "hand_gyro": window[:, 3:6].T,
        "chest_acc": window[:, 6:9].T,
        "ankle_acc": window[:, 9:12].T,
        "ankle_gyro": window[:, 12:15].T,
    }
    return sensors



def extract_all_features(data, fs, channel_names):
    """Extract features for all windows.

    Args:
        data: (N, seq_len, 15)
        fs: target sampling rate
        channel_names: passed through to extract_features

    Returns:
        features: (N, F)
        feature_names: (F,)
    """
    feature_list = []
    feature_names = None

    for i in range(data.shape[0]):
        window = data[i]
        sensors = window_to_sensor_dict(window)

        feats_dict = extract_features(sensors, fs, channel_names)

        if feature_names is None:
            feature_names = np.array(sorted(feats_dict.keys()), dtype=object)

        feat_vec = np.array([feats_dict[k] for k in feature_names], dtype=np.float32)
        feature_list.append(feat_vec)

        if i % 500 == 0:
            print(f"Extracting features: {i}/{data.shape[0]}")

    features = np.vstack(feature_list) if feature_list else np.empty((0, 0), dtype=np.float32)
    return features, feature_names


# =========================
# Segment and window logic
# =========================
def process_activity_block(block_X, act_idx, user_idx, seq_len, window_sample):
    """Downsample and split one continuous activity block into fixed windows."""
    if len(block_X) < window_sample:
        return None, None

    block_down = down_sample(block_X, window_sample)
    num_windows = len(block_down) // seq_len

    if num_windows <= 0:
        return None, None

    block_down = block_down[:num_windows * seq_len]
    block_down = block_down.reshape(num_windows, seq_len, -1)

    label_block = np.zeros((num_windows, seq_len, 2), dtype=np.int64)
    label_block[:, :, 0] = act_idx
    label_block[:, :, 1] = user_idx

    return block_down, label_block


# =========================
# Main preprocessing
# =========================
def preprocess_mhealth(
    data_path,
    save_path,
    version,
    target_sr=20,
    seq_len=120,
):
    """Preprocess MHEALTH dataset.

    Args:
        data_path: folder containing mHealth_subject*.log files
        save_path: output folder
        version: suffix for saved npy files
        target_sr: target sampling rate after downsampling
        seq_len: window length after downsampling

    Saves:
        data_{version}.npy
        label_{version}.npy
        all_features_{version}.npy
        all_features_name_{version}.npy
    """
    original_sr = 50
    window_sample = original_sr / target_sr

    # Explicitly ignore ECG and magnetometer.
    feature_cols = HAND_ACC + HAND_GYRO + CHEST_ACC + ANKLE_ACC + ANKLE_GYRO

    all_data = []
    all_labels = []

    user_to_idx = {}
    next_user_idx = 0

    for root, _, files in os.walk(data_path):
        for file in sorted(files):
            if not file.endswith('.log'):
                continue

            filepath = os.path.join(root, file)
            print(f"Processing {filepath}")

            df = pd.read_csv(filepath, sep='\s+', header=None)

            user_raw = extract_user_id(file)
            if user_raw not in user_to_idx:
                user_to_idx[user_raw] = next_user_idx
                next_user_idx += 1
            user_idx = user_to_idx[user_raw]

            activity_ids = df.iloc[:, LABEL_COL].values.astype(int)
            X = df.iloc[:, feature_cols].values.astype(np.float32)
            X = fill_nan_values(X)

            current_act = None
            start_idx = 0

            for i, act in enumerate(activity_ids):
                if act not in range(1, 13):
                    if current_act is not None:
                        block_X = X[start_idx:i]
                        block_act = current_act

                        act_idx = int(block_act) - 1  # Stable mapping matching configs/datasets/mhealth.json.

                        block_data, block_label = process_activity_block(
                            block_X, act_idx, user_idx, seq_len, window_sample
                        )
                        if block_data is not None:
                            all_data.append(block_data)
                            all_labels.append(block_label)

                        current_act = None
                else:
                    if current_act is None:
                        current_act = act
                        start_idx = i
                    elif current_act != act:
                        block_X = X[start_idx:i]
                        block_act = current_act

                        act_idx = int(block_act) - 1  # Stable mapping matching configs/datasets/mhealth.json.

                        block_data, block_label = process_activity_block(
                            block_X, act_idx, user_idx, seq_len, window_sample
                        )
                        if block_data is not None:
                            all_data.append(block_data)
                            all_labels.append(block_label)

                        current_act = act
                        start_idx = i

            # tail block
            if current_act is not None:
                block_X = X[start_idx:]
                block_act = current_act

                act_idx = int(block_act) - 1  # Stable mapping matching configs/datasets/mhealth.json.

                block_data, block_label = process_activity_block(
                    block_X, act_idx, user_idx, seq_len, window_sample
                )
                if block_data is not None:
                    all_data.append(block_data)
                    all_labels.append(block_label)

    if not all_data:
        raise RuntimeError(
            "No valid windows were generated. Please check data_path, target_sr, seq_len, and raw files."
        )

    data = np.concatenate(all_data, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)

    print("Start extracting features...")
    channel_names = ['hand', 'chest', 'ankle']  # Must match window_to_sensor_dict keys.
    all_features, all_feature_names = extract_all_features(
        data,
        fs=target_sr,
        channel_names=channel_names,
    )

    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, f"data_{version}.npy"), data)
    np.save(os.path.join(save_path, f"label_{version}.npy"), labels)
    np.save(os.path.join(save_path, f"all_features_{version}.npy"), all_features)
    np.save(os.path.join(save_path, f"all_features_name_{version}.npy"), all_feature_names)

    print(f"data shape: {data.shape}")
    print(f"label shape: {labels.shape}")
    print(f"all_features shape: {all_features.shape}")
    print(f"all_features_name shape: {all_feature_names.shape}")
    print("Saved all files successfully.")

    return data, labels, all_features, all_feature_names


if __name__ == "__main__":
    from zshar.preprocessing.cli import main
    main(default_dataset='mhealth')
