#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WISDM raw data preprocessing.

目标：
1. 按照类似 pamap2_preprocess.py 的方式，将原始时序数据切分为固定长度窗口；
2. 组织成适配 all_features.py 中 extract_features() 的输入格式；
3. 保存：
   - data_{version}.npy
   - label_{version}.npy
   - all_features_{version}.npy
   - all_features_name_{version}.npy

WISDM 原始数据结构：
raw/
    phone/
        accel/data_1600_accel_phone.txt
        gyro/data_1600_gyro_phone.txt
    watch/
        accel/data_1600_accel_watch.txt
        gyro/data_1600_gyro_watch.txt

每行格式：
Subject-id, Activity Code, Timestamp, x, y, z;
例如：
1600,A,252207666810782,-0.36476135,8.793503,1.0550842;

说明：
- WISDM 的四路传感器是分文件存储，而不是像 PAMAP2 一样天然同行对齐；
- 不同设备/传感器的采样虽目标都是 20Hz，但实际记录条数并不完全一致；
- 因此这里的做法是：
  先对每个 subject 的每个传感器文件按“连续 activity block”切分，
  再对同一 subject + 同一 activity 的四路传感器取最短长度进行截断对齐，
  最后做非重叠窗口切分。

如果你的 all_features.py 中 extract_features 的签名与这里略有不同，
只需要修改 extract_all_features() 中那一行调用即可。
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from zshar.features.all_features import extract_features


# -----------------------------
# 基本工具函数
# -----------------------------
def down_sample(data: np.ndarray, window_sample: float) -> np.ndarray:
    """与 pamap2_preprocess.py 保持一致的降采样逻辑。"""
    n = data.shape[0]
    if float(window_sample).is_integer():
        window = int(window_sample)
        m = n // window
        if m == 0:
            return np.empty((0, data.shape[1]), dtype=float)
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
        if len(downsampled) == 0:
            return np.empty((0, data.shape[1]), dtype=float)
        return np.asarray(downsampled, dtype=float)


def fill_nan_values(data: np.ndarray) -> np.ndarray:
    """线性插值 + 前后填充。"""
    df = pd.DataFrame(data)
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df.values.astype(float)


def extract_user_id(filename: str) -> int:
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1


# -----------------------------
# WISDM 解析
# -----------------------------
def read_wisdm_sensor_file(file_path: str) -> pd.DataFrame:
    """
    读取单个 WISDM 原始传感器文件。

    输入每行形如：
    1600,A,252207666810782,-0.36476135,8.793503,1.0550842;
    """
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # 行尾分号
            if line.endswith(';'):
                line = line[:-1]

            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 6:
                # 跳过异常行
                continue

            subject, activity, timestamp, x, y, z = parts
            try:
                rows.append(
                    [
                        int(subject),
                        str(activity),
                        int(timestamp),
                        float(x),
                        float(y),
                        float(z),
                    ]
                )
            except ValueError:
                # 跳过无法解析的异常行
                continue

    if not rows:
        raise ValueError(f'No valid rows found in {file_path}')

    df = pd.DataFrame(rows, columns=['subject', 'activity', 'timestamp', 'x', 'y', 'z'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def split_into_activity_blocks(df: pd.DataFrame) -> Dict[str, List[np.ndarray]]:
    """
    按“连续 activity 段”切块。

    返回：
        {
            'A': [block1_array, block2_array, ...],
            'B': [...],
        }
    其中每个 block_array 形状为 (n_i, 3)，列为 x,y,z。

    之所以保留 list，是为了防止某些文件里同一 activity 出现多段。
    """
    activity_blocks: Dict[str, List[np.ndarray]] = {}
    activities = df['activity'].values
    values = df[['x', 'y', 'z']].values.astype(float)

    start = 0
    for i in range(1, len(df)):
        if activities[i] != activities[i - 1]:
            act = activities[start]
            activity_blocks.setdefault(act, []).append(values[start:i])
            start = i

    # last block
    act = activities[start]
    activity_blocks.setdefault(act, []).append(values[start:])
    return activity_blocks


# -----------------------------
# 特征提取适配 all_features.py
# -----------------------------
def window_to_sensor_dict(window: np.ndarray) -> Dict[str, np.ndarray]:
    """
    将 WISDM 窗口 (seq_len, 12) 转成 extract_features 需要的格式。

    12 = phone_acc(3) + phone_gyro(3) + watch_acc(3) + watch_gyro(3)
    输出每个 value 形状为 (3, seq_len)，与 PAMAP2 版本保持一致。
    """
    sensors = {
        'phone_acc': window[:, 0:3].T,
        'phone_gyro': window[:, 3:6].T,
        'watch_acc': window[:, 6:9].T,
        'watch_gyro': window[:, 9:12].T,
    }
    return sensors


def extract_all_features(
    data: np.ndarray,
    fs: int,
    channel_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    data: (N, seq_len, 12)
    return:
        features: (N, num_features)
        feature_names: (num_features,)
    """
    feature_list = []
    feature_names = None

    for i in range(data.shape[0]):
        window = data[i]
        sensors = window_to_sensor_dict(window)

        # 与 pamap2_preprocess.py 保持一致的调用形式
        feats_dict = extract_features(sensors, fs, channel_names)

        if feature_names is None:
            feature_names = np.array(sorted(feats_dict.keys()))

        feat_vec = np.array([feats_dict[k] for k in feature_names], dtype=float)
        feature_list.append(feat_vec)

        if i % 500 == 0:
            print(f'Extracting features: {i}/{data.shape[0]}')

    if len(feature_list) == 0:
        raise ValueError('No windows generated, features cannot be extracted.')

    features = np.vstack(feature_list)
    return features, feature_names


# -----------------------------
# 主预处理流程
# -----------------------------
def preprocess_wisdm(
    data_path: str,
    save_path: str,
    version: str,
    target_sr: int = 20,
    seq_len: int = 120,
    sensors_order: Tuple[Tuple[str, str], ...] = (
        ('phone', 'acc'),
        ('phone', 'gyro'),
        ('watch', 'acc'),
        ('watch', 'gyro'),
    ),
):
    """
    参数：
        data_path: WISDM 数据集根目录，或其中包含 raw/ 的目录
        save_path: 保存路径
        version: 文件名版本后缀
        target_sr: 目标采样率，默认 20Hz
        seq_len: 窗口长度（以 target_sr 下的采样点为单位）

    输出：
        data   -> (N, seq_len, 12)
        labels -> (N, seq_len, 2), [:, :, 0] 为 activity_idx, [:, :, 1] 为 user_idx
        all_features -> (N, F)
        all_features_name -> (F,)
    """
    original_sr = 20
    window_sample = original_sr / target_sr

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f'data_path not found: {data_path}')

    # 兼容传入根目录或 raw 目录
    # 注意：WISDM 目录名用 accel/gyro，但 all_features.py 的通道命名期望是 *_acc / *_gyro。
    raw_root = os.path.join(data_path, 'raw') if os.path.isdir(os.path.join(data_path, 'raw')) else data_path

    required_dirs = [
        os.path.join(raw_root, 'phone', 'accel'),
        os.path.join(raw_root, 'phone', 'gyro'),
        os.path.join(raw_root, 'watch', 'accel'),
        os.path.join(raw_root, 'watch', 'gyro'),
    ]
    for d in required_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f'Required directory not found: {d}')

    # 从 phone/accel 下枚举 subject 文件
    subject_files = sorted(
        [f for f in os.listdir(os.path.join(raw_root, 'phone', 'accel')) if f.endswith('.txt')]
    )
    if len(subject_files) == 0:
        raise ValueError('No .txt files found under raw/phone/accel')

    all_data = []
    all_labels = []

    user_to_idx: Dict[int, int] = {}
    # WISDM has no N activity; keep IDs stable even when a subject lacks classes.
    activity_to_idx = {code: i for i, code in enumerate("ABCDEFGHIJKLMOPQRS")}
    next_user_idx = 0

    for file_name in subject_files:
        subject_id = extract_user_id(file_name)
        if subject_id < 0:
            print(f'[Skip] Cannot parse subject id from {file_name}')
            continue

        if subject_id not in user_to_idx:
            user_to_idx[subject_id] = next_user_idx
            next_user_idx += 1
        user_idx = user_to_idx[subject_id]

        print(f'Processing subject {subject_id} ...')

        # 读取四路传感器
        sensor_blocks_by_name: Dict[str, Dict[str, List[np.ndarray]]] = {}
        missing_sensor = False

        for device, sensor in sensors_order:
            sensor_name = f'{device}_{sensor}'
            sensor_dir = 'accel' if sensor == 'acc' else sensor
            sensor_file = os.path.join(raw_root, device, sensor_dir, f'data_{subject_id}_{sensor_dir}_{device}.txt')

            if not os.path.isfile(sensor_file):
                print(f'[Warning] Missing file: {sensor_file}')
                missing_sensor = True
                break

            df = read_wisdm_sensor_file(sensor_file)
            # 容错：若文件内 subject 不一致，则仍以文件名 subject 为准继续处理
            sensor_blocks_by_name[sensor_name] = split_into_activity_blocks(df)

        if missing_sensor:
            continue

        # 找到该 subject 四路传感器共有的 activity
        common_activities = set(sensor_blocks_by_name['phone_acc'].keys())
        for sensor_name in sensor_blocks_by_name:
            common_activities &= set(sensor_blocks_by_name[sensor_name].keys())
        common_activities = sorted(common_activities)

        for act in common_activities:
            if act not in activity_to_idx:
                continue
            act_idx = activity_to_idx[act]

            # 同一 activity 可能理论上有多段，这里按“每一段的共同编号”对齐
            num_blocks = min(len(sensor_blocks_by_name[s][act]) for s in sensor_blocks_by_name)
            if num_blocks == 0:
                continue

            for block_id in range(num_blocks):
                phone_acc = sensor_blocks_by_name['phone_acc'][act][block_id]
                phone_gyro = sensor_blocks_by_name['phone_gyro'][act][block_id]
                watch_acc = sensor_blocks_by_name['watch_acc'][act][block_id]
                watch_gyro = sensor_blocks_by_name['watch_gyro'][act][block_id]

                min_len = min(len(phone_acc), len(phone_gyro), len(watch_acc), len(watch_gyro))
                if min_len <= 0:
                    continue

                # 截断到四路共同最短长度，保证 4 个模态逐样本对齐
                block = np.concatenate(
                    [
                        phone_acc[:min_len],
                        phone_gyro[:min_len],
                        watch_acc[:min_len],
                        watch_gyro[:min_len],
                    ],
                    axis=1,
                )  # (min_len, 12)

                block = fill_nan_values(block)

                if len(block) < window_sample:
                    continue

                # 保持与 PAMAP2 版本一致：若 target_sr == 20，则相当于不降采样
                block_down = down_sample(block, window_sample)
                if len(block_down) == 0:
                    continue

                num_windows = len(block_down) // seq_len
                if num_windows <= 0:
                    continue

                block_down = block_down[: num_windows * seq_len]
                block_down = block_down.reshape(num_windows, seq_len, -1)

                label_block = np.zeros((num_windows, seq_len, 2), dtype=int)
                label_block[:, :, 0] = act_idx
                label_block[:, :, 1] = user_idx

                all_data.append(block_down)
                all_labels.append(label_block)

    if len(all_data) == 0:
        raise ValueError('No valid windows were generated. Please check data_path and WISDM directory structure.')

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print('Start extracting all_features ...')
    channel_names = ['phone', 'watch']
    features, feature_names = extract_all_features(data, fs=target_sr, channel_names=channel_names)

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f'data_{version}.npy'), data)
    np.save(os.path.join(save_path, f'label_{version}.npy'), labels)
    np.save(os.path.join(save_path, f'all_features_{version}.npy'), features)
    np.save(os.path.join(save_path, f'all_features_name_{version}.npy'), feature_names)

    print(f'data shape: {data.shape}')
    print(f'labels shape: {labels.shape}')
    print(f'all_features shape: {features.shape}')
    print(f'num feature names: {len(feature_names)}')
    print('Saved successfully.')

    return data, labels, features, feature_names


if __name__ == "__main__":
    from zshar.preprocessing.cli import main
    main(default_dataset='wisdm')
