#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 参考LIMU中对于HHAR数据集的对齐方法
"""
Preprocess HHAR dataset using the original hhar.py alignment strategy:
fixed time-window alignment of accelerometer and gyroscope streams.

Outputs:
    data_{version}.npy      : windowed sensor data, shape (N, seq_len, 6)
    label_{version}.npy     : windowed labels, shape (N, seq_len, 2)  [activity, user]
    all_features_{version}.npy      : feature matrix, shape (N, num_features)
    all_features_name_{version}.npy : ordered feature names
"""

import os
import logging
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from zshar.features.all_features import extract_features

# ---------------------------- 日志配置 ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------- 配置 ---------------------------------
DATASET_PATH = r"HHAR_Dataset/Activityrecognitionexp"
SAVE_PATH = "hhar_windows"
VERSION = "20_120"

# 与原版 hhar.py 一致：window_time 单位为毫秒，内部会转为微秒
WINDOW_TIME_MS = 50          # 50 ms
SEQ_LEN = 120                # 一个样本由 120 个小时间窗组成
JUMP = 0                     # 0 表示不重叠；>0 表示保留最后 jump 个时间窗继续组成下一个序列

TARGET_SR = 20.0             # 50ms 一个点，对应 20Hz


# ---------------------------- 辅助函数 ---------------------------------
def extract_sensor(data: pd.DataFrame, time_index: int, time_tag: int, window_time_us: int):
    """
    基本沿用原版 hhar.py 的逻辑：
    从当前 time_index 开始，提取 |Creation_Time - time_tag| < window_time_us 的片段。
    若窗口内 User 或 gt 不唯一，则丢弃。
    否则返回该窗口 x/y/z 的均值 + [User, Model, gt]。
    """
    index = time_index
    total_len = len(data)

    while index < total_len and abs(data.iloc[index]["Creation_Time"] - time_tag) < window_time_us:
        index += 1

    if index == time_index:
        return None, index

    data_slice = data.iloc[time_index:index]

    # 新增：同时过滤 gt 缺失；否则 unique 检查可能不稳定
    data_slice = data_slice[data_slice["gt"].notna()]
    if len(data_slice) == 0:
        return None, index

    if data_slice["User"].nunique() > 1 or data_slice["gt"].nunique() > 1:
        return None, index

    sensor_xyz = data_slice[["x", "y", "z"]].to_numpy(dtype=np.float32)
    sensor_mean = np.mean(sensor_xyz, axis=0)

    label = data_slice[["User", "Model", "gt"]].iloc[0].values
    return np.concatenate([sensor_mean, label]), index


def transform_to_index(arr: np.ndarray, print_label: bool = False):
    uniq = np.unique(arr)
    if print_label:
        logger.info(f"标签集合: {uniq.tolist()}")
    mapping = {v: i for i, v in enumerate(uniq)}
    for k, v in mapping.items():
        arr[arr == k] = v
    return mapping


def separate_data_label(data_raw: np.ndarray):
    """
    与原版结构一致：
    data_raw: (N, seq_len, 9)
      前6列: acc_xyz + gyro_xyz
      后3列: User, Model, gt
    输出：
      data  : (N, seq_len, 6)
      labels: (N, seq_len, 2) -> [activity(gt), user]
    """
    if data_raw.ndim != 3 or data_raw.shape[2] != 9:
        raise ValueError(f"Expected data_raw shape (N, seq_len, 9), got {data_raw.shape}")

    labels = data_raw[:, :, -3:].astype(str)   # [User, Model, gt]

    user_map = transform_to_index(labels[:, :, 0], print_label=True)
    model_map = transform_to_index(labels[:, :, 1], print_label=True)
    gt_map = transform_to_index(labels[:, :, 2], print_label=True)

    logger.info(f"User 映射: {user_map}")
    logger.info(f"Model 映射: {model_map}")
    logger.info(f"gt 映射: {gt_map}")

    data = data_raw[:, :, :6].astype(np.float32)

    # 只保留 [activity, user]
    users = labels[:, :, 0].astype(np.int32)
    activities = labels[:, :, 2].astype(np.int32)
    label_out = np.stack([activities, users], axis=-1).astype(np.int32)

    return data, label_out


def window_to_sensor_dict(window: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "phone_acc": window[:, 0:3].T,
        "phone_gyro": window[:, 3:6].T,
    }


def extract_all_features(data: np.ndarray, fs: float) -> Tuple[np.ndarray, List[str]]:
    if data.ndim != 3 or data.shape[2] != 6:
        raise ValueError(f"Expected data shape (N, seq_len, 6), got {data.shape}")

    logger.info(f"开始为 {data.shape[0]} 个窗口提取特征...")
    start_time = time.time()

    feature_list: List[np.ndarray] = []
    feature_names: List[str] = []

    for i in range(data.shape[0]):
        window = data[i]
        sensors = window_to_sensor_dict(window)
        feats_dict = extract_features(sensors, fs=fs, channel_names=("phone",))

        if i == 0:
            feature_names = sorted(feats_dict.keys())
            logger.info(f"特征总数: {len(feature_names)}")

        feat_vec = np.asarray([feats_dict[k] for k in feature_names], dtype=np.float32)
        feature_list.append(feat_vec)

        if (i + 1) % 500 == 0 or i == data.shape[0] - 1:
            logger.info(f"特征提取进度: {i+1}/{data.shape[0]} 窗口")

    features = np.vstack(feature_list)
    elapsed = time.time() - start_time
    logger.info(f"特征提取完成，耗时 {elapsed:.2f} 秒")
    return features, feature_names


# ---------------------------- 主预处理函数 ---------------------------------
def preprocess_hhar(data_path: str,
                    save_path: str,
                    version: str,
                    window_time_ms: int = 50,
                    seq_len: int = 120,
                    jump: int = 0,
                    target_sr: float = 20.0):
    """
    参考原版 hhar.py 的对齐方法：
    1) 分别从 acc / gyro 流中，按固定时间窗提取数据
    2) 每个小窗内对 x/y/z 求均值
    3) acc / gyro 标签完全一致时，拼成一个 6 维观测点
    4) 连续累积 seq_len 个观测点形成一个样本
    5) 保留新版的特征提取与保存
    """
    logger.info("========== HHAR 数据预处理开始 ==========")
    total_start = time.time()

    acc_path = os.path.join(data_path, "Phones_accelerometer.csv")
    gyro_path = os.path.join(data_path, "Phones_gyroscope.csv")

    logger.info(f"读取数据: {data_path}")
    accs = pd.read_csv(acc_path)
    gyros = pd.read_csv(gyro_path)

    logger.info(f"加速度计原始样本数: {len(accs)}")
    logger.info(f"陀螺仪原始样本数: {len(gyros)}")

    # 时间列转数值并排序，确保和旧版逻辑一致
    for df_name, df in [("acc", accs), ("gyro", gyros)]:
        df["Creation_Time"] = pd.to_numeric(df["Creation_Time"], errors="coerce")
        before = len(df)
        df.dropna(subset=["Creation_Time"], inplace=True)
        after = len(df)
        if before != after:
            logger.info(f"{df_name}: 删除 {before - after} 条 Creation_Time 非法记录")
        df.sort_values("Creation_Time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # 先过滤明显缺失 gt 的记录，减少无效窗口
    acc_before = len(accs)
    gyro_before = len(gyros)
    accs = accs[accs["gt"].notna()].reset_index(drop=True)
    gyros = gyros[gyros["gt"].notna()].reset_index(drop=True)
    logger.info(f"acc 过滤 gt 缺失后: {len(accs)} (移除 {acc_before - len(accs)})")
    logger.info(f"gyro 过滤 gt 缺失后: {len(gyros)} (移除 {gyro_before - len(gyros)})")

    window_time_us = int(window_time_ms * 1e6)

    time_tag = min(accs.iloc[0]["Creation_Time"], gyros.iloc[0]["Creation_Time"])
    time_index = [0, 0]   # [acc_index, gyro_index]
    window_num = 0

    data = []
    data_temp = []

    valid_small_windows = 0
    invalid_small_windows = 0

    logger.info(
        f"开始按原版方法对齐 (window_time={window_time_ms}ms, seq_len={seq_len}, jump={jump})..."
    )
    align_start = time.time()

    while time_index[0] < len(accs) and time_index[1] < len(gyros):
        acc, time_index_new_acc = extract_sensor(
            accs, time_index[0], time_tag, window_time_us
        )
        gyro, time_index_new_gyro = extract_sensor(
            gyros, time_index[1], time_tag, window_time_us
        )

        time_index = [time_index_new_acc, time_index_new_gyro]

        if acc is not None and gyro is not None and np.all(acc[-3:] == gyro[-3:]):
            # acc[:-3] = 3维均值, gyro[:-3] = 3维均值, acc[-3:] = [User, Model, gt]
            one_step = np.concatenate([acc[:-3], gyro[:-3], acc[-3:]])
            data_temp.append(one_step)
            window_num += 1
            valid_small_windows += 1

            time_tag += window_time_us

            if window_num == seq_len:
                data.append(np.array(data_temp, dtype=object))

                if jump == 0:
                    data_temp.clear()
                    window_num = 0
                else:
                    data_temp = data_temp[-jump:]
                    window_num -= jump
        else:
            invalid_small_windows += 1

            if window_num > 0:
                data_temp.clear()
                window_num = 0

            if time_index[0] < len(accs) and time_index[1] < len(gyros):
                time_tag = min(
                    accs.iloc[time_index[0]]["Creation_Time"],
                    gyros.iloc[time_index[1]]["Creation_Time"]
                )
            else:
                break

        if (valid_small_windows + invalid_small_windows) % 50000 == 0:
            logger.info(
                f"小时间窗进度: valid={valid_small_windows}, invalid={invalid_small_windows}, "
                f"已生成序列={len(data)}"
            )

    align_elapsed = time.time() - align_start
    logger.info(f"小时间窗有效数: {valid_small_windows}")
    logger.info(f"小时间窗无效数: {invalid_small_windows}")
    logger.info(f"生成序列数: {len(data)}")
    logger.info(f"原版方式对齐+组序列完成，耗时 {align_elapsed:.2f} 秒")

    if not data:
        raise ValueError("未生成任何有效序列，请尝试减小 seq_len 或增大 window_time_ms")

    data_raw = np.array(data, dtype=object)     # (N, seq_len, 9)
    data_new, label_new = separate_data_label(data_raw)

    logger.info(f"数据形状: {data_new.shape}")
    logger.info(f"标签形状: {label_new.shape}")

    # 特征提取
    features, feature_names = extract_all_features(data_new, fs=target_sr)

    # 保存
    logger.info(f"保存结果到: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, f"data_{version}.npy"), data_new)
    logger.info(f"已保存: data_{version}.npy")

    np.save(os.path.join(save_path, f"label_{version}.npy"), label_new)
    logger.info(f"已保存: label_{version}.npy")

    np.save(os.path.join(save_path, f"all_features_{version}.npy"), features)
    logger.info(f"已保存: all_features_{version}.npy")

    np.save(
        os.path.join(save_path, f"all_features_name_{version}.npy"),
        np.asarray(feature_names, dtype=object)
    )
    logger.info(f"已保存: all_features_name_{version}.npy")

    total_elapsed = time.time() - total_start
    logger.info(f"========== 预处理完成，总耗时 {total_elapsed:.2f} 秒 ==========")
    logger.info(f"特征数量: {len(feature_names)}")
    logger.info(f"特征矩阵形状: {features.shape}")

    return data_new, label_new, features, feature_names


if __name__ == "__main__":
    preprocess_hhar(
        data_path=DATASET_PATH,
        save_path=SAVE_PATH,
        version=VERSION,
        window_time_ms=WINDOW_TIME_MS,
        seq_len=SEQ_LEN,
        jump=JUMP,
        target_sr=TARGET_SR,
    )