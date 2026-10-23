#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess HHAR dataset with sliding windows and extract complete feature vectors.
Outputs:
    data_{version}.npy      : windowed sensor data, shape (N, seq_len, 6)
    label_{version}.npy     : windowed labels, shape (N, seq_len, 2)  [activity, user]
    all_features_{version}.npy  : feature matrix, shape (N, num_features)
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
DATASET_PATH = r"HHAR_Dataset/Activityrecognitionexp"   # 请修改为实际路径
SAVE_PATH = 'hhar_windows'
VERSION = '20_120'          # 表示采样率 50Hz, 窗口长度 300 (6秒)
TARGET_SR = 20.0            # 目标采样率 (Hz)
SEQ_LEN = 120               # 窗口长度 (点数)
STRIDE = SEQ_LEN            # 无重叠窗口，如需重叠可设为较小值
TOLERANCE_US = 300000        # 传感器对齐容差 (微秒，10ms)

# ---------------------------- 辅助函数 ---------------------------------
def align_sensors(acc_df: pd.DataFrame, gyro_df: pd.DataFrame,
                  tolerance_us: int) -> pd.DataFrame:
    """通过 merge_asof 对齐加速度计与陀螺仪，返回对齐后的 DataFrame"""
    logger.info("开始对齐加速度计和陀螺仪数据...")
    start_time = time.time()

    # 准备时间列
    for df in (acc_df, gyro_df):
        df['Creation_Time'] = pd.to_numeric(df['Creation_Time'])
        df.sort_values('Creation_Time', inplace=True)

    # 重命名避免冲突
    acc_df = acc_df.rename(columns={'x': 'acc_x', 'y': 'acc_y', 'z': 'acc_z'})
    gyro_df = gyro_df.rename(columns={'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'})

    # 最近邻合并，允许 tolerance 内的误差
    merged = pd.merge_asof(
        acc_df,
        gyro_df[['Creation_Time', 'gyro_x', 'gyro_y', 'gyro_z', 'User', 'Model', 'gt']],
        on='Creation_Time',
        direction='nearest',
        tolerance=tolerance_us
    )
    before_len = len(merged)
    merged.dropna(subset=['gyro_x', 'gyro_y', 'gyro_z'], inplace=True)
    after_len = len(merged)
    logger.info(f"合并后总样本数: {after_len} (丢弃了 {before_len - after_len} 条无对应陀螺仪的数据)")

    # 统一标签列
    merged['User'] = merged['User_x']
    merged['Model'] = merged['Model_x']
    merged['gt'] = merged['gt_x']
    merged.drop(['User_x', 'User_y', 'Model_x', 'Model_y', 'gt_x', 'gt_y'], axis=1, inplace=True)

    elapsed = time.time() - start_time
    logger.info(f"传感器对齐完成，耗时 {elapsed:.2f} 秒")
    return merged


def encode_labels(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """将字符串标签映射为整数索引，并打印映射关系"""
    logger.info("开始编码标签...")
    for col in columns:
        uniq = df[col].unique()
        mapping = {v: i for i, v in enumerate(uniq)}
        df[col] = df[col].map(mapping)
        logger.info(f"标签 '{col}' 映射: {mapping}")
    logger.info("标签编码完成")
    return df


def window_to_sensor_dict(window: np.ndarray) -> Dict[str, np.ndarray]:
    """
    将窗口数据 (seq_len, 6) 转换为 extract_features 所需的字典格式。
    6 列顺序: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    """
    return {
        "phone_acc": window[:, 0:3].T,   # (3, seq_len)
        "phone_gyro": window[:, 3:6].T,  # (3, seq_len)
    }


def extract_all_features(data: np.ndarray, fs: float) -> Tuple[np.ndarray, List[str]]:
    """
    为每个窗口提取特征向量。
    data: shape (N, seq_len, 6)
    fs: 采样率 (Hz)
    返回 (features, feature_names)
    """
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
                    target_sr: float = 50.0,
                    seq_len: int = 300,
                    stride: int = 300,
                    tolerance_us: int = 10000):
    """
    完整预处理：对齐、滑动窗口、特征提取、保存。
    """
    logger.info("========== HHAR 数据预处理开始 ==========")
    total_start = time.time()

    # 1. 读取原始 CSV
    logger.info(f"读取数据: {data_path}")
    acc_path = os.path.join(data_path, 'Phones_accelerometer.csv')
    gyro_path = os.path.join(data_path, 'Phones_gyroscope.csv')
    acc_df = pd.read_csv(acc_path)
    gyro_df = pd.read_csv(gyro_path)
    logger.info(f"加速度计原始样本数: {len(acc_df)}")
    logger.info(f"陀螺仪原始样本数: {len(gyro_df)}")

    # 2. 对齐传感器
    merged = align_sensors(acc_df, gyro_df, tolerance_us)

    # ========== 新增：过滤掉 gt 为 nan 的行 ==========
    before_filter = len(merged)
    merged = merged[merged['gt'].notna()]
    after_filter = len(merged)
    logger.info(f"过滤掉 {before_filter - after_filter} 条 gt 缺失的数据，剩余 {after_filter} 条")
    # =================================================

    # 3. 标签编码
    merged = encode_labels(merged, ['User', 'Model', 'gt'])

    # 4. 滑动窗口生成 data 和 labels
    logger.info(f"开始滑动窗口生成 (窗口长度={seq_len}, 步长={stride})...")
    win_start = time.time()

    data_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []   # 每个窗口标签形状 (seq_len, 2): [activity, user]

    total = len(merged)
    total_windows = (total - seq_len) // stride + 1 if total >= seq_len else 0
    logger.info(f"预计生成窗口数: {total_windows}")

    discarded = 0
    for start in range(0, total - seq_len + 1, stride):
        window_df = merged.iloc[start:start + seq_len]
        # 检查窗口内标签一致性
        if (window_df['User'].nunique() == 1 and
            window_df['gt'].nunique() == 1):
            data_win = window_df[['acc_x', 'acc_y', 'acc_z',
                                  'gyro_x', 'gyro_y', 'gyro_z']].values.astype(np.float32)
            activity = window_df['gt'].iloc[0]
            user = window_df['User'].iloc[0]
            label_win = np.full((seq_len, 2), [activity, user], dtype=np.int32)
            data_list.append(data_win)
            label_list.append(label_win)
        else:
            discarded += 1

        # 每 1000 个窗口打印一次进度
        if (len(data_list) % 1000 == 0 and len(data_list) > 0) or (start + stride >= total - seq_len + 1):
            logger.info(f"已生成 {len(data_list)} 个有效窗口 (已丢弃 {discarded} 个不一致窗口)")

    if not data_list:
        raise ValueError("未生成任何有效窗口，请尝试减小 seq_len 或增大 tolerance_us")

    data = np.stack(data_list, axis=0)      # (N, seq_len, 6)
    labels = np.stack(label_list, axis=0)   # (N, seq_len, 2)

    win_elapsed = time.time() - win_start
    logger.info(f"窗口生成完成，有效窗口数: {data.shape[0]}，丢弃窗口数: {discarded}，耗时 {win_elapsed:.2f} 秒")
    logger.info(f"数据形状: {data.shape}")
    logger.info(f"标签形状: {labels.shape}")

    # 5. 提取特征
    features, feature_names = extract_all_features(data, fs=target_sr)

    # 6. 保存
    logger.info(f"保存结果到: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, f"data_{version}.npy"), data)
    logger.info(f"已保存: data_{version}.npy")
    np.save(os.path.join(save_path, f"label_{version}.npy"), labels)
    logger.info(f"已保存: label_{version}.npy")
    np.save(os.path.join(save_path, f"all_features_{version}.npy"), features)
    logger.info(f"已保存: all_features_{version}.npy")
    np.save(os.path.join(save_path, f"all_features_name_{version}.npy"),
            np.asarray(feature_names, dtype=object))
    logger.info(f"已保存: all_features_name_{version}.npy")

    total_elapsed = time.time() - total_start
    logger.info(f"========== 预处理完成，总耗时 {total_elapsed:.2f} 秒 ==========")
    logger.info(f"特征数量: {len(feature_names)}")
    logger.info(f"特征矩阵形状: {features.shape}")

    return data, labels, features, feature_names


if __name__ == "__main__":
    preprocess_hhar(
        data_path=DATASET_PATH,
        save_path=SAVE_PATH,
        version=VERSION,
        target_sr=TARGET_SR,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        tolerance_us=TOLERANCE_US
    )