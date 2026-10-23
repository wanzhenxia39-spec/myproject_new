#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from zshar.features.all_features import extract_features


def window_to_sensor_dict_shoaib(window_9d):
    """
    将 Shoaib 的单窗口数据从 (seq_len, 9) 转成 extract_features 需要的格式。
    输入:
        window_9d: ndarray, shape = (seq_len, 9)
                   [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]
    输出:
        sensors: dict
            {
                "device_acc":  (3, seq_len),
                "device_gyro": (3, seq_len),
            }
    注意:
        这里只保留 acc + gyro，自动舍弃最后 3 维 mag
    """
    if window_9d.ndim != 2 or window_9d.shape[1] < 6:
        raise ValueError(f"Expected window shape (seq_len, >=6), got {window_9d.shape}")

    # 只取前6维: acc(3) + gyro(3)
    window_6d = window_9d[:, :6]

    sensors = {
        "device_acc": window_6d[:, 0:3].T,   # (3, seq_len)
        "device_gyro": window_6d[:, 3:6].T,  # (3, seq_len)
    }
    return sensors


def extract_all_features_from_shoaib_data(data, fs=20.0, channel_names=None, verbose_every=500):
    """
    对 Shoaib 已切窗数据批量提特征。

    参数:
        data: ndarray, shape = (N, seq_len, 9)
        fs: 采样率，默认 20Hz（与你的 version=20_120 对应）
        channel_names: 传给 extract_features 的 channel_names，可为 None 或 ["device"]
        verbose_every: 每隔多少个窗口打印一次进度

    返回:
        features: ndarray, shape = (N, num_features)
        feature_names: ndarray, shape = (num_features,)
    """
    if data.ndim != 3:
        raise ValueError(f"`data` must be 3D, got shape {data.shape}")

    n_samples = data.shape[0]
    feature_list = []
    feature_names = None

    for i in range(n_samples):
        window = data[i]  # (seq_len, 9)

        sensors = window_to_sensor_dict_shoaib(window)

        # 提特征
        feats_dict = extract_features(
            sensor_data=sensors,
            fs=fs,
            channel_names=channel_names
        )

        # 固定特征顺序
        if feature_names is None:
            feature_names = np.array(sorted(feats_dict.keys()), dtype=object)

        feat_vec = np.array([feats_dict[k] for k in feature_names], dtype=np.float32)
        feature_list.append(feat_vec)

        if (i % verbose_every == 0) or (i == n_samples - 1):
            print(f"Extracting features: {i + 1}/{n_samples}")

    features = np.vstack(feature_list)
    return features, feature_names


def preprocess_shoaib_from_npy(data_npy_path,
                               save_path,
                               version="20_120",
                               label_npy_path=None,
                               fs=20.0):
    """
    直接从 shoaib.py 生成的 data_*.npy 提取特征，并保存为:
        all_features_{version}.npy
        all_features_name_{version}.npy

    参数:
        data_npy_path: data_20_120.npy 路径
        save_path: 保存目录
        version: 版本号字符串
        label_npy_path: 可选，label_20_120.npy 路径，只用于校验
        fs: 采样率，默认20Hz
    """
    os.makedirs(save_path, exist_ok=True)

    print(f"Loading data from: {data_npy_path}")
    data = np.load(data_npy_path)

    print(f"Data shape: {data.shape}")
    # 期望形状: (N, seq_len, 9)
    if data.ndim != 3 or data.shape[2] < 6:
        raise ValueError(
            f"Expected data shape like (N, seq_len, 9), but got {data.shape}"
        )

    # 可选读取 label，仅用于检查样本数
    if label_npy_path is not None and os.path.exists(label_npy_path):
        print(f"Loading label from: {label_npy_path}")
        label = np.load(label_npy_path)
        print(f"Label shape: {label.shape}")

        if label.shape[0] != data.shape[0]:
            raise ValueError(
                f"Mismatch between data samples ({data.shape[0]}) and label samples ({label.shape[0]})"
            )

    print("Start extracting features from acc + gyro only (discard mag)...")
    features, feature_names = extract_all_features_from_shoaib_data(
        data=data,
        fs=fs,
        channel_names=["device"],  # 对应 device_acc / device_gyro 这一对
        verbose_every=500
    )

    feature_save_path = os.path.join(save_path, f"all_features_{version}.npy")
    feature_name_save_path = os.path.join(save_path, f"all_features_name_{version}.npy")

    np.save(feature_save_path, features)
    np.save(feature_name_save_path, feature_names)

    print(f"Features shape: {features.shape}")
    print(f"Feature names shape: {feature_names.shape}")
    print(f"Saved to: {feature_save_path}")
    print(f"Saved to: {feature_name_save_path}")

    return features, feature_names


if __name__ == "__main__":
    # ===== 这里改成你的实际路径 =====
    data_npy_path = r"shoaib_windows/data_20_120.npy"
    label_npy_path = r"shoaib_windows/label_20_120.npy"   # 可选，不想读可以设为 None
    save_path = r"shoaib_windows"
    version = "20_120"

    preprocess_shoaib_from_npy(
        data_npy_path=data_npy_path,
        save_path=save_path,
        version=version,
        label_npy_path=label_npy_path,
        fs=20.0
    )