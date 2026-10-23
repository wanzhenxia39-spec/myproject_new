# 基于LIMU-BERT的窗口划分方法，加入我的特征提取方法构建完整的窗口
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/29 15:01
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : uci.py
# @Description : http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions

import os
import numpy as np
import pandas as pd


from zshar.features.all_features import extract_features


DATASET_PATH = r'UCI-HAR_Dataset/RawData'


def down_sample(data, window_sample, start, end):
    """
    按照原始 uci.py 的区间下采样逻辑处理单个动作片段。
    data: (T, C)
    return: (T_down, C)
    """
    result = []
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(int(start), int(end) - window, window):
            slice_data = data[i:i + window, :]
            result.append(np.mean(slice_data, axis=0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = int(start)
        while int(start) <= i + window + 1 < int(end):
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice_data = data[i:i + window + 1, :]
                result.append(np.mean(slice_data, axis=0))
                i += window + 1
            else:
                slice_data = data[i:i + window, :]
                result.append(np.mean(slice_data, axis=0))
                i += window
    return np.array(result)


def window_to_sensor_dict(window, sensor_prefix='phone'):
    """
    将 UCI 的单个窗口 (seq_len, 6) 转换为 all_features.extract_features() 需要的格式。
    这里默认 6 个通道顺序为: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    """
    if window.ndim != 2 or window.shape[1] != 6:
        raise ValueError(f'Expected window shape (seq_len, 6), got {window.shape}')

    sensors = {
        f'{sensor_prefix}_acc': window[:, 0:3].T,
        f'{sensor_prefix}_gyro': window[:, 3:6].T,
    }
    return sensors



def extract_all_features(data, fs, channel_names=None, sensor_prefix='phone'):
    """
    data: (N, seq_len, 6)
    return:
        features: (N, num_features)
        feature_names: (num_features,)
    """
    if data.ndim != 3 or data.shape[2] != 6:
        raise ValueError(f'Expected data shape (N, seq_len, 6), got {data.shape}')

    feature_list = []
    feature_names = None

    for i in range(data.shape[0]):
        window = data[i]
        sensors = window_to_sensor_dict(window, sensor_prefix=sensor_prefix)
        feats_dict = extract_features(sensors, fs=fs, channel_names=channel_names)

        if feature_names is None:
            feature_names = np.array(sorted(feats_dict.keys()), dtype=object)

        feat_vec = np.array([feats_dict[k] for k in feature_names], dtype=np.float32)
        feature_list.append(feat_vec)

        if i % 500 == 0:
            print(f'Extracting features: {i}/{data.shape[0]}')

    if len(feature_list) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=object)

    features = np.vstack(feature_list)
    return features, feature_names



def preprocess(path, save_path, version, raw_sr=50, target_sr=20, seq_len=120, sensor_prefix='phone'):
    """
    预处理 UCI HAR RawData，并额外提取 all_features.py 中定义的统计/物理特征。

    保存内容：
        data_{version}.npy
        label_{version}.npy
        all_features_{version}.npy
        all_features_name_{version}.npy
    """
    labels_txt = np.loadtxt(os.path.join(path, 'labels.txt'), delimiter=' ')

    all_data = []
    all_labels = []
    window_sample = raw_sr / target_sr

    for root, dirs, files in os.walk(path):
        for name in files:
            if not name.startswith('acc'):
                continue

            tags = name.split('.')[0].split('_')
            exp_num = int(tags[1][-2:])
            exp_user = int(tags[2][-2:])

            label_index = (labels_txt[:, 0] == exp_num) & (labels_txt[:, 1] == exp_user)
            label_stat = labels_txt[label_index, :]

            acc_path = os.path.join(root, name)
            gyro_path = os.path.join(root, 'gyro' + name[3:])

            exp_data_acc = np.loadtxt(acc_path, delimiter=' ') * 9.80665
            exp_data_gyro = np.loadtxt(gyro_path, delimiter=' ')
            exp_data_full = np.concatenate([exp_data_acc, exp_data_gyro], axis=1)

            for i in range(label_stat.shape[0]):
                activity_id = int(label_stat[i, 2])
                index_start = int(label_stat[i, 3])
                index_end = int(label_stat[i, 4])

                # 保持原始逻辑：仅使用前 6 个 activity
                if activity_id > 6:
                    continue

                exp_data = down_sample(exp_data_full, window_sample, index_start, index_end)

                if exp_data.shape[0] < seq_len:
                    continue

                usable_len = (exp_data.shape[0] // seq_len) * seq_len
                if usable_len == 0:
                    continue

                exp_data = exp_data[:usable_len, :]
                exp_data = exp_data.reshape(usable_len // seq_len, seq_len, exp_data.shape[1])

                exp_label = np.ones((exp_data.shape[0], exp_data.shape[1], 1), dtype=np.int32)
                exp_label = np.concatenate([
                    exp_label * activity_id,
                    exp_label * exp_user
                ], axis=2)

                all_data.append(exp_data)
                all_labels.append(exp_label)

    if len(all_data) == 0:
        raise ValueError('No valid samples were generated. Please check dataset path and preprocessing settings.')

    data = np.concatenate(all_data, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)

    # activity / user 都从 0 开始编号
    labels[:, :, 0] = labels[:, :, 0] - np.min(labels[:, :, 0])
    labels[:, :, 1] = labels[:, :, 1] - np.min(labels[:, :, 1])

    print('All data processed. Size: %d' % data.shape[0])
    print('Start extracting all_features...')

    # 单 IMU，只传一个 prefix；extract_features 会自动配对 phone_acc / phone_gyro
    features, feature_names = extract_all_features(
        data,
        fs=target_sr,
        channel_names=[sensor_prefix],
        sensor_prefix=sensor_prefix,
    )

    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, f'data_{version}.npy'), data)
    np.save(os.path.join(save_path, f'label_{version}.npy'), labels)
    np.save(os.path.join(save_path, f'all_features_{version}.npy'), features)
    np.save(os.path.join(save_path, f'all_features_name_{version}.npy'), feature_names)

    print(f'Data shape: {data.shape}')
    print(f'Label shape: {labels.shape}')
    print(f'Feature shape: {features.shape}')
    print(f'Feature name count: {len(feature_names)}')
    print('Saved all files successfully.')

    return data, labels, features, feature_names


if __name__ == '__main__':
    # activity, user
    path_save = r'uci_windows'
    version = r'20_120'
    data, label, features, feature_names = preprocess(
        DATASET_PATH,
        path_save,
        version,
        raw_sr=50,
        target_sr=20,
        seq_len=120,
        sensor_prefix='phone',
    )
