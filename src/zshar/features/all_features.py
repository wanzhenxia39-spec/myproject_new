# 整合精简后的统计特征代码以及《A Feature Selection-Based Framework for Human Activity Recognition Using Wearable Multimodal Sensors》中提到的物理特征
# 修改点：
# 1) 物理特征中的 ENERGY / AAE / ARE 改为 log1p 压缩后的能量，缓解极端值问题
# 2) 重力去除与重力角特征的低通截止频率改为可配置，默认 0.2 Hz（原先常用 0.3/0.1 的固定值）
# 提取上述两种特征后保存到.npy文件中

from __future__ import annotations

"""
Complete feature extraction for one IMU window.

This module merges statistical-feature extraction code with
paper-aligned physical-feature extraction for HAR.

Main entry:
    extract_features(sensor_data, fs, channel_names=None)

Expected input format:
    sensor_data = {
        "waist_acc": np.ndarray shape (3, T),
        "waist_gyro": np.ndarray shape (3, T),
        ...
    }

Output:
    Standard Python dict mapping feature_name -> float for one window.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable
from collections import Counter

import numpy as np
from scipy.signal import welch, stft, butter, filtfilt
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform


# ============================================================
# Statistical / signal-processing features
# ============================================================

def extract_time_features(x: np.ndarray, name: str, axis: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    feats[f"{name}_{axis}_mean"] = float(np.mean(x))
    feats[f"{name}_{axis}_std"] = float(np.std(x))
    feats[f"{name}_{axis}_rms"] = float(np.sqrt(np.mean(x ** 2)))
    feats[f"{name}_{axis}_range"] = float(np.max(x) - np.min(x))
    feats[f"{name}_{axis}_sma"] = float(np.mean(np.abs(x)))

    t = np.arange(len(x))
    denom = ((t - t.mean()) ** 2).sum()
    slope = ((t - t.mean()) * (x - x.mean())).sum() / denom if denom > 0 else 0.0
    feats[f"{name}_{axis}_slope"] = float(slope)
    return feats


def extract_freq_features(x: np.ndarray, name: str, axis: str, fs: float) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))

    total_power = float(np.trapz(psd, freqs) + 1e-12)

    bands = {"low": (0, 0.5), "mid": (0.5, 3), "high": (3, min(15, freqs[-1]))}
    for b, (l, h) in bands.items():
        mask = (freqs >= l) & (freqs <= h)
        bp = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
        feats[f"{name}_{axis}_bp_fft_{b}_ratio"] = bp / total_power

    dom_idx = int(np.argmax(psd[1:]) + 1) if len(psd) > 1 else 0
    feats[f"{name}_{axis}_fft_dom_freq"] = float(freqs[dom_idx])

    probs = psd / (np.sum(psd) + 1e-12)
    feats[f"{name}_{axis}_fft_sp_entropy"] = float(entropy(probs))
    feats[f"{name}_{axis}_fft_energy"] = float(np.trapz(psd ** 2, freqs))

    weighted_freq = float(np.sum(freqs * psd) / total_power)
    feats[f"{name}_{axis}_fft_weighted_avg_freq"] = weighted_freq

    return feats


def extract_stft_features(x: np.ndarray, name: str, axis: str, fs: float) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    f, _t, Z = stft(x, fs=fs, nperseg=min(128, len(x)))
    psd = np.abs(Z) ** 2

    mask = (f >= 0.5) & (f < 3)
    energy = psd[mask].sum(axis=0) if np.any(mask) else np.zeros(psd.shape[1])
    feats[f"{name}_{axis}_stft_mid_mean"] = float(energy.mean()) if energy.size else 0.0

    p = psd / (psd.sum(axis=0, keepdims=True) + 1e-12)
    ent = -np.sum(p * np.log(p + 1e-12), axis=0)
    feats[f"{name}_{axis}_stft_ent_mean"] = float(ent.mean()) if ent.size else 0.0

    centroid = np.sum(f[:, None] * psd, axis=0) / (psd.sum(axis=0) + 1e-12)
    feats[f"{name}_{axis}_stft_centroid_mean"] = float(centroid.mean()) if centroid.size else 0.0

    return feats


def extract_jerk(x: np.ndarray, name: str, axis: str, fs: float) -> Dict[str, float]:
    dt = 1.0 / fs
    jerk = np.diff(x) / dt
    feats: Dict[str, float] = {}
    if len(jerk) > 0:
        feats[f"{name}_{axis}_jerk_rms"] = float(np.sqrt(np.mean(jerk ** 2)))
        feats[f"{name}_{axis}_jerk_peak"] = float(np.max(np.abs(jerk)))
    return feats


def extract_acf(x: np.ndarray, name: str, axis: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[len(x) - 1:]
    corr = corr / (corr[0] + 1e-12)

    peak = -1
    for i in range(1, len(corr) - 1):
        if corr[i] > corr[i - 1] and corr[i] > corr[i + 1]:
            peak = i
            break

    feats[f"{name}_{axis}_acf_first_peak_lag"] = float(peak)
    return feats


def permutation_entropy(x: np.ndarray, m: int = 3) -> float:
    counts = Counter()
    for i in range(len(x) - m + 1):
        pattern = tuple(np.argsort(x[i:i + m]))
        counts[pattern] += 1

    probs = np.array(list(counts.values()), dtype=float)
    if probs.size == 0:
        return 0.0
    probs /= probs.sum() + 1e-12
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def recurrence_rate(x: np.ndarray) -> float:
    x = x.reshape(-1, 1)
    D = squareform(pdist(x))
    eps = 0.1 * np.std(x)
    R = (D <= eps).astype(int)
    N = len(x)
    return float((R.sum() - N) / (N * (N - 1) + 1e-12))


def lowpass(x: np.ndarray, cutoff: float, fs: float, order: int = 3) -> np.ndarray:
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def extract_gravity_features(
    acc: np.ndarray,
    name: str,
    fs: float,
    cutoff_hz: float = 0.2,
    order: int = 3,
) -> Dict[str, float]:
    """
    Extra gravity-angle features from the original extended code.
    Modified so the cutoff is configurable; default is 0.2 Hz.
    """
    feats: Dict[str, float] = {}
    gx = lowpass(acc[0], cutoff_hz, fs, order)
    gy = lowpass(acc[1], cutoff_hz, fs, order)
    gz = lowpass(acc[2], cutoff_hz, fs, order)

    g_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-12
    theta = np.arccos(np.clip(gz / g_mag, -1, 1))

    feats[f"{name}_z_grav_angle_mean"] = float(np.mean(theta))
    feats[f"{name}_z_grav_angle_std"] = float(np.std(theta))
    return feats


# ======================================
# Paper-aligned physical feature section
# ======================================

@dataclass
class PhysicalFeatureConfig:
    fs: float = 100.0
    gravity_lp_cutoff_hz: float = 0.2  # modified default: configurable in 0.1~0.3 Hz range
    butter_order: int = 3
    gravity_axis: int = 0
    heading_axes: Tuple[int, int] = (1, 2)
    eps: float = 1e-12
    demean_before_integration: bool = True
    log_energy: bool = True


def _validate_imu_window(acc: np.ndarray, gyro: Optional[np.ndarray] = None) -> None:
    if acc.ndim != 2 or acc.shape[0] != 3:
        raise ValueError(f"`acc` must have shape (3, T), got {acc.shape}")
    if gyro is not None and (gyro.ndim != 2 or gyro.shape[0] != 3 or gyro.shape[1] != acc.shape[1]):
        raise ValueError(
            f"`gyro` must have shape (3, T) and match acc length; got {gyro.shape}, acc {acc.shape}"
        )
    if acc.shape[1] < 2:
        raise ValueError("Window length T must be at least 2.")


def estimate_gravity(acc: np.ndarray, cfg: PhysicalFeatureConfig) -> np.ndarray:
    return np.vstack([
        lowpass(acc[i], cfg.gravity_lp_cutoff_hz, cfg.fs, cfg.butter_order)
        for i in range(3)
    ])


def remove_gravity(acc: np.ndarray, cfg: PhysicalFeatureConfig) -> np.ndarray:
    return acc - estimate_gravity(acc, cfg)


def cumulative_trapezoid_zero_start(x: np.ndarray, dt: float) -> np.ndarray:
    v = np.zeros_like(x, dtype=float)
    if x.size >= 2:
        v[1:] = np.cumsum((x[:-1] + x[1:]) * 0.5 * dt)
    return v


def movement_intensity_series(acc_dyn: np.ndarray) -> np.ndarray:
    return np.linalg.norm(acc_dyn, axis=0)


def extract_physical_features_for_sensor_pair(
    acc: np.ndarray,
    gyro: np.ndarray,
    prefix: str,
    cfg: PhysicalFeatureConfig,
) -> Dict[str, float]:
    _validate_imu_window(acc, gyro)
    feats: Dict[str, float] = {}

    # 1) AI / VI from dynamic acceleration magnitude
    acc_dyn = remove_gravity(acc, cfg)
    mi = movement_intensity_series(acc_dyn)
    feats[f"{prefix}_AI"] = float(np.mean(mi))
    feats[f"{prefix}_VI"] = float(np.var(mi))

    # 2) SMA (paper's tri-axial acceleration SMA)
    T = acc.shape[1]
    feats[f"{prefix}_SMA_phys"] = float(
        (np.sum(np.abs(acc[0])) + np.sum(np.abs(acc[1])) + np.sum(np.abs(acc[2]))) / T
    )

    # 3) EVA: top two covariance eigenvalues of acceleration
    cov = np.cov(acc)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals)[::-1]
    feats[f"{prefix}_EVA_1"] = float(evals[0])
    feats[f"{prefix}_EVA_2"] = float(evals[1])

    # 4) CAGH: corr(acc along gravity, derived heading magnitude)
    g_idx = cfg.gravity_axis
    h1, h2 = cfg.heading_axes
    a_gravity = acc[g_idx]
    a_heading = np.sqrt(acc[h1] ** 2 + acc[h2] ** 2)
    if np.std(a_gravity) < cfg.eps or np.std(a_heading) < cfg.eps:
        feats[f"{prefix}_CAGH"] = 0.0
    else:
        feats[f"{prefix}_CAGH"] = float(np.corrcoef(a_gravity, a_heading)[0, 1])

    # 5) AVG / AVH via numerical integration
    dt = 1.0 / cfg.fs
    acc_used = acc.copy()
    if cfg.demean_before_integration:
        acc_used = acc_used - np.mean(acc_used, axis=1, keepdims=True)

    v_g = cumulative_trapezoid_zero_start(acc_used[g_idx], dt)
    v_h1 = cumulative_trapezoid_zero_start(acc_used[h1], dt)
    v_h2 = cumulative_trapezoid_zero_start(acc_used[h2], dt)

    feats[f"{prefix}_AVG"] = float(np.mean(v_g))
    feats[f"{prefix}_AVH"] = float(np.sqrt(np.mean(v_h1) ** 2 + np.mean(v_h2) ** 2))

    # 6) ARATG: averaged rotation angle around gravity direction
    omega_g = gyro[g_idx]
    angle_g = cumulative_trapezoid_zero_start(omega_g, dt)
    feats[f"{prefix}_ARATG"] = float(np.sum(angle_g) / len(angle_g))

    # 7) DF / ENERGY per axis, then AAE / ARE
    acc_energy = []
    gyro_energy = []
    axis_names = ["x", "y", "z"]

    def dominant_frequency_and_energy_1axis(x: np.ndarray) -> Tuple[float, float]:
        N = x.size
        fft_vals = np.fft.rfft(x)
        mags2 = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(N, d=1.0 / cfg.fs)
        if mags2.size > 0:
            mags2_no_dc = mags2.copy()
            mags2_no_dc[0] = 0.0
        else:
            mags2_no_dc = mags2
        dom_freq = 0.0 if np.all(mags2_no_dc <= cfg.eps) else float(freqs[int(np.argmax(mags2_no_dc))])
        raw_energy = float(np.sum(mags2_no_dc) / N)
        energy = float(np.log1p(raw_energy)) if cfg.log_energy else raw_energy
        return dom_freq, energy

    for i, ax_name in enumerate(axis_names):
        df_a, en_a = dominant_frequency_and_energy_1axis(acc[i])
        feats[f"{prefix}_ACC_{ax_name.upper()}_DF"] = df_a
        feats[f"{prefix}_ACC_{ax_name.upper()}_ENERGY_phys"] = en_a
        acc_energy.append(en_a)

        df_g, en_g = dominant_frequency_and_energy_1axis(gyro[i])
        feats[f"{prefix}_GYRO_{ax_name.upper()}_DF"] = df_g
        feats[f"{prefix}_GYRO_{ax_name.upper()}_ENERGY_phys"] = en_g
        gyro_energy.append(en_g)

    feats[f"{prefix}_AAE"] = float(np.mean(acc_energy))
    feats[f"{prefix}_ARE"] = float(np.mean(gyro_energy))
    return feats


# =========================================
# Utilities for pairing acc / gyro channels
# =========================================

def _strip_sensor_suffix(name: str) -> Tuple[str, Optional[str]]:
    lowered = name.lower()
    if lowered.endswith("_acc"):
        return name[:-4], "acc"
    if lowered.endswith("_gyro"):
        return name[:-5], "gyro"
    return name, None


def _infer_prefixes(sensor_data: Dict[str, np.ndarray], channel_names: Optional[Iterable[str]]) -> Iterable[str]:
    if channel_names is not None:
        return list(channel_names)
    prefixes = []
    seen = set()
    for name in sensor_data.keys():
        prefix, suffix = _strip_sensor_suffix(name)
        if suffix is not None and prefix not in seen:
            seen.add(prefix)
            prefixes.append(prefix)
    return prefixes


# ==========
# Main entry
# ==========

def extract_features(
    sensor_data: Dict[str, np.ndarray],
    fs: float,
    channel_names: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """
    Generate a complete feature vector for one window.
    """
    feats: Dict[str, float] = {}
    cfg = PhysicalFeatureConfig(fs=fs)

    # 1) Statistical / signal features for every provided sensor channel
    for name, sensor in sensor_data.items():
        if sensor.ndim != 2 or sensor.shape[0] != 3:
            raise ValueError(f"Sensor '{name}' must have shape (3, T), got {sensor.shape}")

        mag = np.linalg.norm(sensor, axis=0)

        for i, axis in enumerate(["x", "y", "z"]):
            x = sensor[i]
            feats.update(extract_time_features(x, name, axis))
            feats.update(extract_freq_features(x, name, axis, fs))
            feats.update(extract_stft_features(x, name, axis, fs))
            feats.update(extract_jerk(x, name, axis, fs))
            feats.update(extract_acf(x, name, axis))
            feats[f"{name}_{axis}_pe"] = permutation_entropy(x)
            feats[f"{name}_{axis}_rr"] = recurrence_rate(x)

        feats.update(extract_time_features(mag, name, "mag"))
        feats.update(extract_freq_features(mag, name, "mag", fs))

        if "acc" in name.lower():
            feats.update(
                extract_gravity_features(
                    sensor,
                    name,
                    fs,
                    cutoff_hz=cfg.gravity_lp_cutoff_hz,
                    order=cfg.butter_order,
                )
            )

    # 2) Cross-modal statistical ratio from original uploaded code
    prefixes = list(_infer_prefixes(sensor_data, channel_names))
    for prefix in prefixes:
        acc_key = f"{prefix}_acc_mag_rms"
        gyro_key = f"{prefix}_gyro_mag_rms"
        if acc_key in feats and gyro_key in feats:
            feats[f"{prefix}_acc_gyro_ratio"] = feats[acc_key] / (feats[gyro_key] + 1e-12)

    # 3) Paper-aligned physical features for each available acc/gyro pair
    for prefix in prefixes:
        acc_name = f"{prefix}_acc"
        gyro_name = f"{prefix}_gyro"
        if acc_name in sensor_data and gyro_name in sensor_data:
            feats.update(
                extract_physical_features_for_sensor_pair(
                    sensor_data[acc_name],
                    sensor_data[gyro_name],
                    prefix=prefix,
                    cfg=cfg,
                )
            )

    return feats

# 以下代码并不直接执行，而是提供了一个简单的测试示例，具体使用时会根据数据集情况传递sensor_data和channel_names的
# if __name__ == "__main__":
#     fs = 100.0
#     T = 200
#     rng = np.random.default_rng(42)

#     sensor_data = {
#         "waist_acc": rng.normal(size=(3, T)),
#         "waist_gyro": rng.normal(size=(3, T)),
#     }

#     feats = extract_features(sensor_data, fs=fs)
#     print(f"Extracted {len(feats)} features.")
#     for i, (k, v) in enumerate(feats.items()):
#         if i >= 20:
#             break
#         print(k, v)
