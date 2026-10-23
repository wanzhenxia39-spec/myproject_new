"""Exercise raw readers, window labels and the feature schema together."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from zshar.features.all_features import extract_features
from zshar.features.groups import expand_feature_groups, load_groups, resolve_feature_groups
from zshar.preprocessing import reextract_mhealth_features as mhealth_reextract
from zshar.preprocessing.cli import main


@pytest.mark.parametrize("dataset,dimension", [
    ("pamap2", 534), ("mhealth", 434), ("dsads", 890), ("uschad", 178), ("wisdm", 356),
])
def test_raw_to_features_and_stable_activity_labels(tmp_path, dataset, dimension):
    raw, output = tmp_path / "raw", tmp_path / "output"
    raw.mkdir()
    rng = np.random.default_rng(23)
    # Start at a non-first activity to catch encounter-order label remapping.
    expected_id = 4 if dataset == "pamap2" else 3
    if dataset == "pamap2":
        values = rng.normal(size=(600, 54))
        values[:, 1] = 4
        np.savetxt(raw / "subject101.dat", values)
    elif dataset == "mhealth":
        values = rng.normal(size=(302, 24))
        values[:, 23] = 4
        np.savetxt(raw / "mHealth_subject1.log", values)
    elif dataset == "dsads":
        folder = raw / "a04/p1"
        folder.mkdir(parents=True)
        for i in range(2):
            np.savetxt(folder / f"s{i+1:02d}.txt", rng.normal(size=(125, 45)), delimiter=",")
    elif dataset == "uschad":
        folder = raw / "Subject1"
        folder.mkdir()
        savemat(folder / "a4t1.mat", {"sensor_readings": rng.normal(size=(600, 6)),
                                    "activity_number": 4, "subject": 1, "trial": 1})
    else:
        for device in ("phone", "watch"):
            for sensor in ("accel", "gyro"):
                folder = raw / device / sensor
                folder.mkdir(parents=True)
                values = rng.normal(size=(120, 3))
                (folder / f"data_1600_{sensor}_{device}.txt").write_text("".join(
                    f"1600,D,{i * 50000000},{x},{y},{z};\n"
                    for i, (x, y, z) in enumerate(values)))
    main(["--dataset", dataset, "--data-path", str(raw), "--save-path", str(output)])
    features = np.load(output / "all_features_20_120.npy")
    names = np.load(output / "all_features_name_20_120.npy", allow_pickle=True)
    labels = np.load(output / "label_20_120.npy")
    windows = np.load(output / "data_20_120.npy")
    assert features.shape == (len(windows), dimension)
    assert np.isfinite(features).all()
    assert set(labels[:, :, 0].ravel()) == {expected_id}
    assert set(names) == set(expand_feature_groups(dataset)["feature_names"])
    resolved = resolve_feature_groups(names, dataset=dataset)
    assert not resolved["missing_features"]
    assert set(resolved["group_to_features"]) == {f"G{i}" for i in range(1, 15)}
    assert all(resolved["group_to_features"].values())
    if dataset == "mhealth":
        assert "hand_VI" in names and "ankle_VI" in names
        assert not any(n.startswith("chest_gyro_") for n in names)


def test_prefix_override_and_partial_modalities():
    result = expand_feature_groups("mhealth", prefix_overrides={"hand": "right_lower_arm"})
    names = result["feature_names"]
    assert "right_lower_arm_acc_x_mean" in names
    assert "right_lower_arm_ARATG" in names
    assert "hand_ARATG" not in names
    assert "chest_ARATG" not in names
    assert "chest_acc_z_grav_angle_std" in names
    with pytest.raises(ValueError, match="unique"):
        expand_feature_groups("mhealth", prefix_overrides={"hand": "chest"})


def test_groups_cover_every_extractor_feature_for_all_configured_datasets():
    config = load_groups()
    rng = np.random.default_rng(31)
    for dataset, sensors in config["dataset_sensors"].items():
        sensor_data = {
            f"{prefix}_{modality}": rng.normal(size=(3, 120))
            for prefix, sensor in sensors.items()
            for modality in sensor["modalities"]
        }
        extracted = extract_features(
            sensor_data, fs=20, channel_names=list(sensors)
        )
        grouped = expand_feature_groups(dataset, config)["feature_names"]
        assert len(extracted) == len(grouped)
        assert set(extracted) == set(grouped)


def test_pamap_windows_do_not_cross_activity_boundaries():
    module = importlib.import_module("zshar.preprocessing.preprocess_pamap2")
    data, labels = [], []
    for activity in (4, 5):
        module._append_block(data, labels, np.ones((700, 18)), activity, 0, 5, 120)
    assert [x.shape[0] for x in data] == [1, 1]
    assert [int(x[0, 0, 0]) for x in labels] == [4, 5]


def test_mhealth_reextracts_every_saved_window_and_backs_up_incomplete(tmp_path):
    rng = np.random.default_rng(23)
    windows = rng.normal(size=(3, 120, 15)).astype(np.float32)
    labels = np.zeros((3, 120, 2), dtype=np.int64)
    windows_path = tmp_path / "data_20_120.npy"
    labels_path = tmp_path / "label_20_120.npy"
    features_path = tmp_path / "all_features_20_120.npy"
    names_path = tmp_path / "all_features_name_20_120.npy"
    np.save(windows_path, windows)
    np.save(labels_path, labels)
    np.save(features_path, np.zeros((3, 386), dtype=np.float32))
    np.save(names_path, np.asarray([f"old_{i}" for i in range(386)], dtype=object))

    report = mhealth_reextract.reextract_all_windows(
        windows_path=windows_path,
        labels_path=labels_path,
        features_path=features_path,
        feature_names_path=names_path,
        fs=20,
        checkpoint_every=2,
    )

    features = np.load(features_path)
    names = np.load(names_path, allow_pickle=True)
    assert features.shape == (len(windows), 434)
    assert names.shape == (434,)
    assert np.isfinite(features).all()
    assert set(names) == set(expand_feature_groups("mhealth")["feature_names"])
    assert report["completed_windows"] == len(windows)
    assert Path(report["incomplete_feature_backup"]).is_file()
    assert Path(report["incomplete_name_backup"]).is_file()
    assert not (tmp_path / "all_features_20_120.npy.partial.npy").exists()
    assert not (tmp_path / "all_features_20_120.npy.progress.json").exists()


def test_mhealth_reextract_can_resume_after_interrupted_window(tmp_path, monkeypatch):
    rng = np.random.default_rng(29)
    windows = rng.normal(size=(3, 120, 15)).astype(np.float32)
    windows_path = tmp_path / "data.npy"
    features_path = tmp_path / "features.npy"
    names_path = tmp_path / "names.npy"
    np.save(windows_path, windows)
    original = mhealth_reextract._extract_one
    calls = {"count": 0}

    def interrupted(window, fs):
        calls["count"] += 1
        if calls["count"] == 4:  # schema probe, windows 0/1, then fail on window 2
            raise RuntimeError("interrupted")
        return original(window, fs)

    monkeypatch.setattr(mhealth_reextract, "_extract_one", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        mhealth_reextract.reextract_all_windows(
            windows_path=windows_path, features_path=features_path,
            feature_names_path=names_path, checkpoint_every=1,
        )
    progress = json.loads((tmp_path / "features.npy.progress.json").read_text())
    assert progress["completed_windows"] == 2

    monkeypatch.setattr(mhealth_reextract, "_extract_one", original)
    mhealth_reextract.reextract_all_windows(
        windows_path=windows_path, features_path=features_path,
        feature_names_path=names_path, checkpoint_every=1, resume=True,
    )
    assert np.load(features_path).shape == (3, 434)
