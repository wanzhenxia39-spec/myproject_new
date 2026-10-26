from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_initial9_gzsl_temperature_only as driver
from zshar.evaluation.calibration import calibrated_scores, tune_calibration
from zshar.pipeline.training import split_train_validation


def test_temperature_selection_changes_confidence_but_never_applies_seen_penalty():
    scores = pd.DataFrame({"seen": [3.0, 0.0], "unseen": [0.0, 3.0]})
    fold = {"scores": scores, "truth": ["seen", "unseen"],
            "pseudo_seen": ["seen"], "pseudo_unseen": ["unseen"]}
    result = tune_calibration([fold], mode="grid", gamma_values=[-1, 0, 1],
        temperature_grid=[0.1, 1, 10], tpe_trials=1, tau_min=0.01,
        tau_max=10, seed=42, bias_correction="temperature_only")
    assert result["best_gamma"] == 0
    assert result["best_temperature"] == 0.1
    assert result["selection_metric"] == "mean_validation_negative_log_likelihood"
    for temperature in driver.TEMPERATURES:
        adjusted = calibrated_scores(scores, ["seen"], gamma=0,
            temperature=temperature, bias_correction="temperature_only")
        pd.testing.assert_series_equal(adjusted.idxmax(axis=1), scores.idxmax(axis=1))
    with pytest.raises(ValueError, match="gamma must be 0.0"):
        calibrated_scores(scores, ["seen"], gamma=0.1, bias_correction="temperature_only")


def test_resume_refuses_changed_vectors_and_untracked_models(tmp_path):
    protocol = {"input_sha256": {"vectors": "new"}, "config": {"gamma": 0}}
    driver.write(tmp_path / "input_protocol.json", protocol)
    driver.validate_existing_run(tmp_path, protocol, True, "train-test")
    with pytest.raises(FileExistsError):
        driver.validate_existing_run(tmp_path, protocol, False, "train-test")
    changed = deepcopy(protocol)
    changed["input_sha256"]["vectors"] = "old"
    with pytest.raises(ValueError, match="Inputs/configuration changed"):
        driver.validate_existing_run(tmp_path, changed, True, "train-test")
    (tmp_path / "input_protocol.json").unlink()
    (tmp_path / "split_013").mkdir()
    with pytest.raises(ValueError, match="lack the revised-vector input signature"):
        driver.validate_existing_run(tmp_path, protocol, True, "train-test")


def test_stale_constant_regularity_models_are_rejected(tmp_path):
    frame = pd.DataFrame({"activity_name": ["a"] * 20 + ["b"] * 20,
                          "motion_regularity": ["High"] * 20 + ["Medium"] * 20})
    fit, _ = split_train_validation(frame, seed=42)
    manifest = {"manager_type": "independent", "classifier_backend": "autogluon",
        "attribute_cols": ["motion_regularity"],
        "class_counts": {"motion_regularity": {"High": len(fit)}}}
    driver.write(tmp_path / driver.MANIFEST, manifest)
    with pytest.raises(ValueError, match="targets differ"):
        driver.validate_model_manifest(tmp_path, frame, ["motion_regularity"], 42)
    manifest["class_counts"]["motion_regularity"] = dict(fit.motion_regularity.value_counts())
    # JSON serialization must also work with ordinary Python counts.
    manifest["class_counts"]["motion_regularity"] = {
        k: int(v) for k, v in manifest["class_counts"]["motion_regularity"].items()}
    driver.write(tmp_path / driver.MANIFEST, manifest)
    driver.validate_model_manifest(tmp_path, frame, ["motion_regularity"], 42)


def test_system_disk_outputs_are_rejected():
    with pytest.raises(ValueError):
        driver.data_output("/home/wanzhen/forbidden_gzsl_outputs")


def test_incomplete_models_are_preserved_without_touching_complete_models(tmp_path):
    partial = tmp_path / "split_013/calibration/fold_01/models"
    partial.mkdir(parents=True)
    (partial / "keep.txt").write_text("partial")
    complete = tmp_path / "split_013/models"
    driver.write(complete / driver.MANIFEST, {"complete": True})
    driver.preserve_partial_models(tmp_path)
    assert (partial.with_name("models.incomplete") / "keep.txt").read_text() == "partial"
    assert (complete / driver.MANIFEST).exists()


def test_report_excludes_ausuc_and_checks_actual_gamma(tmp_path):
    vector_file = tmp_path / "vectors.json"
    vector_file.write_text("{}")
    row = {"split_index": 13, "gamma": 0, "temperature": 0.3, "attribute_count": 9,
        "gzsl_bias_correction": "temperature_only", "backend": "autogluon",
        "attribute_vectors_sha256": driver.sha(vector_file)}
    for name in ("seen_accuracy", "unseen_accuracy", "harmonic_mean", "overall_accuracy",
                 "activity_top1_accuracy", "activity_top3_recall", "activity_macro_f1"):
        row[name] = 0.5
    pd.DataFrame([row]).to_csv(tmp_path / "summary.csv", index=False)
    metrics = {"gamma": 0, "calibration": {"best_gamma": 0,
        "selection_metric": "mean_validation_negative_log_likelihood", "best_mean_nll": 0.7}}
    driver.write(tmp_path / "split_013/metrics.json", metrics)
    config = SimpleNamespace(split_indices={13}, dataset="mhealth")
    protocol = {"vectors": str(vector_file)}
    driver.summarize(tmp_path, config, protocol)
    report = driver.read(tmp_path / "gzsl_report.json")
    assert "ausuc" not in [r["metric"] for r in report["metrics"]]
    metrics["calibration"]["best_gamma"] = 0.1
    driver.write(tmp_path / "split_013/metrics.json", metrics)
    with pytest.raises(ValueError, match="gamma fixed at zero"):
        driver.summarize(tmp_path, config, protocol)
